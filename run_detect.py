"""Step 4 of the HUBRIS pipeline: YOLO11 car/truck detection on the frames that have gaze.

Usage:
    python run_detect.py --video ../varjo_capture1.mp4 --sync-dir out/video1

Reads gaze_synced.csv from --sync-dir and detects on every `display_frame` with a valid gaze sample
inside the video (from --start-frame on; frames 0-1400 are the foggy pre-drive scene).

Resumable: detections are written in chunks to <sync-dir>/detections_<model>/part_*.parquet and
finished chunks are skipped on a rerun, so a run that gets cut off can simply be started again.

Outputs in --sync-dir:
    detections_<model>.parquet|csv   one row per box: frame, category, class, coco_class, x1, y1, x2, y2, conf
    detections_<model>.json          settings, frames processed, counts, speed
    check/detect_frame_*.jpg         snapshots with boxes and the gaze circle
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import cv2
import numpy as np
import pandas as pd

from hubris import detect, overlay
from hubris.video import read_frames, video_info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--sync-dir", required=True, help="output folder of run_sync.py (gaze_synced.csv)")
    ap.add_argument("--model", default="yolo11l.pt", help="ultralytics weights, e.g. yolo11n.pt ... yolo11x.pt")
    ap.add_argument("--imgsz", type=int, default=1280, help="inference size (frames are 2048x1536)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--start-frame", type=int, default=1400, help="skip the pre-drive scene before this frame")
    ap.add_argument("--end-frame", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=1000, help="frames per saved part (resume granularity)")
    ap.add_argument("--device", default=None, help="e.g. 0 or cpu (default: GPU if available)")
    ap.add_argument("--snapshots", type=int, default=12)
    args = ap.parse_args()

    info = video_info(args.video)
    cal = json.load(open(f"{args.sync_dir}/calibration.json"))
    gaze = pd.read_csv(f"{args.sync_dir}/gaze_synced.csv")
    g = gaze[gaze.valid & gaze.inside_video].dropna(subset=["display_frame"])
    frames = np.unique(g.display_frame.astype(int))
    end = info.n_frames if args.end_frame is None else min(args.end_frame, info.n_frames)
    frames = frames[(frames >= args.start_frame) & (frames < end)]
    print(f"{len(frames)} frames with gaze in [{args.start_frame}, {end})")

    name = os.path.splitext(os.path.basename(args.model))[0]
    part_dir = f"{args.sync_dir}/detections_{name}"
    os.makedirs(part_dir, exist_ok=True)

    model = detect.load_model(args.model)
    fp16 = args.device != "cpu"
    chunks = [frames[i:i + args.chunk] for i in range(0, len(frames), args.chunk)]
    t0, n_run = time.time(), 0
    for ci, chunk in enumerate(chunks):
        path = f"{part_dir}/part_{chunk[0]:06d}_{chunk[-1]:06d}.parquet"
        if os.path.exists(path):
            continue
        parts, ids, imgs = [], [], []
        for fid, img in read_frames(args.video, chunk):
            ids.append(fid)
            imgs.append(img)
            if len(imgs) == args.batch:
                parts.append(detect.detect_batch(model, ids, imgs, args.imgsz, args.conf, args.device, fp16))
                ids, imgs = [], []
        if imgs:
            parts.append(detect.detect_batch(model, ids, imgs, args.imgsz, args.conf, args.device, fp16))
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=detect.COLS)
        df.to_parquet(path + ".tmp", index=False)
        os.replace(path + ".tmp", path)  # a part only counts as done once fully written
        n_run += len(chunk)
        fps = n_run / (time.time() - t0)
        print(f"chunk {ci + 1}/{len(chunks)}: frames {chunk[0]}-{chunk[-1]}, {len(df)} boxes, {fps:.1f} frames/s")

    parts = [pd.read_parquet(p) for p in sorted(glob.glob(f"{part_dir}/part_*.parquet"))]
    dets = pd.concat(parts, ignore_index=True).sort_values(["frame", "conf"], ascending=[True, False])
    dets = dets[dets.frame.isin(frames)]
    out = f"{args.sync_dir}/detections_{name}"
    dets.to_parquet(out + ".parquet", index=False)
    dets.round({"x1": 1, "y1": 1, "x2": 1, "y2": 1, "conf": 3}).to_csv(out + ".csv", index=False)

    per_frame = dets.groupby("frame").size().reindex(frames, fill_value=0)
    report = {
        "model": args.model, "imgsz": args.imgsz, "conf": args.conf,
        "frames_processed": int(len(frames)), "frame_range": [int(frames[0]), int(frames[-1])],
        "boxes": int(len(dets)),
        "boxes_by_class": dets["class"].value_counts().to_dict(),
        "boxes_by_coco_class": dets["coco_class"].value_counts().to_dict(),
        "frames_with_vehicle_pct": float(100 * (per_frame > 0).mean()),
        "boxes_per_frame_mean": float(per_frame.mean()),
        "conf_median": float(dets.conf.median()) if len(dets) else None,
        "frames_per_s_this_run": n_run / (time.time() - t0) if n_run else None,
    }
    with open(out + ".json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

    # Snapshots: boxes + gaze circle, spread over the processed frames.
    os.makedirs(f"{args.sync_dir}/check", exist_ok=True)
    pts = overlay.frame_points(gaze, cal["lag_frames"])
    radius = cal["px_per_deg"] * 1.5
    snap = frames[np.linspace(0, len(frames) - 1, args.snapshots).astype(int)]
    for fid, img in read_frames(args.video, snap):
        detect.draw_boxes(img, dets[dets.frame == fid])
        if fid in pts.index:
            overlay.draw_gaze(img, pts.loc[fid].x, pts.loc[fid].y, radius)
        overlay._stamp(img, f"frame {fid}  t={fid / info.fps:7.2f}s   {name} imgsz={args.imgsz}: "
                            f"green = car, orange = truck, magenta = gaze")
        cv2.imwrite(f"{args.sync_dir}/check/detect_frame_{fid:06d}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"done -> {out}.parquet")


if __name__ == "__main__":
    main()
