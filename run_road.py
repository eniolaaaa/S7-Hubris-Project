"""Step 4b: segment road surface, road lines, road edge and guard rail.

Usage:
    python run_road.py --video ../varjo_capture1.mp4 --sync-dir out/video1

Runs Mask2Former (Mapillary Vistas) on every `display_frame` with a valid gaze sample inside the video,
from --start-frame on (foggy frames included; step 5 finds the drive), and saves the raw label map per
frame. `hubris.road.road_layers` turns a label map into the road classes.

The burned-in gaze cursor is inpainted first, at the positions in <sync-dir>/cursor_track.csv (step 3).

Resumable: one PNG per frame in <sync-dir>/seg_m2f/; frames that already have a PNG are skipped.

Outputs in --sync-dir:
    seg_m2f/<frame>.png          Vistas class id per pixel, 1024x768 (half the video size)
    seg_m2f.json                 model, settings, class names, coverage per layer
    check/road_frame_*.jpg       snapshots with the layers and the gaze circle
"""
from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import numpy as np
import pandas as pd

from hubris import overlay, road
from hubris.video import read_frames, video_info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--sync-dir", required=True, help="output folder of run_sync.py (gaze_synced.csv)")
    ap.add_argument("--start-frame", type=int, default=0, help="skip frames before this one")
    ap.add_argument("--end-frame", type=int, default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--snapshots", type=int, default=12)
    args = ap.parse_args()

    info = video_info(args.video)
    cal = json.load(open(f"{args.sync_dir}/calibration.json"))
    gaze = pd.read_csv(f"{args.sync_dir}/gaze_synced.csv")
    g = gaze[gaze.valid & gaze.inside_video].dropna(subset=["display_frame"])
    frames = np.unique(g.display_frame.astype(int))
    end = info.n_frames if args.end_frame is None else min(args.end_frame, info.n_frames)
    frames = frames[(frames >= args.start_frame) & (frames < end)]

    # Cursor positions found in step 3 for every frame; frames not in the track have no visible cursor.
    track_csv = f"{args.sync_dir}/cursor_track.csv"
    track = None
    if os.path.exists(track_csv):
        t = pd.read_csv(track_csv).drop_duplicates("frame").set_index("frame")
        track = dict(zip(t.index, zip(t.cx, t.cy)))

    seg_dir = f"{args.sync_dir}/seg_m2f"
    os.makedirs(seg_dir, exist_ok=True)
    png = lambda fid: f"{seg_dir}/{fid:06d}.png"
    todo = [f for f in frames if not os.path.exists(png(f))]
    print(f"{len(frames)} frames with gaze in [{args.start_frame}, {end}), {len(todo)} still to segment")

    seg_model = road.RoadSegmenter(device=args.device)
    t0, done = time.time(), 0

    def flush(ids, imgs):
        cursors = [track.get(f) for f in ids] if track is not None else None
        for fid, seg in zip(ids, seg_model.segment(imgs, cursors)):
            cv2.imwrite(png(fid) + ".tmp.png", seg)
            os.replace(png(fid) + ".tmp.png", png(fid))  # only complete files count as done

    ids, imgs = [], []
    for fid, img in read_frames(args.video, todo):
        ids.append(fid)
        imgs.append(img)
        if len(imgs) == args.batch:
            flush(ids, imgs)
            done += len(ids)
            ids, imgs = [], []
            if done % 500 < args.batch:
                fps = done / (time.time() - t0)
                print(f"{done}/{len(todo)} frames, {fps:.1f} frames/s, ~{(len(todo) - done) / fps / 60:.0f} min left",
                      flush=True)
    if imgs:
        flush(ids, imgs)
        done += len(ids)

    # Summary: share of the frame covered by each layer, over a sample of frames.
    sample = frames[np.linspace(0, len(frames) - 1, min(500, len(frames))).astype(int)]
    cover = {k: [] for k in road.LAYER_COLORS}
    for fid in sample:
        layers = road.road_layers(cv2.imread(png(fid), cv2.IMREAD_UNCHANGED), seg_model.id2label)
        for k, m in layers.items():
            cover[k].append(float(m.mean()))
    report = {
        "model": road.MODEL, "seg_size": road.SEG_SIZE,
        "frames_processed": int(len(frames)), "frame_range": [int(frames[0]), int(frames[-1])],
        "frames_per_s_this_run": done / (time.time() - t0) if done else None,
        "mean_frame_coverage_pct": {k: round(100 * float(np.mean(v)), 2) for k, v in cover.items()},
        "frames_with_layer_pct": {k: round(100 * float(np.mean(np.array(v) > 0)), 1) for k, v in cover.items()},
        "id2label": seg_model.id2label,
    }
    with open(f"{seg_dir}.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "id2label"}, indent=2))

    # Snapshots: layers + gaze circle.
    os.makedirs(f"{args.sync_dir}/check", exist_ok=True)
    pts = overlay.frame_points(gaze, cal["lag_frames"])
    radius = cal["px_per_deg"] * 1.5
    snap = frames[np.linspace(0, len(frames) - 1, args.snapshots).astype(int)]
    for fid, img in read_frames(args.video, snap):
        layers = road.road_layers(cv2.imread(png(fid), cv2.IMREAD_UNCHANGED), seg_model.id2label)
        road.draw_layers(img, layers)
        if fid in pts.index:
            overlay.draw_gaze(img, pts.loc[fid].x, pts.loc[fid].y, radius)
        overlay._stamp(img, f"frame {fid}  t={fid / info.fps:7.2f}s   purple = road surface, red = road line, "
                            f"yellow = road edge, blue = guard rail, magenta = gaze")
        cv2.imwrite(f"{args.sync_dir}/check/road_frame_{fid:06d}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"done -> {seg_dir}")


if __name__ == "__main__":
    main()
