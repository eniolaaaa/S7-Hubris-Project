"""Step 4c of the HUBRIS pipeline: track the car/truck boxes from run_detect.py over time.

Usage:
    python run_track.py --sync-dir out/video1 --video ../varjo_capture1.mp4

CPU only (~1 min): ByteTrack runs on the saved boxes, so YOLO is not rerun and the GPU is not used.
--video is only needed for the check clip.

Outputs in --sync-dir:
    detections_<model>_tracked.parquet|csv   one row per box: frame, track_id, category, class (one per track),
                                             class_frame (YOLO's class in that frame), coco_class, x1, y1, x2, y2,
                                             conf, interpolated (filled-in gap box), track_len
    detections_<model>_tracked.json          settings and track statistics
    check/track_clip.mp4                     short clip with the track ids and the gaze circle
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd

from hubris import overlay, track
from hubris.video import read_frames, video_info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sync-dir", required=True, help="output folder of run_sync.py / run_detect.py")
    ap.add_argument("--detections", default="detections_yolo11l", help="file stem written by run_detect.py")
    ap.add_argument("--max-gap", type=int, default=10, help="fill gaps of up to this many frames (30 fps)")
    ap.add_argument("--video", help="video, only for the check clip")
    ap.add_argument("--clip-start", type=int, default=9000, help="first frame of the check clip")
    ap.add_argument("--clip-seconds", type=float, default=15)
    args = ap.parse_args()

    stem = f"{args.sync_dir}/{args.detections}"
    dets = pd.read_parquet(f"{stem}.parquet").sort_values(["frame", "conf"], ascending=[True, False])
    dets = dets.reset_index(drop=True)
    first, last = json.load(open(f"{stem}.json"))["frame_range"]
    print(f"{len(dets)} boxes, frames {first}-{last}")

    dets["track_id"] = track.assign_track_ids(dets, first, last)
    out = track.finish_tracks(dets, max_gap=args.max_gap)
    out.to_parquet(f"{stem}_tracked.parquet", index=False)
    out.round({"x1": 1, "y1": 1, "x2": 1, "y2": 1, "conf": 3}).to_csv(f"{stem}_tracked.csv", index=False)

    real = out[~out.interpolated]
    lengths = real.groupby("track_id").size()
    report = {
        "tracker": "ByteTrack (ultralytics), defaults", "settings": track.BYTETRACK, "max_gap_filled": args.max_gap,
        "boxes_detected": int(len(real)), "boxes_without_track": int((dets.track_id < 0).sum()),
        "tracks": int(len(lengths)),
        "tracks_by_class": real.drop_duplicates("track_id")["class"].value_counts().to_dict(),
        "tracks_3plus_frames": int((lengths >= 3).sum()),
        "track_len_frames_median": float(lengths.median()),
        "track_len_frames_mean": float(lengths.mean()),
        "one_frame_tracks": int((lengths == 1).sum()),
        "boxes_class_changed_by_vote": int((real["class"] != real["class_frame"]).sum()),
        "boxes_interpolated": int(out.interpolated.sum()),
    }
    with open(f"{stem}_tracked.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

    if args.video:
        info = video_info(args.video)
        cal = json.load(open(f"{args.sync_dir}/calibration.json"))
        gaze = pd.read_csv(f"{args.sync_dir}/gaze_synced.csv")
        pts = overlay.frame_points(gaze, cal["lag_frames"])
        radius = cal["px_per_deg"] * 1.5
        os.makedirs(f"{args.sync_dir}/check", exist_ok=True)
        path = f"{args.sync_dir}/check/track_clip.mp4"
        size = (info.width // 2, info.height // 2)
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), info.fps, size)
        by_frame = dict(tuple(out.groupby("frame")))
        frames = range(args.clip_start, min(args.clip_start + int(args.clip_seconds * info.fps), info.n_frames))
        for fid, img in read_frames(args.video, frames):
            boxes = by_frame.get(fid, out.iloc[:0])[["track_id", "class", *track.BOX, "interpolated"]]
            for tid, cls, x1, y1, x2, y2, filled in boxes.itertuples(index=False):
                c = track.track_color(tid)
                p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                cv2.rectangle(img, p1, p2, c, 1 if filled else 3, cv2.LINE_AA)
                cv2.putText(img, f"{cls} #{tid}" + (" (filled)" if filled else ""),
                            (p1[0], max(p1[1] - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2, cv2.LINE_AA)
            if fid in pts.index:
                overlay.draw_gaze(img, pts.loc[fid].x, pts.loc[fid].y, radius)
            overlay._stamp(img, f"frame {fid}  t={fid / info.fps:7.2f}s   ByteTrack ids; thin box = filled-in gap")
            vw.write(cv2.resize(img, size, interpolation=cv2.INTER_AREA))
        vw.release()
        print(f"check clip -> {path}")
    print(f"done -> {stem}_tracked.parquet")


if __name__ == "__main__":
    main()
