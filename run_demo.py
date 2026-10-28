"""Step 6 of the HUBRIS pipeline: render the annotated demo video.

Usage:
    python run_demo.py --sync-dir out/video1 --video ../varjo_capture1.mp4               # 60 s highlight
    python run_demo.py --sync-dir out/video1 --video ../varjo_capture1.mp4 --whole-drive # ~10 min

Needs the outputs of steps 1-5. Writes an H.264 .mp4 (plays in PowerPoint, browsers and Windows) through the
ffmpeg that comes with the imageio-ffmpeg package. The default clip, 10:05-11:05, is the minute with the most
varied gaze: trucks, cars, road lines, the speed display and the rating popup.

Output: <sync-dir>/demo_<start>-<end>.mp4 ('_full' added for --style full)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import cv2
import numpy as np
import pandas as pd

from hubris import demo, label, road
from hubris.video import read_frames, video_info

CORNERS = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sync-dir", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--start-frame", type=int, default=18150)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--whole-drive", action="store_true", help="render frames 2,090-20,460 instead")
    ap.add_argument("--width", type=int, default=1280, help="output width (height keeps 4:3)")
    ap.add_argument("--style", choices=["clean", "full"], default="clean",
                    help="clean: highlight only what the driver looks at; full: draw everything found")
    args = ap.parse_args()
    d = args.sync_dir

    info = video_info(args.video)
    if args.whole_drive:
        start, end = label.DRIVE_FRAMES
    else:
        start, end = args.start_frame, min(args.start_frame + int(args.seconds * info.fps), info.n_frames) - 1
    frames = np.arange(start, end + 1)

    cal = json.load(open(f"{d}/calibration.json"))
    radius = cal["px_per_deg"] * label.RADIUS_DEG
    id2label = {int(k): v for k, v in json.load(open(f"{d}/seg_m2f.json"))["id2label"].items()}
    summary = demo.frame_summary(pd.read_parquet(f"{d}/gaze_labels.parquet"))
    boxes = pd.read_parquet(f"{d}/detections_yolo11l_tracked.parquet")
    dropped = json.load(open(f"{d}/gaze_labels.json")).get("dropped_interior_tracks", [])
    boxes = boxes[(boxes.track_len >= label.MIN_TRACK_LEN) & boxes.frame.between(start, end)
                  & ~boxes.track_id.isin(dropped)]
    by_frame = {f: g for f, g in boxes.groupby("frame")}
    sp = pd.read_csv(f"{d}/speed_display.csv")
    quads = {int(f): q for f, q in zip(sp.frame, sp[CORNERS].to_numpy(float)) if np.isfinite(q).all()}
    strip = demo.timeline_strip(frames, summary)
    legend = [l for l in demo.LABEL_COLORS if l in set(summary.label.reindex(frames).dropna())]

    import imageio_ffmpeg
    size = (args.width, args.width * info.height // info.width)
    out_path = f"{d}/demo_{start}-{end}" + ("_full" if args.style == "full" else "") + ".mp4"
    ff = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}", "-r", f"{info.fps}",
                           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "medium",
                           "-movflags", "+faststart", out_path], stdin=subprocess.PIPE)
    print(f"rendering frames {start}-{end} ({len(frames) / info.fps:.0f} s) -> {out_path}")
    t0 = time.time()
    empty = boxes.iloc[:0]
    layers = None
    for i, (fid, img) in enumerate(read_frames(args.video, frames)):
        # Frames without gaze (blinks) have no road map; keep the last one so the overlay doesn't flicker.
        seg_path = f"{d}/seg_m2f/{fid:06d}.png"
        if os.path.exists(seg_path):
            layers = road.road_layers(cv2.imread(seg_path, cv2.IMREAD_UNCHANGED), id2label)
        if layers is not None and args.style == "full":
            road.draw_layers(img, layers, alpha=0.25)
        row = summary.loc[fid] if fid in summary.index else None
        demo.draw_frame(img, row, by_frame.get(fid, empty), quads.get(fid), radius, fid / info.fps, fid,
                        layers=layers, style=args.style)
        demo.draw_legend(img, legend)
        demo.draw_timeline(img, strip, i / max(len(frames) - 1, 1))
        ff.stdin.write(cv2.resize(img, size, interpolation=cv2.INTER_AREA).tobytes())
        if (i + 1) % 600 == 0:
            print(f"{i + 1}/{len(frames)} frames, {(i + 1) / (time.time() - t0):.1f} frames/s", flush=True)
    ff.stdin.close()
    ff.wait()
    print(f"done -> {out_path}")


if __name__ == "__main__":
    main()
