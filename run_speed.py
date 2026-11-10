"""Step 4d: find the speed display in every frame that has gaze.

Usage:
    python run_speed.py --video ../varjo_capture1.mp4 --sync-dir out/video1

CPU only. Uses the gaze cursor positions in <sync-dir>/cursor_track.csv to restore the end of the display's
light bar when the cursor covers it. Gaps of up to --max-gap frames are filled by interpolation.

Outputs in --sync-dir:
    speed_display.csv       one row per frame: frame, found, interpolated, bar_len, tilt_deg and the screen
                            corners x1,y1 (top-left) .. x4,y4 (bottom-left) in video pixels
    speed_display.json      settings and counts
    check/speed_frame_*.jpg snapshots with the screen outline and the gaze circle
"""
from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import numpy as np
import pandas as pd

from hubris import overlay, speed
from hubris.video import read_frames, video_info

CORNERS = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--sync-dir", required=True, help="output folder of run_sync.py (gaze_synced.csv)")
    ap.add_argument("--start-frame", type=int, default=0, help="skip frames before this one")
    ap.add_argument("--end-frame", type=int, default=None)
    ap.add_argument("--max-gap", type=int, default=5, help="fill gaps of up to this many frames")
    ap.add_argument("--snapshots", type=int, default=12)
    args = ap.parse_args()

    info = video_info(args.video)
    cal = json.load(open(f"{args.sync_dir}/calibration.json"))
    gaze = pd.read_csv(f"{args.sync_dir}/gaze_synced.csv")
    g = gaze[gaze.valid & gaze.inside_video].dropna(subset=["display_frame"])
    frames = np.unique(g.display_frame.astype(int))
    end = info.n_frames if args.end_frame is None else min(args.end_frame, info.n_frames)
    frames = frames[(frames >= args.start_frame) & (frames < end)]
    t = pd.read_csv(f"{args.sync_dir}/cursor_track.csv").drop_duplicates("frame").set_index("frame")
    track = dict(zip(t.index, zip(t.cx, t.cy)))
    print(f"{len(frames)} frames with gaze in [{args.start_frame}, {end})")

    rows, t0 = [], time.time()
    for i, (fid, img) in enumerate(read_frames(args.video, frames)):
        bar = speed.find_bar(img, track.get(fid))
        if bar is None:
            rows.append((fid, False, np.nan, np.nan, *[np.nan] * 8))
        else:
            rows.append((fid, True, bar["length"], bar["tilt"], *speed.display_quad(bar).ravel()))
        if (i + 1) % 2000 == 0:
            print(f"{i + 1}/{len(frames)} frames, {(i + 1) / (time.time() - t0):.0f} frames/s", flush=True)
    df = pd.DataFrame(rows, columns=["frame", "found", "bar_len", "tilt_deg", *CORNERS])

    # Fill short gaps (a missed frame between two hits) by interpolating the corners.
    df["interpolated"] = False
    df = df.set_index("frame").reindex(range(int(frames[0]), int(frames[-1]) + 1))
    hit = df.found.fillna(False).astype(bool)
    run = (hit != hit.shift()).cumsum()
    gap_len = (~hit).groupby(run).transform("sum")
    inside = hit.cummax() & hit[::-1].cummax()[::-1]      # gap has a hit on both sides
    fill = ~hit & (gap_len <= args.max_gap) & inside
    interp = df[CORNERS + ["tilt_deg"]].interpolate(limit_area="inside")
    df.loc[fill, CORNERS + ["tilt_deg"]] = interp.loc[fill]
    df.loc[fill, ["found", "interpolated"]] = True
    df = df.loc[df.index.isin(frames)].reset_index().rename(columns={"index": "frame"})
    df["found"] = df.found.astype(bool)
    df["interpolated"] = df.interpolated.astype(bool)
    cols = ["frame", "found", "interpolated", "bar_len", "tilt_deg", *CORNERS]
    df[cols].round(1).to_csv(f"{args.sync_dir}/speed_display.csv", index=False)

    found = df[df.found]
    cx = found[["x1", "x2"]].mean(axis=1)
    cy = found[["y1", "y2"]].mean(axis=1)
    consecutive = found.frame.diff() == 1
    jump = np.hypot(cx.diff(), cy.diff())[consecutive]
    report = {
        "frames": int(len(df)), "found_pct": round(100 * df.found.mean(), 1),
        "interpolated": int(df.interpolated.sum()),
        "bar_len_px_median": float(found.bar_len.median()),
        "tilt_deg_range": [float(found.tilt_deg.min()), float(found.tilt_deg.max())],
        "jumps_over_60px_between_consecutive_frames": int((jump > 60).sum()),
        "frames_per_s": round(len(df) / (time.time() - t0), 1),
        "settings": {"bar_hsv": [speed.BAR_HSV_LO, speed.BAR_HSV_HI], "bar_len_px": speed.BAR_LEN_PX,
                     "margin_px": speed.MARGIN_PX, "depth_px": speed.DEPTH_PX, "max_gap": args.max_gap},
    }
    with open(f"{args.sync_dir}/speed_display.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

    # Snapshots, spread over the frames where the display was found.
    os.makedirs(f"{args.sync_dir}/check", exist_ok=True)
    pts = overlay.frame_points(gaze, cal["lag_frames"])
    radius = cal["px_per_deg"] * 1.5
    by_frame = found.set_index("frame")
    snap = found.frame.to_numpy()[np.linspace(0, len(found) - 1, args.snapshots).astype(int)]
    for fid, img in read_frames(args.video, snap):
        speed.draw_display(img, by_frame.loc[fid, CORNERS].to_numpy(float))
        if fid in pts.index:
            overlay.draw_gaze(img, pts.loc[fid].x, pts.loc[fid].y, radius)
        overlay._stamp(img, f"frame {fid}  t={fid / info.fps:7.2f}s   magenta box = speed display (step 4d)")
        cv2.imwrite(f"{args.sync_dir}/check/speed_frame_{fid:06d}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"done -> {args.sync_dir}/speed_display.csv")


if __name__ == "__main__":
    main()
