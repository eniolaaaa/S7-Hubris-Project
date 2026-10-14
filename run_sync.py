"""Step 1-3 of the HUBRIS pipeline for one video: clean, time-sync and pixel-map the gaze,
then write an overlay check.

Usage:
    python run_sync.py --video P01.mp4 --gaze P01_gaze.csv --out out/P01

Outputs in --out:
    gaze_synced.csv     one row per Varjo sample: time, frame, valid, x_px, y_px (+ raw columns)
    calibration.json    affine matrix, frame lag, fit error, pixels per degree
    calib_pairs.csv     cursor detections paired with CSV gaze (for the report)
    sync_report.json    summary numbers
    check/              snapshot frames + a short clip with the computed gaze drawn on top
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from hubris import calibrate, gaze_io, overlay
from hubris.video import video_info

KEEP_RAW = ["raw_csv_row", gaze_io.TIME_COL, gaze_io.STATUS_COL]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--gaze", required=True, help="Varjo gaze CSV")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib-frames", type=int, default=800, help="frames sampled for cursor detection")
    ap.add_argument("--cursor-csv", help="precomputed cursor track (detect_cursor_chunk.py); skips detection")
    ap.add_argument("--max-lag", type=int, default=5, help="frame lag searched between CSV and cursor")
    ap.add_argument("--radius-deg", type=float, default=1.5, help="gaze circle radius in degrees")
    ap.add_argument("--eye", choices=["auto", "left", "right"], default="auto",
                    help="which eye's projected view the video shows (auto = best calibration fit)")
    ap.add_argument("--snapshots", type=int, default=12)
    ap.add_argument("--clip-seconds", type=float, default=20)
    args = ap.parse_args()

    os.makedirs(f"{args.out}/check", exist_ok=True)

    info = video_info(args.video)
    print(f"video: {info.width}x{info.height} @ {info.fps:.3f} fps, {info.n_frames} frames ({info.duration_s:.1f} s)")

    raw = gaze_io.read_varjo_csv(args.gaze)
    gaze = gaze_io.prepare_gaze(raw, info.fps, info.n_frames, eye="left" if args.eye == "auto" else args.eye)
    gsum = gaze_io.summary(gaze)
    print("gaze:", json.dumps(gsum))

    # Calibration: spread frames over the whole video, only frames that have valid gaze.
    pf = gaze_io.frame_gaze(gaze)
    pick = np.linspace(0, len(pf) - 1, min(args.calib_frames, len(pf))).astype(int)
    frames = pf.index[pick].to_numpy()
    # Also read the neighbours needed for the lag search.
    to_read = np.unique(np.clip(np.concatenate([frames + k for k in range(-args.max_lag, args.max_lag + 1)]),
                                0, info.n_frames - 1))
    if args.cursor_csv:
        cursor = pd.read_csv(args.cursor_csv).drop_duplicates("frame")
        cursor = cursor[cursor.frame.isin(to_read)]
    else:
        print(f"detecting cursor in {len(to_read)} frames ...")
        cursor = calibrate.collect_cursor_points(args.video, to_read)
    print(f"cursor found in {len(cursor)} / {len(to_read)} frames")

    eyes = ["left", "right"] if args.eye == "auto" else [args.eye]
    best = None
    for eye in eyes:
        g = gaze_io.prepare_gaze(raw, info.fps, info.n_frames, eye=eye)
        c, p = calibrate.fit_calibration(gaze_io.frame_gaze(g), cursor, max_lag=args.max_lag)
        print(f"  eye={eye}: inliers {c.n_inliers}/{c.n_pairs}, median {c.median_err_px:.1f}px, lag {c.lag_frames}")
        if best is None or (c.n_inliers, -c.median_err_px) > (best[1].n_inliers, -best[1].median_err_px):
            best = (eye, c, p, g)
    eye, cal, pairs, gaze = best
    cal.eye = eye
    cal.width, cal.height = info.width, info.height

    gaze["x_px"], gaze["y_px"] = cal.apply(gaze["gx"].to_numpy(), gaze["gy"].to_numpy())
    inside = (gaze.x_px >= 0) & (gaze.x_px < info.width) & (gaze.y_px >= 0) & (gaze.y_px < info.height)
    gaze["inside_video"] = inside
    gaze["display_frame"] = gaze["frame"] + cal.lag_frames

    cal.px_per_deg = calibrate.estimate_px_per_deg(gaze, cal)
    radius_px = (cal.px_per_deg or 20) * args.radius_deg
    cal.save(f"{args.out}/calibration.json")
    pairs.to_csv(f"{args.out}/calib_pairs.csv", index=False)
    print(f"calibration: lag={cal.lag_frames} frames, inliers {cal.n_inliers}/{cal.n_pairs}, "
          f"RMSE {cal.rmse_px:.1f}px, median {cal.median_err_px:.1f}px, px/deg={cal.px_per_deg}")

    cols = KEEP_RAW + ["video_time_s", "frame", "display_frame", "valid", "gx", "gy", "x_px", "y_px", "inside_video"]
    gaze[[c for c in cols if c in gaze.columns]].to_csv(f"{args.out}/gaze_synced.csv", index=False)

    # Overlay check.
    valid_frames = gaze.loc[gaze.valid, "display_frame"].dropna().astype(int)
    valid_frames = valid_frames[(valid_frames >= 0) & (valid_frames < info.n_frames)].unique()
    snap = valid_frames[np.linspace(0, len(valid_frames) - 1, args.snapshots).astype(int)]
    overlay.write_snapshots(args.video, gaze, cal.lag_frames, snap, radius_px, f"{args.out}/check", info.fps)
    start = int(info.n_frames * 0.4)
    overlay.write_clip(args.video, gaze, cal.lag_frames, start, int(args.clip_seconds * info.fps),
                       radius_px, f"{args.out}/check/overlay_clip.mp4", info.fps)

    report = {
        "video": vars(info) | {"duration_s": info.duration_s},
        "gaze": gsum,
        "calibration": json.load(open(f"{args.out}/calibration.json")),
        "gaze_radius_px": radius_px,
        "valid_inside_video_pct": float(100 * (gaze.valid & gaze.inside_video).sum() / len(gaze)),
        "samples_per_frame": float(gsum["rate_hz"] / info.fps),
    }
    with open(f"{args.out}/sync_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
