"""Load a Varjo gaze CSV, clean it and attach each sample to a video frame.

Step 1 (clean) and step 2 (match time) of the pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Column names as exported by the Varjo logger (see deck slide 39).
TIME_COL = "relative_to_video_first_frame_timestamp"  # nanoseconds
STATUS_COL = "status"
LEFT_X, LEFT_Y = "gaze_projected_to_left_view_x", "gaze_projected_to_left_view_y"
RIGHT_X, RIGHT_Y = "gaze_projected_to_right_view_x", "gaze_projected_to_right_view_y"
FORWARD = ("gaze_forward_x", "gaze_forward_y", "gaze_forward_z")

# Varjo gaze status: 0 = invalid, 1 = adjusting, 2 = valid.
VALID_STATUS = 2


def read_varjo_csv(path: str) -> pd.DataFrame:
    """Read the CSV, sniffing the separator (comma or semicolon) and decimal mark."""
    with open(path, "r", encoding="utf-8-sig") as f:
        head = f.readline()
    sep = ";" if head.count(";") > head.count(",") else ","
    decimal = "," if sep == ";" else "."
    # The Windows Varjo logger writes NaN as "-nan(ind)".
    df = pd.read_csv(path, sep=sep, decimal=decimal, low_memory=False,
                     na_values=["-nan(ind)", "nan(ind)", "-nan", "-inf", "inf"])
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype) == "str":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["raw_csv_row"] = np.arange(len(df))
    return df


def prepare_gaze(df: pd.DataFrame, fps: float, n_frames: int | None = None, eye: str = "left") -> pd.DataFrame:
    """Add video time, frame index, a combined 2D gaze point and a validity flag.

    - video_time_s: Varjo timestamp relative to the first video frame, in seconds.
    - frame: the frame on screen at that moment, floor(t * fps) (frame k spans k/fps..(k+1)/fps).
    - gx, gy: combined gaze projected into the left or right eye's view (`eye`). The two views use
      different coordinates (left x ~ +0.47 where right x ~ -0.34 for the same sample), so they are
      never averaged; the video shows one eye's view and calibration picks which.
    - valid: status is valid, the point is finite, and the sample lies inside the video.
    """
    out = df.copy()
    out["video_time_s"] = out[TIME_COL].astype(float) / 1e9
    out["frame"] = np.floor(out["video_time_s"] * fps).astype("Int64")

    cols = set(out.columns)
    x, y = (LEFT_X, LEFT_Y) if eye == "left" else (RIGHT_X, RIGHT_Y)
    out["gx"], out["gy"] = out[x].astype(float), out[y].astype(float)

    valid = np.isfinite(out["gx"]) & np.isfinite(out["gy"]) & (out["video_time_s"] >= 0)
    if STATUS_COL in cols:
        valid &= out[STATUS_COL] == VALID_STATUS
    if n_frames is not None:
        valid &= out["frame"] < n_frames
    out["valid"] = valid.astype(bool)
    return out


def frame_gaze(gaze: pd.DataFrame) -> pd.DataFrame:
    """One row per frame: median projected gaze of the valid samples in that frame."""
    v = gaze[gaze["valid"]]
    return v.groupby("frame").agg(gx=("gx", "median"), gy=("gy", "median"), n=("gx", "size"))


def summary(gaze: pd.DataFrame) -> dict:
    t = gaze["video_time_s"].to_numpy()
    dt = np.diff(t)
    return {
        "samples": int(len(gaze)),
        "duration_s": float(t[-1] - t[0]) if len(t) else 0.0,
        "median_dt_ms": float(np.median(dt) * 1e3) if len(dt) else float("nan"),
        "rate_hz": float(1 / np.median(dt)) if len(dt) else float("nan"),
        "valid_pct": float(100 * gaze["valid"].mean()),
        "pre_roll_samples": int((t < 0).sum()),
    }
