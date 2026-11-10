"""Map Varjo projected gaze (x, y) to video pixels (step 3 of the pipeline).

The simulator burns its own gaze cursor into the video as a teal dot. We find that
dot in a sample of frames, pair it with the median Varjo gaze of the same frame and
fit an affine map with RANSAC (same idea as slide 41 of the client's deck):

    x_px = a*x_varjo + b*y_varjo + c
    y_px = d*x_varjo + e*y_varjo + f

The dot can be drawn a frame or two after the CSV timestamp, so a few frame lags are
tried and the one with the lowest error is kept.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import cv2
import numpy as np
import pandas as pd

# Teal / turquoise in OpenCV HSV (H is 0..180). Tune with `detect_cursor(debug=True)`.
CURSOR_HSV_LO = (78, 120, 120)
CURSOR_HSV_HI = (100, 255, 255)
CURSOR_MIN_AREA = 15
CURSOR_MAX_AREA = 5000


def detect_cursor(img, hsv_lo=CURSOR_HSV_LO, hsv_hi=CURSOR_HSV_HI, debug=False):
    """Return (x, y) of the teal gaze cursor in a BGR frame, or None.

    Picks the most circular teal blob within the area limits.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lo), np.array(hsv_hi))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if not CURSOR_MIN_AREA <= area <= CURSOR_MAX_AREA:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        fill = area / (np.pi * r * r + 1e-6)  # 1.0 for a filled disc; a ring scores lower but still passes
        score = fill * np.sqrt(area)
        if fill > 0.25 and score > best_score:
            best, best_score = (float(cx), float(cy)), score
    if debug:
        return best, mask
    return best


@dataclass
class Calibration:
    matrix: list          # 2x3 affine, Varjo (x, y, 1) -> pixel (x, y)
    lag_frames: int       # cursor frame = gaze frame + lag
    n_pairs: int
    n_inliers: int
    rmse_px: float
    median_err_px: float
    px_per_deg: float | None = None
    width: int | None = None
    height: int | None = None
    eye: str | None = None

    def apply(self, gx, gy):
        m = np.asarray(self.matrix)
        x = m[0, 0] * gx + m[0, 1] * gy + m[0, 2]
        y = m[1, 0] * gx + m[1, 1] * gy + m[1, 2]
        return x, y

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))


def collect_cursor_points(video_path, frame_ids) -> pd.DataFrame:
    """Detect the cursor in the given frames. Returns columns frame, cx, cy."""
    from .video import read_frames

    rows = []
    for fid, img in read_frames(video_path, frame_ids):
        p = detect_cursor(img)
        if p is not None:
            rows.append((fid, p[0], p[1]))
    return pd.DataFrame(rows, columns=["frame", "cx", "cy"])


def _fit(pairs: pd.DataFrame, thresh_px: float):
    src = pairs[["gx", "gy"]].to_numpy(np.float32)
    dst = pairs[["cx", "cy"]].to_numpy(np.float32)
    m, inl = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC,
                                  ransacReprojThreshold=thresh_px, maxIters=5000, confidence=0.999)
    if m is None:
        return None, None, np.inf
    pred = src @ m[:, :2].T + m[:, 2]
    err = np.linalg.norm(pred - dst, axis=1)
    inl = inl.ravel().astype(bool)
    return m, err, float(np.sqrt(np.mean(err[inl] ** 2)))


def fit_calibration(per_frame_gaze: pd.DataFrame, cursor: pd.DataFrame,
                    max_lag: int = 5, thresh_px: float = 15.0) -> tuple[Calibration, pd.DataFrame]:
    """Fit the affine map, trying lags in [-max_lag, max_lag]. Returns calibration and the pairs used."""
    best = None
    for lag in range(-max_lag, max_lag + 1):
        g = per_frame_gaze.reset_index()
        g["frame"] = g["frame"] + lag
        pairs = cursor.merge(g, on="frame")
        if len(pairs) < 10:
            continue
        m, err, rmse = _fit(pairs, thresh_px)
        if m is None:
            continue
        if best is None or rmse < best[2]:
            pairs = pairs.assign(err_px=err, inlier=err <= thresh_px, lag=lag)
            best = (lag, m, rmse, pairs)
    if best is None:
        raise RuntimeError("calibration failed: too few cursor/gaze pairs")
    lag, m, rmse, pairs = best
    inl = pairs["inlier"]
    cal = Calibration(matrix=m.tolist(), lag_frames=lag, n_pairs=len(pairs),
                      n_inliers=int(inl.sum()), rmse_px=rmse,
                      median_err_px=float(pairs.loc[inl, "err_px"].median()))
    return cal, pairs


def estimate_px_per_deg(gaze: pd.DataFrame, cal: Calibration) -> float | None:
    """Pixels per degree of visual angle, from the 3D gaze direction vs the mapped pixel position.

    Regresses horizontal pixel position on horizontal gaze angle (atan2(fx, fz)) and
    vertical on vertical; returns the mean slope. Used to turn a 1-2 deg gaze radius into pixels.
    """
    cols = ["gaze_forward_x", "gaze_forward_y", "gaze_forward_z"]
    if not set(cols) <= set(gaze.columns):
        return None
    v = gaze[gaze["valid"]].dropna(subset=cols)
    if len(v) < 100:
        return None
    ax = np.degrees(np.arctan2(v[cols[0]], v[cols[2]]))
    ay = np.degrees(np.arctan2(v[cols[1]], v[cols[2]]))
    px, py = cal.apply(v["gx"].to_numpy(), v["gy"].to_numpy())
    sx = abs(np.polyfit(ax, px, 1)[0])
    sy = abs(np.polyfit(ay, py, 1)[0])
    return float((sx + sy) / 2)
