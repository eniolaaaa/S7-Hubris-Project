"""Draw the computed gaze on video frames to check sync against the burned-in teal cursor."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from .video import read_frames

MAGENTA = (255, 0, 255)
YELLOW = (0, 255, 255)


def draw_gaze(img, x, y, radius_px, color=MAGENTA, label=None):
    if not (np.isfinite(x) and np.isfinite(y)):
        return img
    p = (int(round(x)), int(round(y)))
    cv2.circle(img, p, int(radius_px), color, 2, cv2.LINE_AA)
    cv2.drawMarker(img, p, color, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
    if label:
        cv2.putText(img, label, (p[0] + 10, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return img


def _stamp(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def frame_points(gaze: pd.DataFrame, lag: int) -> pd.DataFrame:
    """Median mapped pixel gaze per *display* frame (gaze frame shifted by the fitted lag)."""
    v = gaze[gaze["valid"]].copy()
    v["frame"] = v["frame"] + lag
    return v.groupby("frame").agg(x=("x_px", "median"), y=("y_px", "median"))


def write_snapshots(video_path, gaze, lag, frame_ids, radius_px, out_dir, fps):
    pts = frame_points(gaze, lag)
    paths = []
    for fid, img in read_frames(video_path, frame_ids):
        if fid in pts.index:
            r = pts.loc[fid]
            draw_gaze(img, r.x, r.y, radius_px)
        _stamp(img, f"frame {fid}  t={fid / fps:7.2f}s   magenta = gaze from CSV, teal = burned-in cursor")
        p = f"{out_dir}/check_frame_{fid:06d}.jpg"
        cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        paths.append(p)
    return paths


def write_clip(video_path, gaze, lag, start_frame, n_frames, radius_px, out_path, fps, scale=0.5):
    """Short overlay clip (mp4) so the alignment can be judged in motion."""
    pts = frame_points(gaze, lag)
    writer = None
    for fid, img in read_frames(video_path, range(start_frame, start_frame + n_frames)):
        if fid in pts.index:
            r = pts.loc[fid]
            draw_gaze(img, r.x, r.y, radius_px)
        _stamp(img, f"frame {fid}  t={fid / fps:7.2f}s")
        if scale != 1:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if writer is None:
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(img)
    if writer is not None:
        writer.release()
    return out_path
