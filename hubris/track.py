"""Step 4c: track the YOLO car/truck boxes over time (ByteTrack on the saved detections).

Tracking needs only the boxes, not the images, so it runs on detections_*.parquet on the CPU, without
rerunning YOLO. It gives every vehicle an id that stays the same across frames, one class per vehicle
(confidence-weighted vote, so a van does not flip between car and truck), and fills short gaps where YOLO
missed a vehicle for a few frames (linear interpolation).

The tracker is ultralytics' ByteTrack with its default settings (cfg/trackers/bytetrack.yaml), i.e. the
same tracker `model.track()` uses. Only the boxes with conf >= 0.25 were saved, so ByteTrack's second,
low-score matching stage has nothing to work with; it then behaves like SORT (Kalman filter + IoU).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

BOX = ["x1", "y1", "x2", "y2"]
CLASS_IDS = {"car": 0, "truck": 1}  # tracker-internal class ids
BYTETRACK = dict(track_high_thresh=0.25, track_low_thresh=0.1, new_track_thresh=0.25,
                 track_buffer=30, match_thresh=0.8, fuse_score=True)


def assign_track_ids(dets: pd.DataFrame, first_frame: int, last_frame: int, img_shape=(1536, 2048),
                     **params) -> np.ndarray:
    """Track id per row of `dets` (positional). Every frame in [first_frame, last_frame] is fed to the
    tracker in order; frames without rows count as empty, so lost tracks age correctly."""
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.byte_tracker import BYTETracker

    tracker = BYTETracker(SimpleNamespace(**(BYTETRACK | params)))
    ids = np.full(len(dets), -1, dtype=int)
    rows = {f: np.asarray(ix) for f, ix in dets.reset_index(drop=True).groupby("frame").indices.items()}
    data = np.c_[dets[BOX + ["conf"]].to_numpy(np.float32), dets["class"].map(CLASS_IDS).to_numpy(np.float32)]
    empty = np.zeros((0, 6), np.float32)
    for fid in range(first_frame, last_frame + 1):
        ix = rows.get(fid)
        tracker.update(Boxes(empty if ix is None else data[ix], img_shape))
        if ix is None:
            continue
        # tracked_stracks also holds tracks born in this frame, which the tracker's output leaves out until
        # they are matched again; reading it gives the first box of every track its id too.
        for t in tracker.tracked_stracks:
            if t.frame_id == tracker.frame_id:
                ids[ix[int(t.idx)]] = t.track_id
    return ids


def finish_tracks(dets: pd.DataFrame, max_gap: int = 10) -> pd.DataFrame:
    """One class per track, track length, and interpolated boxes for gaps of up to `max_gap` frames.

    Columns added: `track_id`, `class` (track vote), `class_frame` (what YOLO said in that frame),
    `track_len` (frames with a real detection), `interpolated` (True for filled-in boxes, which have no conf).
    """
    d = dets[dets.track_id >= 0].copy()
    vote = d.groupby(["track_id", "class"])["conf"].sum().unstack(fill_value=0).idxmax(axis=1)
    d["class_frame"] = d["class"]
    d["class"] = d.track_id.map(vote)
    d["track_len"] = d.track_id.map(d.groupby("track_id").size())
    d["interpolated"] = False

    filled = []
    for tid, g in d.groupby("track_id"):
        f = g.frame.to_numpy()
        if len(f) < 2:
            continue
        missing = np.setdiff1d(np.arange(f[0], f[-1] + 1), f)
        if not len(missing):
            continue
        nxt = np.searchsorted(f, missing)
        missing = missing[f[nxt] - f[nxt - 1] - 1 <= max_gap]
        if not len(missing):
            continue
        box = {c: np.interp(missing, f, g[c].to_numpy()) for c in BOX}
        filled.append(pd.DataFrame({"frame": missing, "track_id": tid, "category": "object",
                                    "class": g["class"].iloc[0], "class_frame": None, "coco_class": None,
                                    **box, "conf": np.nan, "track_len": g.track_len.iloc[0],
                                    "interpolated": True}))
    out = pd.concat([d, *filled], ignore_index=True) if filled else d
    cols = ["frame", "track_id", "category", "class", "class_frame", "coco_class", *BOX, "conf",
            "interpolated", "track_len"]
    return out[cols].sort_values(["frame", "track_id"]).reset_index(drop=True)


def track_color(track_id: int):
    """Stable, distinct BGR colour per track id."""
    rng = np.random.default_rng(int(track_id) * 7919)
    return tuple(int(v) for v in rng.integers(60, 256, 3))
