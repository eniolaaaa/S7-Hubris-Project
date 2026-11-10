"""Step 5: label every gaze sample with what the driver looked at (`category - class`).

Each valid sample in the drive gets a circle of RADIUS_DEG around its gaze point on `display_frame`. We walk
through PRIORITY and the first thing the circle touches gives the label; if it touches two vehicles, the one
with the most overlap wins. The strict label from the gaze point alone is kept too (`label_point`): far ahead,
vehicles (~67 px tall in video 1) are smaller than the circle (79 px across), so the circle often touches a
car, a line and an edge at the same time.

Settings agreed in the group meeting of 2026-10-27: fog is left out, plain road is `road - surface`, fixed
priority instead of largest overlap (otherwise the road surface wins almost every time), guard rail included.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

RADIUS_DEG = 1.5
DRIVE_FRAMES = None                 # None: found automatically where the road is visible (drive.json);
                                    # outside it is fog -> `other - fog`. Or set (first, last) by hand.
MIN_TRACK_LEN = 3                   # vehicles: ignore boxes that never form a track
PRIORITY = ["speed display", "vehicle", "road line", "road edge", "guard rail", "road surface"]
LABELS = {"speed display": ("interior", "speed display"), "road line": ("road", "road line"),
          "road edge": ("road", "road edge"), "guard rail": ("road", "guard rail"),
          "road surface": ("road", "surface")}   # set "road surface" to ("other", "other") to merge it
SCALE = 0.5                         # masks are at half the video size


def disc(radius: float):
    """Pixel offsets inside a circle of `radius`."""
    r = int(np.ceil(radius))
    dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
    keep = dx ** 2 + dy ** 2 <= radius ** 2
    return dx[keep], dy[keep]


def label_frame(xy: np.ndarray, radius_px: float, masks: dict, boxes: pd.DataFrame) -> list[tuple]:
    """Labels for the gaze points `xy` (N x 2, video pixels) in one frame.

    masks: half-size uint8 masks for "speed display", "road line", "road edge", "guard rail", "road surface".
    boxes: vehicle boxes in video pixels (track_id, class, x1, y1, x2, y2, conf).

    Per point: (category, class, track_id, overlap, on_target, ambiguous, touches, label_point).
    on_target: the gaze point itself is on the labelled thing.
    ambiguous: the circle touches several things and the gaze point isn't on the chosen one, so the
    priority order decided.
    touches: everything the circle touches except plain road surface.
    """
    dx, dy = disc(radius_px * SCALE)
    h, w = next(iter(masks.values())).shape
    b = boxes[["x1", "y1", "x2", "y2"]].to_numpy(float) * SCALE
    out = []
    for x, y in xy * SCALE:
        cx, cy = int(np.rint(x)), int(np.rint(y))
        px, py = cx + dx, cy + dy
        inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        qx, qy = px[inside], py[inside]
        hits = {k: int(m[qy, qx].sum()) for k, m in masks.items()}
        centre_ok = 0 <= cx < w and 0 <= cy < h
        at_point = {k: centre_ok and bool(m[cy, cx]) for k, m in masks.items()}
        veh = []
        for (x1, y1, x2, y2), tid, cls, conf in zip(b, boxes.track_id, boxes["class"], boxes.conf.fillna(0)):
            n = int(((px >= x1) & (px <= x2) & (py >= y1) & (py <= y2)).sum())
            if n:
                veh.append((n, conf, int(tid), cls, x1 <= x <= x2 and y1 <= y <= y2))
        veh.sort(key=lambda v: (v[0], v[1]), reverse=True)
        touches = (["speed display"] if hits.get("speed display") else []) \
            + [f"{v[3]}#{v[2]}" for v in veh] \
            + [k for k in ("road line", "road edge", "guard rail") if hits.get(k)]

        result, on_target = ("other", "other", -1, 0.0), False
        for kind in PRIORITY:
            if kind == "vehicle" and veh:
                n, _, tid, cls, centre_in = veh[0]
                result, on_target = ("object", cls, tid, n / len(dx)), centre_in
                break
            if kind != "vehicle" and hits.get(kind):
                result, on_target = (*LABELS[kind], -1, hits[kind] / len(dx)), at_point[kind]
                break

        point = "other - other"
        for kind in PRIORITY:
            if kind == "vehicle":
                under = [v for v in veh if v[4]]
                if under:
                    point = f"object - {under[0][3]}"
                    break
            elif at_point.get(kind):
                point = " - ".join(LABELS[kind])
                break
        out.append((*result, on_target, len(touches) > 1 and not on_target, ";".join(touches), point))
    return out


def drive_frames(sync_dir: str, id2label: dict) -> tuple[int, int]:
    """The drive's first and last frame: DRIVE_FRAMES if set, else <sync_dir>/drive.json, computed and saved
    from the road segmentation the first time (~1 min)."""
    import json
    import os
    from .road import drive_range
    if DRIVE_FRAMES is not None:
        return tuple(DRIVE_FRAMES)
    path = f"{sync_dir}/drive.json"
    if os.path.exists(path):
        return tuple(json.load(open(path))["drive_frames"])
    frames = [int(f[:6]) for f in os.listdir(f"{sync_dir}/seg_m2f") if f.endswith(".png") and f[:6].isdigit()]
    rng = drive_range(f"{sync_dir}/seg_m2f", frames, id2label)
    with open(path, "w") as f:
        json.dump({"drive_frames": rng, "method": "first/last frame with the road visible in 90% of the "
                                                  "next/previous 10 s (hubris.road.drive_range)"}, f, indent=2)
    return rng


def interior_tracks(boxes: pd.DataFrame, sync_dir: str, id2label: dict, max_share: float = 0.5) -> list[int]:
    """Tracks whose boxes are mostly centred on the car's own interior (segmentation "Ego Vehicle").

    YOLO sometimes boxes the bottom edge of the windscreen as a wide, flat "car" (~1,170 x 115 px, conf ~0.3)
    for long enough to form a track; a real vehicle can't be inside the car.
    """
    from .road import EGO, _ids
    ego = _ids(EGO, id2label)
    real = boxes[~boxes.interpolated].copy()
    real["cx"], real["cy"] = (real.x1 + real.x2) / 2, (real.y1 + real.y2) / 2
    real["on_ego"] = False
    low = real[real.cy > 700]                        # the interior never reaches higher up the view
    for fid, g in low.groupby("frame"):
        seg = cv2.imread(f"{sync_dir}/seg_m2f/{int(fid):06d}.png", cv2.IMREAD_UNCHANGED)
        if seg is None:
            continue
        x = np.clip((g.cx * SCALE).astype(int), 0, seg.shape[1] - 1)
        y = np.clip((g.cy * SCALE).astype(int), 0, seg.shape[0] - 1)
        real.loc[g.index, "on_ego"] = np.isin(seg[y, x], ego)
    share = real.groupby("track_id").on_ego.mean()
    return sorted(int(t) for t in share[share > max_share].index)


def display_mask(quad, shape) -> np.ndarray:
    """Half-size mask of the speed-display box (corners in video pixels; may reach past the frame)."""
    m = np.zeros(shape, np.uint8)
    if quad is not None and np.isfinite(quad).all():
        cv2.fillPoly(m, [np.rint(np.asarray(quad).reshape(4, 2) * SCALE).astype(np.int32)], 1)
    return m
