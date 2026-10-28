"""Step 5: label every gaze sample with what the driver looked at (`category - class`).

For a valid sample in the drive, a circle of RADIUS_DEG around its pixel position on `display_frame` is
checked against everything found in steps 4-4d. The first kind in PRIORITY that the circle touches gives the
label; within a kind (two cars) the one with the most overlap wins. `touches` lists everything the circle
touches (road surface not counted), `on_target` says whether the gaze point itself is on the labelled thing,
and `ambiguous` marks samples where the circle touches several things but the gaze point is not on
the chosen one, i.e. the priority order made the choice. `label_point` is the strict label from the gaze point
alone. Far ahead, where this driver mostly looks, vehicles (~67 px tall) are smaller than the circle (79 px
across), so the circle usually touches a car, a line and an edge at once.

The group's choices (2026-10-27) are the settings below: drop the fog, label plain road `road - surface`,
fixed priority instead of largest overlap (the road surface would otherwise beat almost everything),
include guard rail.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

RADIUS_DEG = 1.5
DRIVE_FRAMES = (2090, 20460)        # outside: fog at the start and end -> `other - fog`
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
    Returns per point: (category, class, track_id, overlap share of the circle, on_target, ambiguous, touches,
    label_point). `on_target`: the gaze point itself lies on the labelled thing. `ambiguous`: the circle touches
    several things and the gaze point is not on the chosen one, so the priority order made the choice.
    `label_point`: the strict label from the gaze point alone, same priority, no circle.
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
