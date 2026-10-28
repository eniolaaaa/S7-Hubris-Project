"""Step 6: draw the annotated demo video frames.

Each frame shows everything the pipeline found (road lines/edges, guard rail, vehicles with their track
number, the speed display), the gaze circle, a banner with the current label, and a timeline strip of the
labels over the whole clip. What the driver is looking at is drawn thicker and filled.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from . import detect

LABEL_COLORS = {  # BGR
    "object - car": (60, 200, 60),
    "object - truck": (0, 140, 255),
    "road - road line": (40, 40, 230),
    "road - road edge": (0, 220, 255),
    "road - guard rail": (255, 160, 0),
    "road - surface": (170, 90, 170),
    "interior - speed display": (255, 255, 255),
    "other - other": (150, 150, 150),
    "other - fog": (205, 205, 205),
    "other - invalid": (80, 80, 80),
}
GAZE = (255, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def frame_summary(labels: pd.DataFrame) -> pd.DataFrame:
    """One row per display frame: the most common label of its gaze samples, the vehicle it belongs to,
    the median gaze point and whether most samples were ambiguous."""
    s = labels.dropna(subset=["display_frame"]).copy()
    s["frame"] = s.display_frame.astype(int)
    valid = s[s.label != "other - invalid"]
    top = s.groupby("frame").label.agg(lambda x: x.value_counts().index[0])
    # Prefer a real label when the frame has any valid sample.
    top.update(valid.groupby("frame").label.agg(lambda x: x.value_counts().index[0]))
    out = top.to_frame("label")
    m = s[s.label == s.frame.map(out.label)]   # the samples behind each frame's label
    out["track_id"] = m.groupby("frame").track_id.agg(lambda x: x.value_counts().index[0])
    out["ambiguous"] = m.groupby("frame").ambiguous.mean() > 0.5
    pts = valid.groupby("frame")[["x_px", "y_px"]].median()
    return out.join(pts)


def _fill(img, pts, color, alpha):
    over = img.copy()
    cv2.fillPoly(over, [pts], color)
    cv2.addWeighted(over, alpha, img, 1 - alpha, 0, dst=img)


ROAD_LAYER = {"road - road line": "road line", "road - road edge": "road edge",
              "road - guard rail": "guard rail", "road - surface": "road surface"}


def _highlight_near(img, mask_half, centre, radius_px, color, alpha=0.65):
    """Colour a half-size mask only within 2x the gaze radius around `centre`."""
    h, w = img.shape[:2]
    r = int(2 * radius_px)
    x0, y0 = max(int(centre[0]) - r, 0) // 2 * 2, max(int(centre[1]) - r, 0) // 2 * 2
    x1, y1 = min(int(centre[0]) + r, w), min(int(centre[1]) + r, h)
    if x1 <= x0 or y1 <= y0:
        return
    m = cv2.resize(mask_half[y0 // 2:(y1 + 1) // 2, x0 // 2:(x1 + 1) // 2], None, fx=2, fy=2,
                   interpolation=cv2.INTER_NEAREST)[:y1 - y0, :x1 - x0] > 0
    yy, xx = np.ogrid[y0:y0 + m.shape[0], x0:x0 + m.shape[1]]
    m &= (xx - centre[0]) ** 2 + (yy - centre[1]) ** 2 <= r * r
    roi = img[y0:y0 + m.shape[0], x0:x0 + m.shape[1]]
    roi[m] = ((1 - alpha) * roi[m] + alpha * np.array(color)).astype(np.uint8)


def draw_frame(img, info, boxes: pd.DataFrame, quad, radius_px, t_s, fid, layers=None, style="clean"):
    """Annotate one full-size frame in place. `info` is the frame's row of frame_summary (or None).

    style "clean": only what the driver looks at is highlighted (for road labels just the part near the gaze),
    other vehicles get a thin outline. style "full": everything found is drawn (road layers are drawn by the
    caller), with the target thick and filled.
    """
    label = info["label"] if info is not None else "other - invalid"
    target = int(info["track_id"]) if info is not None and label.startswith("object") else -1
    clean = style == "clean"
    has_gaze = info is not None and np.isfinite(info["x_px"])

    if clean and has_gaze and layers is not None and label in ROAD_LAYER:
        _highlight_near(img, layers[ROAD_LAYER[label]], (info["x_px"], info["y_px"]), radius_px, LABEL_COLORS[label])

    if quad is not None:
        q = np.rint(np.asarray(quad).reshape(4, 2)).astype(np.int32)
        hot = label == "interior - speed display"
        if hot:
            _fill(img, q, LABEL_COLORS[label], 0.25)
        if hot or not clean:
            cv2.polylines(img, [q], True, LABEL_COLORS["interior - speed display"], 6 if hot else 2, cv2.LINE_AA)
            cv2.putText(img, "speed display", (int(q[0][0]), max(int(q[0][1]) - 10, 14)), FONT, 0.9,
                        LABEL_COLORS["interior - speed display"], 2, cv2.LINE_AA)

    for tid, cls, x1, y1, x2, y2 in boxes[["track_id", "class", "x1", "y1", "x2", "y2"]].itertuples(index=False):
        color = detect.BOX_COLORS.get(cls, (255, 255, 255))
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        hot = tid == target
        if hot:
            _fill(img, np.array([p1, (p2[0], p1[1]), p2, (p1[0], p2[1])], np.int32), color, 0.3)
        cv2.rectangle(img, p1, p2, color, 6 if hot else 2, cv2.LINE_AA)
        if hot or not clean:
            cv2.putText(img, f"{cls} #{tid}", (p1[0], max(p1[1] - 8, 14)), FONT, 1.0 if hot else 0.7, color,
                        3 if hot else 2, cv2.LINE_AA)

    if has_gaze:
        c = (int(info["x_px"]), int(info["y_px"]))
        cv2.circle(img, c, int(radius_px), GAZE, 4, cv2.LINE_AA)
        cv2.drawMarker(img, c, GAZE, cv2.MARKER_CROSS, 22, 3, cv2.LINE_AA)

    # Banner: time and current label.
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 92), (20, 20, 20), -1)
    mm, ss = divmod(t_s, 60)
    cv2.putText(img, f"{int(mm):02d}:{ss:04.1f}   frame {fid}", (24, 60), FONT, 1.3, (230, 230, 230), 2, cv2.LINE_AA)
    text = label + (f"  #{target}" if target >= 0 else "") \
        + ("  (ambiguous)" if info is not None and info["ambiguous"] else "")
    color = LABEL_COLORS.get(label, (200, 200, 200))
    cv2.circle(img, (520, 46), 18, color, -1, cv2.LINE_AA)
    cv2.putText(img, "looking at:  " + text, (556, 62), FONT, 1.6, color, 3, cv2.LINE_AA)
    return img


def draw_timeline(img, strip: np.ndarray, pos: float):
    """Paste the clip's label strip at the bottom and mark the current position (0..1)."""
    h, w = img.shape[:2]
    sh = strip.shape[0]
    img[h - sh:h] = cv2.resize(strip, (w, sh), interpolation=cv2.INTER_NEAREST)
    x = int(pos * (w - 1))
    cv2.line(img, (x, h - sh - 10), (x, h - 1), (255, 255, 255), 4)
    return img


def timeline_strip(frames, summary: pd.DataFrame, height=36) -> np.ndarray:
    labels = summary.label.reindex(frames).fillna("other - invalid")
    row = np.array([LABEL_COLORS.get(l, (200, 200, 200)) for l in labels], np.uint8)
    return np.repeat(row[None], height, axis=0)


SHORT = {"other - other": "other", "other - invalid": "no gaze", "other - fog": "fog",
         "road - surface": "road surface"}


def draw_legend(img, labels, strip_height=36):
    """One-line colour key for the timeline, just above it."""
    h, w = img.shape[:2]
    y0 = h - strip_height - 48
    cv2.rectangle(img, (0, y0), (w, h - strip_height), (20, 20, 20), -1)
    x = 24
    for l in labels:
        name = SHORT.get(l, l.split(" - ")[1])
        cv2.rectangle(img, (x, y0 + 14), (x + 22, y0 + 36), LABEL_COLORS[l], -1)
        cv2.putText(img, name, (x + 32, y0 + 33), FONT, 0.85, (230, 230, 230), 2, cv2.LINE_AA)
        x += 32 + cv2.getTextSize(name, FONT, 0.85, 2)[0][0] + 34
    return img
