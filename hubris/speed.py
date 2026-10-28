"""Step 4d: find the speed display ("NN km/h" screen) in each frame.

The view moves with the driver's head, so the display moves, tilts (up to ~20 deg when looking down) and is
often half cut off at the bottom of the frame. Its screen has a pale-teal light bar along the top that stays
about the same size (~225 px) in every head position, so the bar is the anchor:
- find thin, straight, pale-teal stripes with a dark screen just below them (this rules out the glowing
  steering-wheel rim, the strongly coloured teal gaze cursor and the sky);
- glue pieces of one stripe back together (the gaze cursor can split the bar) and restore an end hidden by
  the cursor or the frame border;
- the screen is a box hanging below the bar: the bar plus MARGIN_PX on each side, DEPTH_PX deep.
Feature matching (SIFT + homography) was tried first; the dark dashboard has too few features (5-7 inliers).
"""
from __future__ import annotations

import cv2
import numpy as np

BAR_HSV_LO = (75, 40, 170)    # pale teal; the gaze cursor is the same hue but saturation ~220
BAR_HSV_HI = (95, 160, 255)
SEARCH_TOP = 600              # the display never shows above this row
BAR_LEN_PX = 225.0            # full bar length, roughly constant over head poses (212-243 px seen)
MARGIN_PX = 20.0              # screen edge beyond the bar ends / above the bar
DEPTH_PX = 380.0              # screen height below the bar (matches the bottom bar when it is visible)
MAX_TILT_DEG = 40


def find_bar(img, cursor_xy=None) -> dict | None:
    """The display's top light bar: ends e1 (left) and e2 (right), direction u, normal v (pointing down
    into the screen), visible length and tilt. None when the display is not in view."""
    sub = img[SEARCH_TOP:]
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BAR_HSV_LO, BAR_HSV_HI)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    segs = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 60:
            continue
        ys, xs = np.nonzero(lab == i)
        pts = np.c_[xs, ys + SEARCH_TOP].astype(np.float32)
        (_, _), (w, h), _ = cv2.minAreaRect(pts)
        length, thick = max(w, h), min(w, h)
        if thick > 14 or not 25 <= length <= 300 or st[i, cv2.CC_STAT_AREA] < 0.4 * length * max(thick, 1):
            continue
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        u = np.array([vx, vy]) * (1 if vx >= 0 else -1)
        segs.append({"pts": pts, "u": u, "p0": np.array([x0, y0])})

    # Glue collinear pieces of one stripe together.
    groups = []
    for s in segs:
        for g in groups:
            angle = abs(np.degrees(np.arctan2(s["u"][1], s["u"][0]) - np.arctan2(g["u"][1], g["u"][0])))
            d = s["p0"] - g["p0"]
            if angle < 5 and abs(g["u"][0] * d[1] - g["u"][1] * d[0]) < 8:
                g["pts"] = np.vstack([g["pts"], s["pts"]])
                break
        else:
            groups.append(dict(s))

    best = None
    for g in groups:
        vx, vy, x0, y0 = cv2.fitLine(g["pts"], cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        u = np.array([vx, vy]) * (1 if vx >= 0 else -1)
        v = np.array([-u[1], u[0]])
        p0 = np.array([x0, y0])
        t = (g["pts"] - p0) @ u
        a, b = t.min(), t.max()
        tilt = np.degrees(np.arctan2(u[1], u[0]))
        if b - a < 60 or abs(tilt) > MAX_TILT_DEG:
            continue
        # The screen just below the bar is dark.
        vals = []
        for tt in np.linspace(a, b, 9):
            for dd in (25, 45, 65):
                x, y = p0 + u * tt + v * dd
                if 0 <= x < img.shape[1] and SEARCH_TOP <= y < img.shape[0]:
                    vals.append(hsv[int(y) - SEARCH_TOP, int(x), 2])
        if not vals or np.median(vals) > 80:
            continue
        e1, e2 = p0 + u * a, p0 + u * b
        if best is None or (e1 + e2)[1] < (best["e1"] + best["e2"])[1]:  # topmost: the top bar, not the bottom one
            best = {"e1": e1, "e2": e2, "u": u, "v": v, "length": float(b - a), "tilt": float(tilt)}
    if best is None:
        return None

    # Restore an end hidden by the gaze cursor or cut off by the frame border.
    short = BAR_LEN_PX - best["length"]
    if short > 15:
        def hidden(e):
            near_cursor = cursor_xy is not None and np.hypot(cursor_xy[0] - e[0], cursor_xy[1] - e[1]) < 30
            return near_cursor or e[0] < 6 or e[0] > img.shape[1] - 7
        if hidden(best["e2"]) and not hidden(best["e1"]):
            best["e2"] = best["e2"] + best["u"] * short
        elif hidden(best["e1"]) and not hidden(best["e2"]):
            best["e1"] = best["e1"] - best["u"] * short
    return best


def display_quad(bar: dict) -> np.ndarray:
    """Screen corners (top-left, top-right, bottom-right, bottom-left) in video pixels; may reach past the
    bottom of the frame when the display is half out of view."""
    u, v = bar["u"], bar["v"]
    tl = bar["e1"] - u * MARGIN_PX - v * MARGIN_PX
    tr = bar["e2"] + u * MARGIN_PX - v * MARGIN_PX
    return np.array([tl, tr, tr + v * DEPTH_PX, tl + v * DEPTH_PX], np.float32)


def draw_display(img, quad, color=(255, 0, 255), label="speed display"):
    q = np.asarray(quad).reshape(4, 2).astype(np.int32)
    cv2.polylines(img, [q], True, color, 3, cv2.LINE_AA)
    cv2.putText(img, label, (int(q[0][0]), max(int(q[0][1]) - 10, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                cv2.LINE_AA)
    return img
