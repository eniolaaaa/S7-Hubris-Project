"""Step 4b: road surface, road lines, road edge (and guard rail) from semantic segmentation.

Model: Mask2Former Swin-L trained on Mapillary Vistas (65 classes). Unlike Cityscapes it has separate
lane-marking, guard-rail, terrain and ego-vehicle classes, and on this simulator it stops the road at the
grass verge; SegFormer/Cityscapes spread "road" over the verges and grass median.

The raw label map (Vistas class id per pixel, at SEG_SIZE) is saved per frame; `road_layers` turns it into
the classes we label with, so their definitions can change without rerunning the model.
"""
from __future__ import annotations

import cv2
import numpy as np

from .calibrate import CURSOR_HSV_HI, CURSOR_HSV_LO, detect_cursor

MODEL = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
SEG_SIZE = (1024, 768)  # (w, h): half of the 2048x1536 video

# Vistas class names (model.config.id2label).
SURFACE = ["Road", "Service Lane", "Lane Marking - General", "Lane Marking - Crosswalk",
           "Manhole", "Catch Basin", "Pothole", "Bike Lane"]
LINE = ["Lane Marking - General", "Lane Marking - Crosswalk"]
GUARD_RAIL = ["Guard Rail"]
# The road edge is where the road surface meets one of these (not vehicles, the ego car or the sky).
OFFROAD = ["Curb", "Fence", "Guard Rail", "Barrier", "Wall", "Sidewalk", "Terrain", "Vegetation",
           "Sand", "Snow", "Water", "Mountain", "Building", "Pedestrian Area", "Parking", "Curb Cut"]


def _ids(names, id2label):
    lut = {v: int(k) for k, v in id2label.items()}
    return [lut[n] for n in names]


def _disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def remove_cursor(img, cursor_xy, reach_px=40):
    """Inpaint the burned-in teal gaze cursor at `cursor_xy` (None = not visible, nothing to do).

    Left in, the segmenter sees it as an object on the road: it punches a hole in the road surface, with a
    road edge around it, exactly where the driver is looking. Only teal pixels near the cursor are touched,
    so a teal car or the speedometer bar stay as they are. Works on a small patch: same result as
    inpainting the full frame, ~10x faster.
    """
    if cursor_xy is None:
        return img
    h, w = img.shape[:2]
    x, y = int(cursor_xy[0]), int(cursor_xy[1])
    r = reach_px + 16  # room for the mask dilation and the inpainting radius
    x0, y0, x1, y1 = max(x - r, 0), max(y - r, 0), min(x + r + 1, w), min(y + r + 1, h)
    patch = img[y0:y1, x0:x1]
    teal = cv2.inRange(cv2.cvtColor(patch, cv2.COLOR_BGR2HSV), CURSOR_HSV_LO, CURSOR_HSV_HI)
    near = np.zeros_like(teal)
    cv2.circle(near, (x - x0, y - y0), reach_px, 255, -1)
    mask = cv2.dilate(teal & near, _disk(4))
    out = img.copy()
    out[y0:y1, x0:x1] = cv2.inpaint(patch, mask, 5, cv2.INPAINT_TELEA)
    return out


class RoadSegmenter:
    def __init__(self, model_name=MODEL, device="cuda", fp16=True):
        import torch
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        self.torch = torch
        self.proc = AutoImageProcessor.from_pretrained(model_name)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name).to(device).eval()
        self.dtype = torch.float16 if fp16 and device != "cpu" else torch.float32
        self.model = model.to(self.dtype)
        self.device = device
        self.id2label = {int(k): v for k, v in model.config.id2label.items()}

    def segment(self, images_bgr, cursors=None):
        """Vistas class id per pixel (uint8, SEG_SIZE) for a list of full-size BGR frames.

        `cursors`: gaze-cursor (x, y) per frame, or None where it is not visible (e.g. from cursor_track.csv);
        detected here when not given.
        """
        if cursors is None:
            cursors = [detect_cursor(im) for im in images_bgr]
        small = [cv2.resize(remove_cursor(im, c), SEG_SIZE, interpolation=cv2.INTER_AREA)[:, :, ::-1].copy()
                 for im, c in zip(images_bgr, cursors)]
        pv = self.proc(images=small, return_tensors="pt", do_resize=False)["pixel_values"]
        with self.torch.no_grad():
            out = self.model(pixel_values=pv.to(self.device, self.dtype))
            segs = self.proc.post_process_semantic_segmentation(out, target_sizes=[SEG_SIZE[::-1]] * len(small))
        return [s.to("cpu").numpy().astype(np.uint8) for s in segs]


def road_layers(seg: np.ndarray, id2label: dict, edge_px: int = 2, min_area_frac: float = 0.02) -> dict:
    """Binary masks (uint8, same size as seg) for road surface, road line, road edge and guard rail.

    - road surface: road-like classes; small blobs (< min_area_frac of the largest) are dropped, which removes
      e.g. road seen in the rear-view mirror.
    - road line: lane markings on that surface.
    - road edge: boundary of the surface where it meets an off-road class, `edge_px` wide on each side.
    """
    surface = np.isin(seg, _ids(SURFACE, id2label)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(surface, connectivity=8)
    if n > 1:
        area = st[1:, cv2.CC_STAT_AREA]
        keep = np.r_[False, area >= min_area_frac * area.max()]
        surface = keep[lab].astype(np.uint8)

    line = (np.isin(seg, _ids(LINE, id2label)) & (surface > 0)).astype(np.uint8)

    offroad = np.isin(seg, _ids(OFFROAD, id2label)).astype(np.uint8)
    boundary = surface - cv2.erode(surface, _disk(1))
    edge = (boundary > 0) & (cv2.dilate(offroad, _disk(3)) > 0)
    edge = cv2.dilate(edge.astype(np.uint8), _disk(edge_px))

    rail = np.isin(seg, _ids(GUARD_RAIL, id2label)).astype(np.uint8)
    return {"road surface": surface, "road line": line, "road edge": edge, "guard rail": rail}


LAYER_COLORS = {"road surface": (128, 64, 128), "guard rail": (255, 160, 0),
                "road edge": (0, 255, 255), "road line": (0, 0, 255)}


def draw_layers(img, layers: dict, alpha=0.45):
    """Overlay the layers on a full-size frame (masks are upscaled from SEG_SIZE)."""
    h, w = img.shape[:2]
    for name in ["road surface", "guard rail", "road edge", "road line"]:
        m = cv2.resize(layers[name], (w, h), interpolation=cv2.INTER_NEAREST) > 0
        a = alpha if name in ("road surface", "guard rail") else 0.9
        img[m] = ((1 - a) * img[m] + a * np.array(LAYER_COLORS[name])).astype(np.uint8)
    return img
