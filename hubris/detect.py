"""Step 4: vehicle detection with a pretrained YOLO11 (ultralytics, COCO weights).

COCO car=2, bus=5, truck=7. The class list (README.md) has only car and truck, so bus is folded into
truck; the original COCO name is kept in `coco_class`.
"""
from __future__ import annotations

import cv2
import pandas as pd

COCO_TO_CLASS = {2: "car", 5: "truck", 7: "truck"}
COLS = ["frame", "category", "class", "coco_class", "x1", "y1", "x2", "y2", "conf"]
BOX_COLORS = {"car": (0, 200, 0), "truck": (0, 140, 255)}


def load_model(weights: str = "yolo11l.pt"):
    from ultralytics import YOLO  # imported here so the rest of the package works without it
    return YOLO(weights)


def detect_batch(model, frame_ids, images, imgsz=1280, conf=0.25, device=None, fp16=True) -> pd.DataFrame:
    """Run the model on a batch of BGR frames; one row per vehicle box, in full-frame pixels."""
    results = model.predict(images, imgsz=imgsz, conf=conf, classes=list(COCO_TO_CLASS),
                            device=device, quantize=16 if fp16 else 32, verbose=False)
    rows = []
    for fid, r in zip(frame_ids, results):
        b = r.boxes
        if b is None or len(b) == 0:
            continue
        xyxy, cls, cf = b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int), b.conf.cpu().numpy()
        for (x1, y1, x2, y2), c, p in zip(xyxy, cls, cf):
            rows.append((fid, "object", COCO_TO_CLASS[c], model.names[c], x1, y1, x2, y2, p))
    return pd.DataFrame(rows, columns=COLS)


def draw_boxes(img, dets: pd.DataFrame):
    for cls, x1, y1, x2, y2, p in dets[["class", "x1", "y1", "x2", "y2", "conf"]].itertuples(index=False):
        color = BOX_COLORS.get(cls, (255, 255, 255))
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, color, 2, cv2.LINE_AA)
        cv2.putText(img, f"{cls} {p:.2f}", (p1[0], max(p1[1] - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return img
