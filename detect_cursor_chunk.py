"""Detect the teal cursor in every frame of [start, end) and append to a CSV.

Lets the cursor track be built in chunks when a single long run is not possible:
    python detect_cursor_chunk.py --video v.mp4 --out cursor.csv --start 0 --end 5000
Then pass --cursor-csv cursor.csv to run_sync.py.
"""
import argparse
import os

import cv2

from hubris.calibrate import detect_cursor

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--start", type=int, required=True)
ap.add_argument("--end", type=int, required=True)
a = ap.parse_args()

cap = cv2.VideoCapture(a.video)
cap.set(cv2.CAP_PROP_POS_FRAMES, a.start)
new = not os.path.exists(a.out)
with open(a.out, "a") as f:
    if new:
        f.write("frame,cx,cy\n")
    for fid in range(a.start, a.end):
        ok, img = cap.read()
        if not ok:
            break
        p = detect_cursor(img)
        if p is not None:
            f.write(f"{fid},{p[0]:.1f},{p[1]:.1f}\n")
print("done", a.start, fid)
