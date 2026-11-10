"""Find the teal gaze cursor in frames [start, end) and append the positions to a CSV.

run_all.py calls this in chunks to build cursor_track.csv, so a long run that gets cut off only loses
the current chunk. By hand:
    python detect_cursor_chunk.py --video ../varjo_capture1.mp4 --out out/video1/cursor_track.csv --start 0 --end 3000
Then pass --cursor-csv out/video1/cursor_track.csv to run_sync.py.
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
