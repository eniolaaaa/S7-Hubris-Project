"""Run every step of the pipeline on one recording.

Usage:
    python run_all.py --video ../varjo_capture2.mp4 --gaze ../varjo_gaze_output2.csv --out out/video2

Steps whose output already exists are skipped, so after a crash or a laptop that went to sleep, just run
the same command again. Order, with rough times on the RTX 4070 laptop:
    0  cursor track   detect_cursor_chunk.py (in chunks)   cursor_track.csv        ~10 min
    1  sync           run_sync.py                          gaze_synced.csv         ~1 min
    2  vehicles       run_detect.py  (GPU)                 detections_yolo11l.*    ~15 min
    3  tracking       run_track.py                         ..._tracked.*           ~1 min
    4  road           run_road.py    (GPU)                 seg_m2f/                ~50 min
    5  speed display  run_speed.py                         speed_display.csv       ~5 min
    6  labels         run_label.py                         gaze_labels.csv         ~2 min
    7  demo video     run_demo.py                          demo_*.mp4              ~2 min
Nothing is tuned per video: the gaze calibration, the drive (fog) range and the demo clip are all found
automatically. --redo forces steps to run again, e.g. --redo labels demo.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

from hubris.video import video_info

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = ["cursor", "sync", "vehicles", "tracking", "road", "speed", "labels", "demo"]
CHUNK = 3000  # frames per cursor-track chunk


def keep_awake(on: bool):
    """Ask Windows not to sleep while the pipeline runs (idle sleep paused a run overnight). Closing the
    lid can still put the laptop to sleep. No effect on other systems."""
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def run(cmd):
    print(">", " ".join(cmd), flush=True)
    subprocess.run([sys.executable, os.path.join(HERE, cmd[0]), *cmd[1:]], check=True, cwd=HERE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--gaze", required=True, help="Varjo gaze CSV of the same recording")
    ap.add_argument("--out", required=True, help="output folder, e.g. out/video2")
    ap.add_argument("--redo", nargs="*", default=[], choices=STEPS, help="run these steps even if done")
    ap.add_argument("--whole-drive-demo", action="store_true", help="demo of the whole drive instead of 60 s")
    args = ap.parse_args()
    o, v, gaze = (os.path.abspath(p) for p in (args.out, args.video, args.gaze))
    os.makedirs(o, exist_ok=True)
    n_frames = video_info(v).n_frames

    done = {
        # A cursor track without a progress file was made outside run_all (e.g. video 1) and is complete.
        "cursor": os.path.exists(f"{o}/cursor_track.done")
                  or (os.path.exists(f"{o}/cursor_track.csv") and not os.path.exists(f"{o}/cursor_track.progress")),
        "sync": os.path.exists(f"{o}/gaze_synced.csv") and os.path.exists(f"{o}/calibration.json"),
        "vehicles": os.path.exists(f"{o}/detections_yolo11l.parquet"),
        "tracking": os.path.exists(f"{o}/detections_yolo11l_tracked.parquet"),
        "road": os.path.exists(f"{o}/seg_m2f.json"),
        "speed": os.path.exists(f"{o}/speed_display.csv"),
        "labels": os.path.exists(f"{o}/gaze_labels.parquet"),
        "demo": bool(glob.glob(f"{o}/demo_*.mp4")),
    }
    t_all = time.time()
    keep_awake(True)
    try:
        run_steps(args, o, v, gaze, n_frames, done)
    finally:
        keep_awake(False)
    print(f"\nall steps done in {(time.time() - t_all) / 60:.1f} min -> {o}")


def run_steps(args, o, v, gaze, n_frames, done):
    for step in STEPS:
        if done[step] and step not in args.redo:
            print(f"[{step}] already done, skipped")
            continue
        print(f"\n[{step}] starting", flush=True)
        t0 = time.time()
        if step == "cursor":
            # Detect the burned-in gaze cursor in every frame, in chunks so a cut-off run loses little.
            track, prog = f"{o}/cursor_track.csv", f"{o}/cursor_track.progress"
            if step in args.redo:
                for p in (track, prog, f"{o}/cursor_track.done"):
                    if os.path.exists(p):
                        os.remove(p)
            start = json.load(open(prog))["next"] if os.path.exists(prog) else 0
            json.dump({"next": start}, open(prog, "w"))   # marks the track as unfinished until .done exists
            for s in range(start, n_frames, CHUNK):
                run(["detect_cursor_chunk.py", "--video", v, "--out", track, "--start", str(s),
                     "--end", str(min(s + CHUNK, n_frames))])
                json.dump({"next": s + CHUNK}, open(prog, "w"))
            open(f"{o}/cursor_track.done", "w").close()
        elif step == "sync":
            run(["run_sync.py", "--video", v, "--gaze", gaze, "--out", o, "--cursor-csv", f"{o}/cursor_track.csv"])
        elif step == "vehicles":
            run(["run_detect.py", "--video", v, "--sync-dir", o])
        elif step == "tracking":
            run(["run_track.py", "--sync-dir", o, "--video", v])
        elif step == "road":
            run(["run_road.py", "--video", v, "--sync-dir", o])
        elif step == "speed":
            run(["run_speed.py", "--video", v, "--sync-dir", o])
        elif step == "labels":
            run(["run_label.py", "--sync-dir", o, "--video", v])
        elif step == "demo":
            run(["run_demo.py", "--sync-dir", o, "--video", v] + (["--whole-drive"] if args.whole_drive_demo else []))
        print(f"[{step}] done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
