"""Step 5 of the HUBRIS pipeline: label every gaze sample with what the driver looked at.

Usage:
    python run_label.py --sync-dir out/video1 --video ../varjo_capture1.mp4

Needs the outputs of steps 1-4d in --sync-dir: gaze_synced.csv, calibration.json,
detections_yolo11l_tracked.parquet, seg_m2f/ (+ seg_m2f.json) and speed_display.csv. CPU only.
The rules and the group's choices are the settings at the top of hubris/label.py. --video is only needed
for the check images.

Outputs in --sync-dir:
    gaze_labels.csv|parquet   one row per Varjo sample: raw_csv_row, video_time_s, display_frame, x_px, y_px,
                              label, category, class, track_id, overlap, ambiguous, touches, reason
    gaze_labels.json          label counts, time per label, ambiguous share, most-looked-at vehicles
    check/label_*.jpg         example frames per label, with the gaze circle and what it touched
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd

from hubris import label, overlay, road, speed

CORNERS = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]


def label_chunk(job):
    """Label all samples of a list of frames (runs in a worker process)."""
    sync_dir, frames, samples, boxes, quads, id2label, radius = job
    rows = []
    for fid in frames:
        seg = cv2.imread(f"{sync_dir}/seg_m2f/{fid:06d}.png", cv2.IMREAD_UNCHANGED)
        layers = road.road_layers(seg, id2label)
        masks = {"speed display": label.display_mask(quads.get(fid), seg.shape),
                 "road line": layers["road line"], "road edge": layers["road edge"],
                 "guard rail": layers["guard rail"], "road surface": layers["road surface"]}
        s = samples[samples.display_frame == fid]
        b = boxes[boxes.frame == fid]
        for idx, res in zip(s.index, label.label_frame(s[["x_px", "y_px"]].to_numpy(float), radius, masks, b)):
            rows.append((idx, *res))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sync-dir", required=True)
    ap.add_argument("--video", help="video, only for the check images")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--examples", type=int, default=2, help="check images per label")
    args = ap.parse_args()
    d = args.sync_dir

    cal = json.load(open(f"{d}/calibration.json"))
    radius = cal["px_per_deg"] * label.RADIUS_DEG
    id2label = {int(k): v for k, v in json.load(open(f"{d}/seg_m2f.json"))["id2label"].items()}
    gaze = pd.read_csv(f"{d}/gaze_synced.csv")
    boxes = pd.read_parquet(f"{d}/detections_yolo11l_tracked.parquet")
    boxes = boxes[boxes.track_len >= label.MIN_TRACK_LEN]
    dropped = label.interior_tracks(boxes, d, id2label)
    boxes = boxes[~boxes.track_id.isin(dropped)]
    print(f"dropped {len(dropped)} vehicle tracks that sit on the car's own interior: {dropped}")
    sp = pd.read_csv(f"{d}/speed_display.csv")
    quads = {int(f): q for f, q in zip(sp.frame, sp[CORNERS].to_numpy(float)) if np.isfinite(q).all()}

    # Samples that can't be labelled from the picture.
    out = gaze[["raw_csv_row", "video_time_s", "display_frame", "x_px", "y_px"]].copy()
    out["category"], out["class"], out["reason"] = "other", "invalid", ""
    out["track_id"], out["overlap"], out["on_target"], out["ambiguous"], out["touches"] = -1, 0.0, False, False, ""
    out.loc[gaze.video_time_s < 0, "reason"] = "before the video starts"
    out.loc[(gaze.video_time_s >= 0) & ~gaze.valid, "reason"] = "blink or tracking lost"
    out.loc[gaze.valid & ~gaze.inside_video, "reason"] = "gaze outside the video image"
    ok = gaze.valid & gaze.inside_video
    lo, hi = label.DRIVE_FRAMES
    fog = ok & ((gaze.display_frame < lo) | (gaze.display_frame > hi))
    out.loc[fog, "class"] = "fog"
    out.loc[fog, "reason"] = "fog: road not visible"
    todo = ok & ~fog

    # Label the rest, frame by frame, in parallel.
    samples = gaze.loc[todo, ["display_frame", "x_px", "y_px"]]
    frames = np.unique(samples.display_frame.astype(int))
    print(f"{int(todo.sum())} samples in {len(frames)} frames to label, radius {radius:.1f} px")
    t0 = time.time()
    chunks = np.array_split(frames, max(1, len(frames) // 300))
    jobs = [(d, c, samples[samples.display_frame.isin(c)], boxes[boxes.frame.isin(c)],
             {f: quads[f] for f in c if f in quads}, id2label, radius) for c in chunks]
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for i, r in enumerate(pool.map(label_chunk, jobs)):
            rows += r
            if (i + 1) % 10 == 0:
                print(f"{i + 1}/{len(jobs)} chunks, {time.time() - t0:.0f} s", flush=True)
    res = pd.DataFrame(rows, columns=["idx", "category", "class", "track_id", "overlap", "on_target", "ambiguous",
                                      "touches", "label_point"]).set_index("idx")
    out["label"] = out.category + " - " + out["class"]
    out["label_point"] = out["label"]
    out.loc[res.index, res.columns] = res
    out["label"] = out.category + " - " + out["class"]
    cols = ["raw_csv_row", "video_time_s", "display_frame", "x_px", "y_px", "label", "category", "class",
            "track_id", "overlap", "on_target", "ambiguous", "touches", "label_point", "reason"]
    out = out[cols].astype({"track_id": int, "on_target": bool, "ambiguous": bool, "overlap": float})
    out.to_parquet(f"{d}/gaze_labels.parquet", index=False)
    out.round({"video_time_s": 4, "x_px": 1, "y_px": 1, "overlap": 3}).to_csv(f"{d}/gaze_labels.csv", index=False)

    # Summary for the report. 200 samples per second.
    drive = out[todo]
    veh = drive[drive.category == "object"]
    report = {
        "samples": int(len(out)), "labelled_in_drive": int(len(drive)),
        "settings": {"radius_deg": label.RADIUS_DEG, "radius_px": round(radius, 1), "drive_frames": label.DRIVE_FRAMES,
                     "min_track_len": label.MIN_TRACK_LEN, "priority": label.PRIORITY},
        "dropped_interior_tracks": dropped,
        "all_samples_pct": out.label.value_counts(normalize=True).mul(100).round(2).to_dict(),
        "drive_pct": drive.label.value_counts(normalize=True).mul(100).round(2).to_dict(),
        "drive_seconds": drive.label.value_counts().div(200).round(1).to_dict(),
        "drive_on_target_pct_by_label": drive.groupby("label").on_target.mean().mul(100).round(1).to_dict(),
        "drive_ambiguous_pct": round(100 * drive.ambiguous.mean(), 1),
        "drive_pct_gaze_point_only": drive.label_point.value_counts(normalize=True).mul(100).round(2).to_dict(),
        "vehicles_looked_at": int(veh.track_id.nunique()),
        "top_vehicles_seconds": {f"{c} #{t}": round(n / 200, 1) for (t, c), n in
                                 veh.groupby(["track_id", "class"]).size().sort_values(ascending=False).head(5).items()},
        "seconds_to_label": round(time.time() - t0, 1),
    }
    with open(f"{d}/gaze_labels.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

    if args.video:
        write_examples(args, d, drive, gaze, boxes, quads, id2label, radius)
    print(f"done -> {d}/gaze_labels.csv")


def write_examples(args, d, drive, gaze, boxes, quads, id2label, radius):
    """A few frames per label: everything found, the gaze circle and the label it got."""
    from hubris.video import read_frames
    rng = np.random.default_rng(0)
    picks = {}
    for lab_name, g in drive.groupby("label"):
        for i in rng.choice(g.index, min(args.examples, len(g)), replace=False):
            picks.setdefault(int(g.loc[i, "display_frame"]), []).append(i)
    os.makedirs(f"{d}/check", exist_ok=True)
    for fid, base in read_frames(args.video, sorted(picks)):
        img = base
        seg = cv2.imread(f"{d}/seg_m2f/{fid:06d}.png", cv2.IMREAD_UNCHANGED)
        road.draw_layers(img, road.road_layers(seg, id2label), alpha=0.3)
        for tid, cls, x1, y1, x2, y2 in boxes[boxes.frame == fid][["track_id", "class", "x1", "y1", "x2", "y2"]] \
                .itertuples(index=False):
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(img, f"{cls} #{tid}", (int(x1), max(int(y1) - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2, cv2.LINE_AA)
        if fid in quads:
            speed.draw_display(img, quads[fid], color=(0, 165, 255))
        for i in picks[fid]:
            r, one = drive.loc[i], img.copy()
            text = r.label + (f" #{r.track_id}" if r.track_id >= 0 else "") + (" (ambiguous)" if r.ambiguous else "")
            overlay.draw_gaze(one, r.x_px, r.y_px, radius, label=text)
            name = r.label.replace(" - ", "_").replace(" ", "-")
            overlay._stamp(one, f"frame {fid}  t={r.video_time_s:7.2f}s   label: {text}   touches: {r.touches or '-'}")
            cv2.imwrite(f"{d}/check/label_{name}_{fid:06d}.jpg", one, [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    main()
