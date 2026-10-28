# HUBRIS gaze-to-AOI pipeline

Goal: label every Varjo gaze sample of one simulator video with what the driver was looking at
(`category - class`, e.g. `object - car`).

University group project (EVML, HAN, 3 students) for HAN Automotive Research's HUBRIS driving simulator.
Drivers wore a Varjo Aero VR headset; the eye tracker logs *where* they looked (200 Hz). This pipeline
works out *what* they looked at (car, truck, lane line, speed display, ...) for every gaze sample.

## Scope (agreed in the group meeting of 2026-10-07)
- Make the pipeline work properly on **one complete video** (not all ~70). A second video may be run
  afterwards, untouched, as a generalisation check. Hand-labelled test frames come from video 1 only.
- Labels are two-level: `category - class`.
- Deliverables: the script, the model(s) and settings, output proving it works (labelled gaze file +
  accuracy on hand-labelled test frames), a live demo, an annotated demo video, and a report.

## Classes (MoSCoW)
| Category | Classes | Priority |
|---|---|---|
| object | car, truck | Must |
| road | road line, road edge | Must |
| interior | speed display | Must |
| road | dotted vs solid line, guard rail | Should |
| interior | mirror | Should |
| object part | car/truck indicator, wheel | Should |
| environment | bridge | Could |
| other | other, invalid (blink / tracking lost) | always |

Won't: traffic signs, traffic lights, the rating popup ("Please rate your Perceived Risk..."); gaze on the
popup is labelled `other`. Lane assignment of vehicles (ego / adjacent / beyond) is a nice extra.

## Status
| Step | What | Code | State |
|---|---|---|---|
| 1 | Clean gaze (status == 2, drop pre-roll) | `hubris/gaze_io.py` | done, tested on synthetic data |
| 2 | Time sync: sample to video frame, `floor(t * fps)` | `hubris/gaze_io.py` | done, tested on synthetic data |
| 3 | Pixel mapping: teal cursor + RANSAC affine, frame-lag search, px per degree | `hubris/calibrate.py` | done, validated on video 1 |
| 3b | Overlay check (snapshots + short clip) | `hubris/overlay.py` | done |
| | **Video 1 result** (varjo_capture1.mp4 + varjo_gaze_output1.csv) | `out/video1/` | median error 4.2 px (~0.16 deg), cursor lags CSV by 2 frames |
| 4 | Car/truck detection: YOLO11l (COCO), imgsz 1280, conf 0.25 | `hubris/detect.py`, `run_detect.py` | done on video 1 |
| | **Video 1 result** | `out/video1/detections_yolo11l.*` | 18,966 frames, 69,379 boxes (62,173 car / 7,206 truck), vehicle in 93.7% of frames |
| 4b | Road surface, road lines, road edge, guard rail: Mask2Former (Mapillary Vistas) | `hubris/road.py`, `run_road.py` | done on video 1 |
| | **Video 1 result** | `out/video1/seg_m2f/` | 18,966 label maps (224 MB); road, lines, edge found in every sampled frame of the drive (frames ~2,090-20,460) |
| 4c | Car/truck tracking: ByteTrack on the saved boxes (CPU, ~30 s) | `hubris/track.py`, `run_track.py` | done on video 1 |
| | **Video 1 result** | `out/video1/detections_yolo11l_tracked.*` | 3,177 tracks; 96% of boxes in tracks of 3+ frames; 363 car/truck flips fixed; 3,976 gap boxes filled |
| 4d | Speed display: anchored on the screen's teal light bar (CPU, ~5 min) | `hubris/speed.py`, `run_speed.py` | done on video 1 |
| | **Video 1 result** | `out/video1/speed_display.csv` | found in 100% of drive frames (2,090-20,460) incl. all frames with gaze low in the view; missed only in the fog |
| 5 | Label every gaze sample: 1.5 deg circle, fixed priority (CPU, ~1 min) | `hubris/label.py`, `run_label.py` | done on video 1 |
| | **Video 1 result** | `out/video1/gaze_labels.csv` | drive: 55% car, 21% truck, 10% road line, 10% other, 1.1% speed display; 91 vehicles looked at |
| 6 | Annotated demo video (H.264 .mp4) | `hubris/demo.py`, `run_demo.py` | done: 60 s highlight of video 1 |
| 5b | Fixations, dwell time | | next |

## Next steps
1. ~~Speed display~~ done (step 4d).
2. ~~Label every gaze sample~~ done (step 5).
3. Hand-label ~200-300 test frames from video 1, spread across the drive, for accuracy numbers (can start now).
4. Fixations (I-VT or I-DT) on the 200 Hz stream; label each fixation; dwell time, time to first fixation,
   road <-> speed display switches.
5. ~~Annotated demo video~~ done (step 6).
6. Report. Then run the finished pipeline on video 2, untouched.

Should-haves after that: dotted vs solid line (from the shape of the line mask), mirror (a fixed region
relative to the cockpit is likely better than the segmentation, which calls the mirror "Bridge"), guard rail
(already segmented in step 4b).

Decided by the group (2026-10-27): drop the fog (`other - fog`), plain road is `road - surface`, fixed priority
instead of largest overlap, guard rail included. Still open: report the circle (tolerant) or the gaze-point
(strict) labels, or both? See step 5 below; a table over several circle sizes would show how much it matters.

## Open in VS Code (Windows)
1. Keep this folder next to the data: `Downloads\hubris_pipeline`, with `varjo_capture1.mp4` and
   `varjo_gaze_output1.csv` in `Downloads`.
2. In VS Code: File > Open Folder > `Downloads\hubris_pipeline`. Install the recommended Python extension when asked.
3. Terminal > Run Task... > **Set up Python environment** (once). This creates `.venv` and installs the requirements.
   Then pick the `.venv` interpreter if VS Code asks (Ctrl+Shift+P > "Python: Select Interpreter").
4. Run and Debug (Ctrl+Shift+D) > **2. Sync video 1 (reuse cursor track, fast)** > play.
   **1. Sync video 1 (full run)** redoes the cursor detection over the whole video (several minutes).

## Run on one video
```
pip install -r requirements.txt
python run_sync.py --video <video.mp4> --gaze <varjo_gaze.csv> --out out/<participant>
```
Check `out/<participant>/check/*.jpg`: the magenta cross (gaze from the CSV) should sit on the teal
cursor burned into the video. `sync_report.json` and `calibration.json` hold the numbers for the report
(fit error in pixels, frame lag, pixels per degree, % valid samples).

## Video 1 findings (2026-10-07)
- 2048x1536, 30 fps, 21,069 frames, 702.3 s; gaze 140,374 samples, 94.6% valid, 6.7 samples per frame.
- Affine fit: 6,355 / 8,465 inlier pairs, median 4.2 px, RMSE 5.5 px; 80% of all pairs within 1 deg (26 px).
  Big errors are frames mid-saccade (the cursor jumps between the frame's samples) and blinks.
- The burned-in cursor is drawn **2 frames (~67 ms) after** the CSV timestamp; `display_frame` in
  `gaze_synced.csv` already includes this.
- First ~47 s (frames 0-1400) is the foggy pre-drive scene; there the cursor sits 30-70 px right of the
  computed gaze. The fit ignores it (RANSAC); treat that segment as pre-drive / exclude from analysis.
  (Found in step 4b: the fog lasts until ~frame 2,090, see below.)
- A 2nd/3rd-order polynomial map gained <0.1 px overall (about 2 px at the bottom of the view), so affine stays.
- 1.5 deg gaze circle = 39 px (26.3 px/deg).
- Long runs on eni's device get cut off; use `detect_cursor_chunk.py` in ~7,000-frame chunks, then
  `run_sync.py --cursor-csv`.

## Step 4: car/truck detection (2026-10-07)
```
python run_detect.py --video ..\varjo_capture1.mp4 --sync-dir out\video1
```
(VS Code: Run and Debug > **3. Detect cars/trucks video 1**.) Detects on every `display_frame` with valid gaze,
from frame 1400. Resumable: parts go to `out/video1/detections_yolo11l/`; rerun after a cut-off and only the
unfinished chunk is redone. ~21 frames/s on the RTX 4070 laptop GPU (~15 min for video 1).
- Output `detections_yolo11l.parquet|csv`: `frame, category, class, coco_class, x1, y1, x2, y2, conf`
  (frame = video frame, i.e. match it to `display_frame`). COCO bus (54 boxes) is folded into `truck`.
- Frames with no vehicle have no rows; `detections_yolo11l.json` lists the frame range and counts.
- GPU: `pip install -r requirements.txt` gives CPU-only torch on Windows. For CUDA first run
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`.
- **The end of video 1 is foggy too**: fog from ~frame 20,470 (t ~ 682 s), black from ~20,960. Exclude it
  like the pre-drive segment.
- 20% of boxes are < 20 px tall (far-away vehicles); fine for detection, but they rarely matter for gaze.

## Step 4b: road surface, road lines, road edge, guard rail (2026-10-13)
```
python run_road.py --video ..\varjo_capture1.mp4 --sync-dir out\video1
```
(VS Code: Run and Debug > **4. Segment road lines/edge video 1**.) Same frames as step 4. ~7 frames/s on the
RTX 4070 laptop GPU (~45 min for video 1). Resumable: one PNG per frame; rerun and finished frames are skipped.
- Model: Mask2Former Swin-L trained on Mapillary Vistas (`facebook/mask2former-swin-large-mapillary-vistas-semantic`),
  run at 1024x768. It has separate lane-marking, guard-rail, terrain and ego-vehicle classes. Tried first and
  dropped: SegFormer-B2/B5 (Cityscapes), also with colour/texture clean-up; they spread "road" over the grass verges
  and median.
- The teal gaze cursor is inpainted before segmenting (positions from `cursor_track.csv`); left in, it punched a
  hole with a fake road edge into the road exactly where the driver looked.
- Output `seg_m2f/<frame>.png`: the raw Vistas class id per pixel at 1024x768 (half the video size; scale x2 to
  video pixels). `hubris.road.road_layers(seg, id2label)` turns it into masks, so the definitions below can change
  without rerunning the model (`id2label` is in `seg_m2f.json`):
  - **road surface**: Road, Service Lane (mostly the asphalt hard shoulder), lane markings, manholes; small blobs
    such as the road seen in the rear-view mirror are dropped.
  - **road line**: lane markings on the road surface (dashed and solid).
  - **road edge**: where the road surface meets terrain, vegetation, guard rail, barrier, curb, wall, etc.
    (2 px wide on each side at 1024x768). Not where it meets vehicles, the ego car or the sky.
  - **guard rail**: Vistas guard rail (a "Should" class).
- Video 1: present in every sampled frame of the drive. Known weak spot: a dry brown verge is sometimes taken for
  road/service lane, and the edge then runs through the verge (e.g. frame 4962, right side). The hand-labelled
  test frames will show how often.
- **The fog at the start lasts until ~frame 2,090 (t ~ 70 s)**, not 1,400: frames 1,400-2,060 are thick fog with
  no road or vehicles visible (YOLO finds nothing there either), although the car is already driving at 84-94 km/h
  and the speed display is visible.
- Check `check/road_frame_*.jpg`: purple = road surface, red = road line, yellow = road edge, blue = guard rail.

## Step 5: label every gaze sample (2026-10-27)
```
python run_label.py --sync-dir out\video1 --video ..\varjo_capture1.mp4
```
(VS Code: Run and Debug > **7. Label gaze video 1**.) CPU only, ~1 min with 8 worker processes. Needs the outputs of
steps 1-4d. The rules are the settings at the top of `hubris/label.py`.
- For each valid sample in the drive (frames 2,090-20,460): a 1.5 deg circle (39 px) around (`x_px`, `y_px`) on
  `display_frame`. The first thing in this order that the circle touches gives the label:
  speed display -> car / truck -> road line -> road edge -> guard rail -> road surface -> `other - other`.
  Two vehicles: the one with most overlap. Vehicles only from tracks of 3+ frames.
- Other samples: `other - invalid` (blink / tracking lost, before the video, gaze outside the image) or
  `other - fog` (start and end of the video).
- Road patches with the car's own interior just above them are dropped first (`hubris/road.py`): when the driver
  looks down, the segmentation sometimes calls the footwell "road".
- Vehicle tracks whose boxes are mostly centred on the car's own interior are dropped too (10 tracks in video 1,
  listed in `gaze_labels.json`): YOLO sometimes boxes the bottom edge of the windscreen as a wide, flat "car".
- Output `gaze_labels.csv`: one row per Varjo sample (140,374): `label` (`category - class`), `track_id` of the
  vehicle, `overlap` (share of the circle on the labelled thing), `on_target` (the gaze point itself is on it),
  `ambiguous` (the circle touches several things and the gaze point is not on the chosen one, so the order
  decided), `touches` (everything the circle touched), `label_point` (strict label from the gaze point alone)
  and `reason` for samples that could not be labelled. `gaze_labels.json` has the summary numbers.
- Video 1, drive samples (117,087 = 585 s):

  | Label | Circle (1.5 deg) | Gaze point only |
  |---|---|---|
  | object - car | 55.3% | 27.6% |
  | object - truck | 21.5% | 8.5% |
  | road - road line | 10.3% | 10.0% |
  | other - other | 10.2% | 17.0% |
  | interior - speed display | 1.14% | 1.05% |
  | road - guard rail / road edge / surface | 0.6% / 0.6% / 0.5% | 0.5% / 3.2% / 32.2% |

- **The circle size matters a lot.** This driver mostly looks far ahead, where vehicles are small (median 67 px
  tall, the circle is 79 px across), so the circle often touches a car, a lane line and the road edge at once.
  For half of the `car` labels the gaze point itself is just next to the car. The speed display is clear-cut
  (gaze point on it in 92% of its samples). 43.7% of drive samples are `ambiguous`. The hand-labelled test
  frames will show which labelling is closer to the truth.
- 91 different vehicles were looked at; most: truck #809 (63 s), car #88 (44 s), car #485 (41 s).
- Gaze on the rating popup ("Please rate your Perceived Risk...") ends up as `other - other`, as agreed.
- Check `check/label_*.jpg`: two example frames per label with everything found, the gaze circle and its label.

## Step 6: annotated demo video (2026-10-27)
```
python run_demo.py --sync-dir out\video1 --video ..\varjo_capture1.mp4                # 60 s highlight
python run_demo.py --sync-dir out\video1 --video ..\varjo_capture1.mp4 --whole-drive  # frames 2,090-20,460
```
(VS Code: Run and Debug > **8. Demo video, 60 s highlight** or **9. Demo video, whole drive**.) Needs steps 1-5.
- Default `--style clean`: only what the driver looks at is highlighted (the vehicle box filled with its track
  number, the speed display, or for road labels just the line / edge / guard rail / road near the gaze), other
  vehicles get a thin outline. `--style full` draws everything found (all road lines, edges, guard rails, road
  surface, every box with its number) and adds `_full` to the file name.
- Every frame has the gaze circle, a banner with the current label, and a colour-coded timeline of the labels
  along the bottom with a cursor and a one-line key.
- A frame's label is the most common label of its 6-7 gaze samples. During blinks (frames without gaze) the
  last road map is kept, so the overlay does not flicker.
- Default clip: 10:05-11:05 (frames 18,150-19,949), the minute with the most varied gaze (27 s truck, 13 s car,
  3 s road line, 3 s speed display, 10 s rating popup). `--start-frame` and `--seconds` pick another one.
- Output `out/video1/demo_<start>-<end>.mp4`: H.264, 1280x960 at 30 fps (plays in PowerPoint, browsers and the
  Windows player), written with the ffmpeg bundled in the `imageio-ffmpeg` package.

## Step 4d: speed display (2026-10-27)
```
python run_speed.py --video ..\varjo_capture1.mp4 --sync-dir out\video1
```
(VS Code: Run and Debug > **6. Find speed display video 1**.) CPU only, ~60 frames/s (~5 min for video 1).
- The display moves with the driver's head: it tilts up to ~23 deg when they look down, and in normal
  driving it is half cut off at the bottom of the view. A fixed region does not work.
- Anchor: the pale-teal light bar along the top of the screen. It stays ~225 px long in every head position.
  `hubris/speed.py` finds thin, straight, pale-teal stripes with a dark screen below them (this skips the
  glowing steering-wheel rim, the gaze cursor, which is the same hue but far more saturated, and the sky),
  glues pieces split by the gaze cursor back together and restores an end hidden by the cursor or the frame
  border. The screen is a box hanging below the bar: bar + 20 px each side, 380 px deep.
- Tried first and dropped: SIFT feature matching against a reference frame (the dark dashboard gives only
  5-7 matching features).
- Output `speed_display.csv`: per frame `found`, `bar_len`, `tilt_deg` and the screen corners `x1,y1`
  (top-left) to `x4,y4` (bottom-left) in video pixels; the box may reach below the frame. Gaps of up to 5
  frames would be interpolated (`interpolated`); video 1 needed none.
- Video 1: found in 100% of the 17,769 drive frames and in all 271 frames where the gaze is low in the view;
  no jumps over 60 px between consecutive frames. Missed in the fog (1,174 frames): the fog washes out the
  bar, although the display is still faintly visible there.
- Check `check/speed_frame_*.jpg`: magenta box = speed display.

## Step 4c: car/truck tracking (2026-10-13)
```
python run_track.py --sync-dir out\video1 --video ..\varjo_capture1.mp4
```
(VS Code: Run and Debug > **5. Track cars/trucks video 1**.) CPU only, ~30 s: ByteTrack (ultralytics' default
tracker, default settings) runs on the boxes saved in step 4, so YOLO is not rerun and the GPU is not used.
`--video` is only needed for the check clip.
- Output `detections_yolo11l_tracked.parquet|csv`: the step 4 columns plus `track_id`; `class` is now one class
  per track (confidence-weighted vote; YOLO's class in that frame is in `class_frame`); `interpolated` marks a box
  filled in for a gap of up to 10 frames where YOLO missed the vehicle (no `conf`); `track_len` = frames with a
  real detection.
- Video 1: 3,177 tracks (2,773 car, 404 truck). 2,672 of them last one frame, but they hold only ~4% of the
  boxes: mostly tiny far-away cars (median 16 px tall, conf ~0.3) and false detections. 96% of boxes are in
  tracks of 3+ frames, 92% in tracks of 1 s or longer.
- **Use `track_len >= 3` when labelling gaze.** It drops unreliable boxes, e.g. YOLO sometimes boxes the ego
  car's dashboard as a "car" (around frames 10,230-10,390, conf ~0.3); those never form a track.
- Only boxes with conf >= 0.25 were saved, so ByteTrack's low-score matching stage has nothing to use and it
  works like SORT (Kalman filter + IoU). 18 boxes right at the 0.25 threshold got no track.
- Check `check/track_clip.mp4` (15 s from frame 9000): ids stay on the same vehicle through head movement and
  the rating popup (truck #809 keeps its id for 12 s).

## Notes
- Varjo status: 0 invalid, 1 adjusting, 2 valid. Only 2 is kept.
- The lag search (default +/-5 frames) handles a cursor drawn a frame or two behind the CSV time.
- `px_per_deg` comes from the 3D gaze direction (`gaze_forward_*`) vs the mapped pixel position, and turns
  the 1.5 deg gaze circle into pixels.
- If the cursor is not detected, tune `CURSOR_HSV_LO/HI` in `hubris/calibrate.py`
  (`detect_cursor(img, debug=True)` returns the mask).
- `frame_number` in the Varjo CSV is the eye tracker's own counter, not the video frame. Time comes from
  `relative_to_video_first_frame_timestamp` (ns).
- The Varjo CSV writes NaN as `-nan(ind)` (handled in `hubris/gaze_io.py`).
- Left- and right-view projected gaze use different x coordinates; never average them. Calibration picks the
  eye whose view the video shows (video 1: left).
- The speed display is the "NN km/h" screen bottom-centre.

## Conventions
- Python; code in `hubris/` as modules plus small scripts at the top level. Outputs go in `out/<video>/`.
- Reference: the client's prototype ("V17 pipeline", slides 35-49 of DrivingSimulatorExperiments_Hubris_01102026.pptx)
  used YOLO11n for cars, SegFormer (Cityscapes) for road, Canny + Hough for lane lines, a heuristic for the speed
  display, and the priority ADAS -> CAR -> LANE -> EDGES -> ROAD. Its weaknesses: every vehicle was "car", lane
  lines were barely found, and about half of the samples ended up OTHER.
