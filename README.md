# HUBRIS gaze labelling pipeline

Labels every Varjo eye-tracker sample from a HUBRIS driving-simulator recording with what the driver was
looking at, as `category - class`: `object - truck`, `road - road line`, `interior - speed display`, and so on.

The eye tracker logs *where* the driver looks (200 Hz, in headset view coordinates). The recording is a
capture of the headset view with the simulator's own gaze dot burned in. The pipeline lines the two up and
works out *what* is under the gaze point in every frame.

Group project for the EVML minor at HAN, for HAN Automotive Research. The agreed scope is to make it work
properly on one complete recording (video 1). A second recording is then run without changing anything, as
a generalisation check.

## How it works

| Step | Script | What it does | Runs on | Video 1 |
|---|---|---|---|---|
| 0 | `detect_cursor_chunk.py` | find the teal gaze dot in every frame | CPU | ~10 min |
| 1-3 | `run_sync.py` | clean the gaze log, match samples to frames, fit the gaze-to-pixel map | CPU | ~1 min |
| 4 | `run_detect.py` | cars and trucks: YOLO11l (COCO weights) at 1280 px | GPU | ~15 min |
| 4c | `run_track.py` | ByteTrack ids, one class per vehicle, fill short gaps | CPU | ~30 s |
| 4b | `run_road.py` | road surface, lines, edge, guard rail: Mask2Former (Mapillary Vistas) | GPU | ~45 min |
| 4d | `run_speed.py` | speed display, found through the teal light bar on top of the screen | CPU | ~5 min |
| 5 | `run_label.py` | one label per gaze sample | CPU | ~1 min |
| 6 | `run_demo.py` | annotated demo video | CPU | ~2 min |

The step numbers follow the order we built things in. `run_all.py` runs them top to bottom as listed. The
logic for each step lives in `hubris/` (`calibrate.py`, `gaze_io.py`, `detect.py`, `track.py`, `road.py`,
`speed.py`, `label.py`, `demo.py`).

**Labelling rule.** Each valid sample gets a 1.5° circle (39 px in video 1) around its gaze point. The first
thing in this order that the circle touches gives the label:

speed display > car / truck > road line > road edge > guard rail > road surface > `other - other`

If two vehicles are in the circle, the one with the most overlap wins. Samples that can't be labelled get
`other - invalid` (blink, tracking lost, gaze outside the image) or `other - fog` (the foggy scene at the
start and end of every recording). The strict label from the gaze point alone is saved as well.

## Classes

| Category | Classes | Priority | Status |
|---|---|---|---|
| object | car, truck | Must | done |
| road | road line, road edge | Must | done |
| interior | speed display | Must | done |
| road | guard rail | Should | done |
| road | dotted vs solid line | Should | not yet |
| interior | mirror | Should | not yet |
| object part | car/truck indicator, wheel | Should | not yet |
| environment | bridge | Could | not yet |
| other | other, invalid, fog | | done |

Won't: traffic signs, traffic lights and the "Please rate your Perceived Risk" popup. Gaze on the popup is
labelled `other - other`.

## Setup

We work on Windows with VS Code.

1. Put this folder next to the recordings: `Downloads\hubris_pipeline`, with `varjo_capture1.mp4` and
   `varjo_gaze_output1.csv` in `Downloads`.
2. Create the environment. In VS Code: Terminal > Run Task... > **Set up Python environment**. Or by hand:
   ```
   py -3 -m venv .venv
   .venv\Scripts\python -m pip install -r requirements.txt
   ```
3. With an NVIDIA GPU, swap in CUDA torch (the pip default on Windows is CPU-only):
   ```
   .venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   ```

The YOLO and Mask2Former weights download automatically on the first run.

Without a GPU everything still runs, but steps 4 and 4b are much slower. We haven't timed a full CPU run.
`run_detect.py` uses the GPU when there is one. `run_road.py` defaults to `--device cuda`, so pass
`--device cpu` (see Known issues).

## Running

All steps on one recording:
```
python run_all.py --video ..\varjo_capture2.mp4 --gaze ..\varjo_gaze_output2.csv --out out\video2
```
- About 100 min on the RTX 4070 laptop. Most of that is the road segmentation.
- Steps whose output already exists are skipped. After a crash or sleep, run the same command again.
  `--redo labels demo` forces those steps to run again.
- Nothing is set per recording. The gaze calibration, the drive (start and end of the fog) and the demo clip
  are all found automatically.
- Keep the laptop plugged in. On battery, Windows throttles the GPU and CPU 2.5-4x: one minute of video took
  24 min in a test. `run_all.py` blocks idle sleep, but closing the lid still puts the laptop to sleep.

Single steps, with video 1 as the example:
```
python detect_cursor_chunk.py --video ..\varjo_capture1.mp4 --out out\video1\cursor_track.csv --start 0 --end 21069
python run_sync.py   --video ..\varjo_capture1.mp4 --gaze ..\varjo_gaze_output1.csv --out out\video1 --cursor-csv out\video1\cursor_track.csv
python run_detect.py --video ..\varjo_capture1.mp4 --sync-dir out\video1
python run_track.py  --sync-dir out\video1 --video ..\varjo_capture1.mp4
python run_road.py   --video ..\varjo_capture1.mp4 --sync-dir out\video1
python run_speed.py  --video ..\varjo_capture1.mp4 --sync-dir out\video1
python run_label.py  --sync-dir out\video1 --video ..\varjo_capture1.mp4
python run_demo.py   --sync-dir out\video1 --video ..\varjo_capture1.mp4   # add --whole-drive or --style full
```
Every script has `--help`. Detection and road segmentation resume where they stopped. The cursor script
appends to the CSV, so it can be run in chunks (`--start`/`--end`) if one long run gets cut off. In VS Code,
Run and Debug (Ctrl+Shift+D) has a configuration for each step, numbered 1-10.

## Output

Everything goes to `out/<video>/`, which is not in git. The main result is `gaze_labels.csv`, with one row
per Varjo sample:

| Column | Meaning |
|---|---|
| `raw_csv_row` | row in the original Varjo CSV |
| `video_time_s`, `display_frame` | time, and the video frame the sample is shown on |
| `x_px`, `y_px` | gaze point in video pixels |
| `label`, `category`, `class` | label from the circle, e.g. `object - car` |
| `track_id` | vehicle id, -1 if the label is not a vehicle |
| `overlap` | share of the circle on the labelled thing |
| `on_target` | the gaze point itself is on the labelled thing |
| `ambiguous` | the circle touches several things and the gaze point isn't on the chosen one, so the priority order decided |
| `touches` | everything the circle touched |
| `label_point` | strict label from the gaze point alone |
| `reason` | why a sample couldn't be labelled |

The other files, roughly in pipeline order:
- `cursor_track.csv`, `gaze_synced.csv`, `calibration.json`, `sync_report.json`: synced gaze, the fitted map
  and its error.
- `detections_yolo11l.*` and `detections_yolo11l_tracked.*`: vehicle boxes, before and after tracking.
- `seg_m2f/`: the raw Vistas class id per pixel for every frame, at 1024x768. `hubris.road.road_layers` turns
  these into the road masks, so the road definitions can change without rerunning the 45-minute model.
- `speed_display.csv`: the screen corners per frame.
- `drive.json`: first and last frame of the drive.
- `gaze_labels.json`: summary numbers (time per label, most-watched vehicles, settings used).
- `demo_<start>-<end>.mp4`: the demo video. H.264, so it plays in PowerPoint and browsers.
- `check/`: images to eyeball each step. On `check_frame_*.jpg` the magenta cross (our gaze) should sit on
  the teal dot. On `road_frame_*.jpg`: purple is road surface, red is road line, yellow is road edge, blue is
  guard rail.

## Results so far

| | Video 1 | Video 2 |
|---|---|---|
| Length, valid gaze | 702 s, 94.6% | 694 s, 94.2% |
| Gaze-to-pixel map: median error, lag | 4.2 px, 2 frames | 6.2 px, 3 frames |
| Drive (found automatically) | frames 2,090-20,460 | frames 2,350-20,255 |
| Frames with a vehicle | 93.7% | 85.8% |
| Speed display found (drive frames) | 100% | 99.1% |
| car / truck / other | 55.3 / 21.5 / 10.2% | 45.6 / 15.8 / 25.3% |
| speed display / road line | 1.1 / 10.3% | 8.0 / 1.9% |
| Ambiguous samples | 43.8% | 17.6% |
| Vehicles looked at | 89 | 181 |

Label shares are of the samples in the drive, using the circle. Video 2 was run with `run_all.py` and
nothing changed.

Circle vs gaze point, video 1 (117,087 drive samples):

| Label | Circle (1.5°) | Gaze point only |
|---|---|---|
| object - car | 55.3% | 27.4% |
| object - truck | 21.5% | 8.5% |
| road - road line | 10.3% | 10.1% |
| other - other | 10.2% | 17.0% |
| interior - speed display | 1.1% | 1.1% |
| road - guard rail / road edge / surface | 0.6 / 0.6 / 0.5% | 0.5 / 3.2 / 32.3% |

What we take from this so far:
- **The circle size matters a lot.** The video 1 driver mostly looks far ahead, where vehicles are small (median
  67 px tall against a circle 79 px across). For about half of the car labels the gaze point itself is just
  next to the car. The speed display is clear-cut: in 92% of its samples the gaze point is on it.
- The video 2 driver holds their head lower. The road sits in the top half of the image and the whole speed
  display is in view, and they look at it far more often (8% vs 1.1%).
- `other` is higher in video 2. In a random check of 8 such moments, 4 were the rating popup (correct), 2 were
  far-away or shaded vehicles that YOLO missed, 1 was the bridge and 1 was just above the speed display.
- Most-watched vehicles in video 1: truck #809 (63 s), car #88 (44 s), car #485 (41 s).
- `run_all.py` on a 60 s clip cut from video 1 gave the same label for 98.8% of samples as the full run.

These are consistency checks, not accuracy. Accuracy needs the hand-labelled test frames (to do).

## Things we found along the way

Gaze and sync
- `frame_number` in the Varjo CSV is the eye tracker's own counter, not the video frame. Time comes from
  `relative_to_video_first_frame_timestamp` (ns), and the frame is `floor(t * fps)`.
- Only status 2 (valid) is kept; 0 means invalid, 1 means adjusting. The CSV writes NaN as `-nan(ind)`.
- The left- and right-eye projected views use different x coordinates, so never average them. Calibration
  tries both and keeps the better fit (left, in both videos).
- The burned-in dot is drawn 2 frames (~67 ms) after the CSV timestamp in video 1, and 3 in video 2.
  `display_frame` already includes this.
- Affine fit in video 1: 6,355 of 8,465 pairs are inliers, RMSE 5.5 px. The outliers are mostly frames in the
  middle of a saccade, and blinks. A 2nd/3rd-order polynomial gained less than 0.1 px, so we kept the affine.
- The map is precise on the road (4.6 px in video 1, 10-14 px in video 2) but worse near the dashboard
  (~88 px at the bottom of video 1, mostly a horizontal offset). The speed display label still agrees with the
  simulator's own dot in 98% (video 1) and 94% (video 2) of frames.
- Pixels per degree come from the 3D gaze direction (`gaze_forward_*`) against the mapped pixel: 26.3 px/deg
  in video 1.
- If the dot isn't detected, tune `CURSOR_HSV_LO/HI` in `hubris/calibrate.py`. `detect_cursor(img, debug=True)`
  also returns the mask.

Fog
- Each recording starts and ends in thick fog with no road or vehicles visible, although the car is already
  doing 84-94 km/h. In video 1 the fog lasts until ~frame 2,090 (70 s) and returns at ~20,470. The drive is
  found from the segmentation: the road has to be visible in 90% of the next or previous 10 s.

Vehicles
- 2,672 of the 3,177 tracks in video 1 last one frame, but they hold only ~4% of the boxes (tiny far-away cars,
  false detections). Labelling only uses tracks of 3+ frames.
- Only boxes with conf ≥ 0.25 are saved, so ByteTrack's low-score stage has nothing to work with and it acts
  like SORT. Ids hold well: truck #809 keeps its id for 12 s through head movement and the popup.
- YOLO sometimes boxes the bottom edge of the windscreen as a wide, flat "car". Tracks centred on the car's
  own interior are dropped (10 in video 1, 5 in video 2).

Road
- We tried SegFormer-B2/B5 (Cityscapes) first, also with colour and texture clean-up. They spread "road" over
  the grass verges and the median. Mask2Former on Vistas stops at the verge and has separate lane-marking,
  guard-rail, terrain and ego-vehicle classes.
- The teal dot is inpainted before segmenting. Left in, it punched a hole with a fake road edge into the road,
  exactly where the driver was looking.
- Road patches with the car's interior just above them are dropped. When the driver looks down, the model
  sometimes calls the footwell "road".

Speed display
- The display moves with the head. It tilts up to ~23° when the driver looks down, and in normal driving it is
  half cut off at the bottom of the view, so a fixed region doesn't work.
- SIFT matching against a reference frame failed: the dark dashboard gives only 5-7 matching features.
- What works is anchoring on the pale-teal light bar along the top of the screen. It stays ~225 px long in
  every head position. The screen is a box below it: the bar plus 20 px on each side, 380 px deep.

## Known issues

- `run_road.py` defaults to `--device cuda`, and `run_all.py` doesn't pass a device. On a laptop without an
  NVIDIA GPU, run the road step by hand with `--device cpu`.
- A dry brown verge is sometimes taken for road, and the road edge then runs through the verge (e.g. video 1,
  frame 4962).
- The speed display is not found in the fog, where the light bar is washed out.
- Vistas calls the mirror "Bridge", so the mirror will need its own approach.

## To do

1. Hand-label 200-300 test frames from video 1, spread over the drive, and measure accuracy for both the
   circle and the gaze-point labels.
2. Fixations (I-VT or I-DT) on the 200 Hz stream, a label per fixation, dwell time, time to first fixation
   and switches between road and speed display.
3. Decide whether to report circle labels, gaze-point labels or both. A table over a few circle sizes would
   show how much it matters.
4. Should-haves: dotted vs solid lines (from the shape of the line mask) and the mirror (probably a fixed
   region relative to the cockpit).

## Decisions

- 2026-10-07: one complete recording instead of all ~70. The second recording is run untouched as a
  generalisation check. Test frames come from video 1 only. Labels have two levels, `category - class`.
- 2026-10-27: the fog is left out of the analysis, plain road is `road - surface`, a fixed priority instead of
  largest overlap, and guard rail is included.

## Background

The client's prototype ("V17 pipeline", slides 35-49 of `DrivingSimulatorExperiments_Hubris_01102026.pptx`)
used YOLO11n for cars, SegFormer (Cityscapes) for the road, Canny + Hough for lane lines, a heuristic for the
speed display, and the priority ADAS > CAR > LANE > EDGES > ROAD. Every vehicle came out as "car", lane lines
were barely found, and about half of the samples ended up as OTHER. This pipeline started from that.
