# DriftLens

DriftLens is a computer vision portfolio project for reviewing Formula Drift tandem runs in real footage. The dashboard opens with one complete run from launch through the visible finish, combining four camera shots into an annotated replay and a measurement timeline.

The project lives separately from PitWall and the banking project. It uses a dedicated Python environment and stores its own inputs and outputs.

## Open the app

1. Open the DriftLens folder on the G drive.
2. Double click `launch.cmd`.
3. Keep the terminal open while using the app.
4. Open [the local dashboard](http://127.0.0.1:8510) if the browser does not open automatically.
5. Watch the Complete run tab and use the camera shot selector to jump between views.
6. Inspect the role review intervals, missing observations and image separation chart.
7. Export the complete replay, timeline CSV, camera shot metadata or summary.

The downloaded source is `longbeach2024_action.mp4`. The prepared run covers source time 31:44.5 through 32:11.2, lasting 26.7 seconds. Its replay contains 268 sampled frames at ten frames per second and lasts 26.8 seconds because individual camera shots end between sampled frames. The timeline preserves both source time and replay time. Player jumps use whole seconds and may include less than one second before the selected shot.

The Shot review tab retains the twelve original diagnostic clips. Process a clip accepts another continuous camera shot. Processing uses the CPU and may run slower than playback.

## What it does

1. Combines contiguous camera shots into a complete local run review.
2. Detects cars with a small pretrained YOLO model.
3. Tracks identities with ByteTrack or BoTSORT.
4. Records detections and every sampled frame in CSV files.
5. Exports an annotated video for run review.
6. Computes a screen separation proxy using reviewed role assignments.
7. Leaves measurements empty when either assigned car is missing.
8. Compares saved results with independent visual reference boxes for detection precision, recall, visible coverage, and sparse identity checks.

The [Ultralytics tracking documentation](https://docs.ultralytics.com/modes/track/) describes the tracking backend. OpenCV handles video processing, and Streamlit provides the local review interface.

## Reading the measurements

Screen separation is the distance between bounding box centres divided by their mean width. Camera position, zoom, perspective, and changing box sizes affect the value. It describes the image rather than physical distance.

A missing observation or unknown role creates a gap in the chart. Tracker IDs are local to each camera shot. Lead and chase roles were visually reviewed by the AI assistant against source car livery and travel order, without human expert validation. Reviewed time intervals exclude ambiguous overlaps, parked vehicles and identity swaps. Role changes and camera cuts break chart lines. The project does not claim automatic identity association across cameras.

The correction form can replace a shot's interval map with two selected IDs for the whole shot and rebuild the complete exports. Inspect the entire shot before accepting that override.

This version does not estimate metres, vehicle speed, drift angle, judging scores, or driver skill. It has no custom trained drift detector. The pretrained model is a baseline whose failures form part of the portfolio evidence.

## Evidence and reproducibility

Twelve diagnostic camera shots are processed. Six tuning clips and six test clips come from separate battles within the same broadcast. Six test shots also have BoTSORT comparison exports. The complete run reuses material from that broadcast and is a demonstration, not an additional independent test.

The original comparison uses 416 pixel inference. The complete replay uses BoTSORT with 640 pixel inference and suppression of overlapping duplicate detections across car and truck classes. Original comparison scores do not establish accuracy for this complete replay.

On 24 sparse test frames containing 48 reference car boxes, the raw detector precision and recall were both 85.4 percent. ByteTrack visible reference coverage was 83.3 percent with four sampled identity changes. BoTSORT coverage was 75.0 percent with zero changes across fewer comparable samples. These are small diagnostic results, using approximate boxes drafted visually by the AI assistant without human expert validation. They do not establish general accuracy or continuous identity correctness.

Complete outputs are under `outputs/full_runs/full_run01`, including `annotated.mp4`, `timeline.csv`, `shots.json` and `summary.json`. Each child shot retains raw detections, confirmed observations and sampled frame metadata. `data/full_run_catalog.json` records source boundaries and explicit role review decisions. `outputs/full_run_report.json` is the numerical summary.

[Evaluation notes](docs/EVALUATION.md) explain the annotation format, commands, matching rules, and limitations. Evaluation figures belong to their specific labelled clips and settings. Sparse labels are not a complete tracking benchmark.

[Measured results](docs/RESULTS.md) explain the actual tracker tradeoff, hardware speed, and failure examples. The Evidence tab shows the saved report and exports it as JSON. Broader validation needs human reviewed labels and footage from other events.

## Recreate the environment

Run `setup.cmd` to create the dedicated Python environment and fetch the free CPU runtime and model. The supplied environment is already installed. The setup needs internet access; review and processing then work locally with the downloaded inputs. `requirements.txt` pins direct dependencies and `requirements.lock.txt` records the installed environment.

## Footage and licensing

The local source is the official [Long Beach 2024 Top 16 ALL ACTION video](https://www.youtube.com/watch?v=nobounLesY4). Record its URL and chosen timestamps with the clip metadata. The downloaded copy is an input for local analysis. Download access does not establish permission to redistribute the footage.

Project source uses AGPL3. Ultralytics provides an [open source licensing option](https://www.ultralytics.com/license). The code license does not grant rights to Formula Drift video. Source footage, generated video, model weights, and the Python environment should remain outside the public source repository unless their distribution is authorised.

## Tests

The test command is in the evaluation notes. Checks cover detection matching, identity comparisons, missing observations, complete source coverage, camera boundaries, role intervals, replay timestamp mapping and export recovery. Browser verification checks actual video playback, shot navigation, charts and downloaded exports.
