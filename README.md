# DriftLens

DriftLens is a computer vision portfolio project for reviewing Formula Drift tandem runs in real footage. It detects cars, follows their identities, and lets a reviewer assign lead and chase roles before inspecting an annotated replay.

The project lives separately from PitWall and the banking project. It uses a dedicated Python environment and stores its own inputs and outputs.

## Open the app

1. Open the DriftLens folder on the G drive.
2. Double click `launch.cmd`.
3. Keep the terminal open while using the app.
4. Open [the local dashboard](http://127.0.0.1:8510) if the browser does not open automatically.
5. Select a completed run, inspect the car identities, and assign the lead and chase IDs when both are reliable.

The downloaded source file is `longbeach2024_action.mp4`. Review short intervals containing one camera shot. Processing uses the CPU and may run slower than playback. The first analysis also requires the detector weights to be available locally.

## What it does

1. Processes short intervals from a local video.
2. Detects cars with a small pretrained YOLO model.
3. Tracks identities with ByteTrack or BoTSORT.
4. Records detections and every sampled frame in CSV files.
5. Exports an annotated video for run review.
6. Computes a screen separation proxy after manual role assignment.
7. Leaves measurements empty when either assigned car is missing.
8. Compares saved results with independent visual reference boxes for detection precision, recall, visible coverage, and sparse identity checks.

The [Ultralytics tracking documentation](https://docs.ultralytics.com/modes/track/) describes the tracking backend. OpenCV handles video processing, and Streamlit provides the local review interface.

## Reading the measurements

Screen separation is the distance between bounding box centres divided by their mean width. Camera position, zoom, perspective, and changing box sizes affect the value. It describes the image rather than physical distance.

A missing car creates a gap in the measurement chart. Tracker identity numbers do not establish driver identity. Smoke, overlap, replay edits, and camera cuts can interrupt tracking. Inspect the replay before accepting assigned roles.

This version does not estimate metres, vehicle speed, drift angle, judging scores, or driver skill. It has no custom trained drift detector. The pretrained model is a baseline whose failures form part of the portfolio evidence.

## Evidence and reproducibility

Twelve real camera shots are already processed. Six tuning clips and six test clips come from separate battles within the same broadcast. Six test shots also have BoTSORT comparison results. The default demo has reviewed lead and chase labels, with replay and separation measurements ready to inspect.

On 24 sparse test frames containing 48 reference car boxes, the raw detector precision and recall were both 85.4 percent. ByteTrack visible reference coverage was 83.3 percent with four sampled identity changes. BoTSORT coverage was 75.0 percent with zero changes across fewer comparable samples. These are small diagnostic results, using approximate boxes drafted visually by the AI assistant without human expert validation. They do not establish general accuracy or continuous identity correctness.

Run outputs include `detections.csv`, `observations.csv`, `frames.csv`, `frame_metrics.csv`, and `summary.json`. Raw detections are recorded separately from confirmed tracker observations. The summary records the source interval and processing settings. Annotated exports support visual inspection alongside the original footage.

[Evaluation notes](docs/EVALUATION.md) explain the annotation format, commands, matching rules, and limitations. Evaluation figures belong to their specific labelled clips and settings. Sparse labels are not a complete tracking benchmark.

[Measured results](docs/RESULTS.md) explain the actual tracker tradeoff, hardware speed, and failure examples. The Evidence tab shows the saved report and exports it as JSON. Broader validation needs human reviewed labels and footage from other events.

## Recreate the environment

Run `setup.cmd` to create the dedicated Python environment and fetch the free CPU runtime and model. The supplied environment is already installed. The setup needs internet access; review and processing then work locally with the downloaded inputs. `requirements.txt` pins direct dependencies and `requirements.lock.txt` records the installed environment.

## Footage and licensing

The local source is the official [Long Beach 2024 Top 16 ALL ACTION video](https://www.youtube.com/watch?v=nobounLesY4). Record its URL and chosen timestamps with the clip metadata. The downloaded copy is an input for local analysis. Download access does not establish permission to redistribute the footage.

Project source uses AGPL3. Ultralytics provides an [open source licensing option](https://www.ultralytics.com/license). The code license does not grant rights to Formula Drift video. Source footage, generated video, model weights, and the Python environment should remain outside the public source repository unless their distribution is authorised.

## Tests

The test command is in the evaluation notes. These checks cover detection matching, identity comparisons, missing observations, camera cuts, and unknown separation when a car is absent.
