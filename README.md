# DriftLens

DriftLens V1 is a local computer vision tool for review assisted analysis of tandem drifting footage. Upload a private video, process a short interval and review one reliable lead and chase pair. Appearance based role continuity can propose later identities after camera changes or tracker fragmentation. Inspect the automatic matches and unknown spans before reading the image separation chart. The dashboard also opens with a prepared complete run from launch through the visible finish across four camera views.

The project lives separately from PitWall and the banking project. It uses a dedicated Python environment and stores its own inputs and outputs.

## First setup from a clone

1. Install Python 3.12 on Windows and make its Python launcher available.
2. Download or clone the source into its own folder, such as `G:\DriftLens`.
3. Double click `setup.cmd` and wait for the dedicated environment, free dependencies and pretrained model download to finish.
4. Double click `launch.cmd` and open [the local dashboard](http://127.0.0.1:8510).
5. Open Process a clip and upload your own permitted video. Saved example replays and source footage are private and are absent from a fresh clone.

For an empty destination folder, PowerShell setup from the published repository is:

```powershell
git clone https://github.com/m11ahmed/DriftLens.git G:\DriftLensClone
cd G:\DriftLensClone
.\setup.cmd
.\launch.cmd
```

Normal analysis runs locally after setup. The numerical reports describe the saved local experiments; they do not bundle the visual evidence or make every historical replay available in a clean checkout. [The local demo guide](docs/DEMO_GUIDE.md) contains manual PowerShell launch commands.

## Open the app

1. Open the DriftLens folder on the G drive.
2. Double click `launch.cmd`.
3. Keep the terminal open while using the app.
4. Open [the local dashboard](http://127.0.0.1:8510) if the browser does not open automatically.
5. Watch the Complete run tab and use the camera shot selector to jump between views.
6. Inspect the role review intervals, missing observations and image separation chart.
7. Export the complete replay, timeline CSV, camera shot metadata or summary.

The downloaded source is `longbeach2024_action.mp4`. The prepared run covers source time 31:44.5 through 32:11.2, lasting 26.7 seconds. Its replay contains 268 sampled frames at ten frames per second and lasts 26.8 seconds because individual camera shots end between sampled frames. The timeline preserves both source time and replay time. Player jumps use whole seconds and may include less than one second before the selected shot.

The Shot review tab retains the twelve original diagnostic clips and your own results. Your result includes a replay, a readable report and numerical exports. Confirm lead and chase roles to enable tandem separation measurements.

## Test your own video

1. Open Process a clip and choose Upload my video.
2. Select an MP4, MOV, MKV, AVI or WebM file of at most 200 MB.
3. Begin with ten to fifteen seconds of one continuous camera view showing both participants. Set Start seconds and End seconds within the video. Each interval must be longer than zero and no longer than 60 seconds.
4. Keep the selected analysis profile for the first comparison and enter a new Result name such as `my_first_test`.
5. Choose Process clip locally and keep the terminal open until processing finishes.
6. Open the result in Shot review. Check boxes, local tracker IDs, missing observations and the readable report against the visible cars.
7. Watch a clear initial interval and confirm which observed ID is lead and which is chase using livery and travel order.
8. Open Follow lead and chase across views. Set Seed start seconds, Seed end seconds, Seed lead ID and Seed chase ID. Confirm I have checked lead and chase in the seed interval, then choose Follow this pair through the clip. No prior manual interval save is required.
9. Inspect the resulting replay, image separation chart and downloadable report. Later automatic assignments are marked MATCH and remain unverified. Correct unsuitable matches in Correct roles within bounded clip intervals or seed another reliable interval when needed. Manual changes override the assignments, retain the diagnostics and deactivate continuity status.

Uploads stay under `data/uploads` and results under `outputs/runs`. The analysis uses your CPU. A recorded 28 second test took about ten minutes before replay export and review, so process first and record the finished dashboard afterward. The sampling rate describes selected video frames and is different from processing speed.

[Local testing and recording guide](docs/DEMO_GUIDE.md) gives launch instructions, a short demo outline and LinkedIn wording. Use your own recording or footage you have permission to publish for a public demo. The supplied broadcast and its visual derivatives remain private while permission is unverified.

## What it does

1. Combines contiguous camera shots into a complete local run review.
2. Detects cars with a small pretrained YOLO model.
3. Tracks identities with ByteTrack or BoTSORT.
4. Records detections and every sampled frame in CSV files.
5. Exports an annotated video for run review.
6. Computes a screen separation proxy using reviewed role assignments.
7. Leaves measurements empty when either assigned car is missing.
8. Compares saved results with independent visual reference boxes for detection precision, recall, visible coverage, and sparse identity checks.
9. Explains accepted measurements and unknown intervals in a readable run report.
10. Provides timestamped review events and exact annotated sample images.
11. Records deliberate human frame reviews separately from frozen assistant reference labels.

The [Ultralytics tracking documentation](https://docs.ultralytics.com/modes/track/) describes the tracking backend. OpenCV handles video processing, and Streamlit provides the local review interface.

## Reading the measurements

Screen separation is the distance between bounding box centres divided by their mean width. Camera position, zoom, perspective, and changing box sizes affect the value. It describes the image rather than physical distance.

A missing observation or unknown role creates a gap in the chart. Tracker IDs remain local observations. The prepared historical examples use roles visually reviewed by the AI assistant without human expert validation. The newer continuity feature starts from a reviewer supplied pair and compares observed car appearance after cuts or local ID changes. Later proposed labels display MATCH; match scores and margins describe heuristic evidence rather than identity probability. A match must pass similarity and competing match checks; uncertain cases remain unknown. These automatic proposals can be wrong when appearance changes, liveries are similar, cars overlap or the view is unfamiliar. Role changes and camera cuts still break chart lines, and no position is carried across hidden frames.

Unassigned candidates can include background vehicles and false detections. Tandem separation uses only the reviewed participant roles with accepted current observations. A visible candidate label is different from a confirmed participating car.

Uploaded clips use a bounded role editor beside the replay. Review the initial identities and inspect continuity proposals beside their appearance evidence. The proposals assign roles to current observed IDs; they do not change detector boxes, retrain YOLO or prove physical identity. Saved corrections rebuild the replay, metrics, readable report and analysis JSON together. Rejected matches and missing observations stay unknown. The Complete run editor also supports a whole camera shot correction.

This version does not estimate metres, vehicle speed, drift angle, judging scores, or driver skill. One detector candidate was fine tuned for twelve epochs. Its diagnostic test results regressed, so the original weights were preserved. The Evidence tab shows that experiment separately from the earlier tracker comparison. Upload support does not establish accuracy on every event or personal video.

## Evidence and reproducibility

Twelve diagnostic camera shots are processed. Six tuning clips and six test clips come from separate battles within the same broadcast. Six test shots also have BoTSORT comparison exports. The complete run reuses material from that broadcast and is a demonstration, not an additional independent test.

The original comparison uses 416 pixel inference. The complete replay uses BoTSORT with 640 pixel inference and suppression of overlapping duplicate detections across car and truck classes. Original comparison scores do not establish accuracy for this complete replay.

On 24 sparse test frames containing 48 reference car boxes, the raw detector precision and recall were both 85.4 percent. ByteTrack visible reference coverage was 83.3 percent with four sampled identity changes. BoTSORT coverage was 75.0 percent with zero changes across fewer comparable samples. These are small diagnostic results, using approximate boxes drafted visually by the AI assistant without human expert validation. They do not establish general accuracy or continuous identity correctness.

The default enhanced replay is under `outputs/full_runs/full_run02`, including `annotated.mp4`, `timeline.csv`, `shots.json`, `summary.json`, `analysis.json` and `report.txt`. The original replay remains under `outputs/full_runs/full_run01`. Each child shot retains current detections, confirmed observations and sampled frame metadata. `data/full_run_enhanced_catalog.json` records the enhanced settings and source reviewed role decisions. `outputs/visibility_report.json` contains the matched comparison.

On this reused demonstration, reviewed pair availability increased from 44.8 percent to 71.3 percent, or 191 of 268 sampled frames. Both cars now have accepted current observations in all eighteen final samples of the aerial shot, compared with zero previously. Orientation passes, tracker settings and conservative recent appearance recovery produced this gain with the original weights. Experimental recovered boxes display R and preserve their original predicted class and score. No hidden position is reconstructed. Remaining visible misses and merged boxes are excluded from pair measurements. This result is not an independent accuracy benchmark or proof of reliability on other footage.

A separate 28 second Thorne versus Olsen excerpt from the same event was uploaded through the app. The original result provided 173 accepted pairs in 280 samples, or 61.8 percent, after assistant visual role and camera review without human expert validation. Processing took 611.995 seconds. Its automatic camera safeguard missed three cuts, and background false detections and fragmented identities remained. The excerpt was outside the existing selected windows when first tested; once used to guide fixes it became a regression example. Its reviewed pair availability is different from detector accuracy and is not evidence of generalisation to other events or your own video. The original output is preserved.

The final camera safeguard was scanned on both complete source intervals. It found all six visually reviewed boundaries with no other flags in these two known examples. The original safeguard found none of the three cuts in the uploaded battle and one of three in the earlier complete interval. The saved reviewed replays and pair totals remain historical results; these direct camera scans do not replace them or establish general camera accuracy. A conservative recovery check also rejected the inspected curb and bin artifacts among saved recovered candidates while retaining the inspected car recoveries. It may withhold optional recovery of monochrome vehicles. Native detector false candidates still require visual review.

Saved numerical camera checks are under `outputs/v1_camera_cut_regression` and crop screening evidence is `outputs/recovery_quality_evidence.json`. They preserve the tested settings and counts. The final runtime, browser, upload, provenance, documentation and test checks are recorded in `outputs/v1_closeout_verification.json`.

[Evaluation notes](docs/EVALUATION.md) explain the annotation format, commands, matching rules, and limitations. Evaluation figures belong to their specific labelled clips and settings. Sparse labels are not a complete tracking benchmark.

[Measured results](docs/RESULTS.md) explain the actual tracker tradeoff, hardware speed, and failure examples. The Evidence tab shows the saved report and exports it as JSON. Broader validation needs human reviewed labels and footage from other events.

## Recreate the environment

Run `setup.cmd` to create the dedicated Python environment and fetch the free CPU runtime and model. The supplied environment is already installed. The setup needs internet access; review and processing then work locally with the downloaded inputs. `requirements.txt` pins direct dependencies and `requirements.lock.txt` records the installed environment.

## Footage and licensing

The local source is the official [Long Beach 2024 Top 16 ALL ACTION video](https://www.youtube.com/watch?v=nobounLesY4). Record its URL and chosen timestamps with the clip metadata. The downloaded copy is an input for local analysis. Download access does not establish permission to redistribute the footage.

Project source uses AGPL3. Ultralytics provides an [open source licensing option](https://www.ultralytics.com/license). The code license does not grant rights to Formula Drift video. Source footage, generated video, model weights, and the Python environment should remain outside the public source repository unless their distribution is authorised.

## Tests

The test command is in the evaluation notes. Checks cover detection matching, identity comparisons, missing observations, complete source coverage, camera boundaries, role intervals, replay timestamp mapping and export recovery. Browser verification checks actual video playback, shot navigation, charts and downloaded exports.

The earlier V1 verification passed 122 tests and 80 subtests, plus an actual four second browser upload that detected its camera change and played the complete replay. Both preserved examples and report exports passed that check. Those counts predate the role continuity feature. Continuity requires a reviewed seed and visual checks on your own clip; it has no independent event benchmark.
