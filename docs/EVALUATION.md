# Evaluation protocol

DriftLens reports evidence about specific detectors and trackers on specific local video intervals. The original pretrained baseline remains active after a completed detector adaptation experiment failed its promotion gate. The project does not currently establish accuracy across Formula Drift events. The experiments use excerpts from one broadcast, so their shared venue and camera style limit generalisation.

## Current reference set

The saved `data/annotations/labels.json` contains 24 test frames with 48 car boxes and four tuning frames with eight boxes. These approximate visible extents were drafted by the AI assistant from source imagery independently of detector predictions. No human expert has validated them. Current scores are a small diagnostic comparison against those references. The reference car identities and initial role suggestions carry the same limitation.

## Reference annotation protocol

Annotate source frames in their original pixel dimensions. Do not use the dimensions of a resized dashboard thumbnail. Bounding boxes use `[x1, y1, x2, y2]` with the top left and bottom right corners. Draw a tight box around the visible vehicle. Use the same box convention throughout the labelled set.

Give each physical car a stable reference label such as `car_a` and `car_b` within a clip. The reference identity is separate from a tracker ID and from an assigned lead or chase role. Include every visible target car in a labelled frame. Omitting a visible car makes its correct prediction look like a false positive.

Use `visible` for clear cars and `partial` when smoke or overlap leaves enough of the vehicle visible to draw a defensible box. A completely hidden car can be omitted or recorded as `hidden` without a box; it is excluded from the detection denominator. Do not invent its location. Such a sample breaks the identity comparison chain for that car.

The following is an illustration of the file format, not a result or an actual annotation:

```json
{
  "coordinate_space": "source_pixels",
  "samples": [
    {
      "clip_id": "run01_clear",
      "time_seconds": 0.4,
      "frame_index": 12,
      "shot_index": 0,
      "cars": [
        {"identity": "car_a", "bbox": [100, 200, 220, 280], "visibility": "visible"},
        {"identity": "car_b", "bbox": [280, 210, 390, 285], "visibility": "partial"}
      ]
    }
  ]
}
```

`time_seconds` is relative to the start of the selected interval, not the full broadcast. `clip_id` must match the processed CSV. `frame_index` is optional, but copying it from `frames.csv` makes the selected frame explicit. `shot_index` is optional; when present it must agree with the processed frame metadata. Use distinct sampled frames and avoid duplicate timestamps.

Labels should be made from the original frame before inspecting model predictions when possible. Record who labelled the sample, when it was labelled, and any ambiguity separately. A second reviewer should check difficult smoke and overlap examples before a broader accuracy claim.

## Alignment and detection metrics

The evaluator reads `observations.csv` and automatically uses `frames.csv` beside it. The frame manifest is essential because it records sampled frames with no detections. Without that manifest, time-only annotations outside observed frames are counted as unaligned and excluded. Check the unaligned count before interpreting scores.

A label aligns to the nearest processed frame within 0.1 seconds by default. An explicit `frame_index` also has to meet the time tolerance. Two annotations cannot evaluate the same processed frame twice. The time error for every aligned sample appears in the report.

Predictions flagged `observed=false` are excluded. An observation must have a finite box with positive width and height. All boxes are in source pixels. Matching sorts candidate overlaps by descending intersection over union and greedily accepts pairs with IoU at least 0.5. Each prediction and each human box can participate in at most one match. This is a simple one-to-one greedy matcher, not a globally optimal assignment algorithm.

1. A matched box is a true positive.
2. An unmatched observed prediction is a false positive.
3. An unmatched visible reference box is a false negative.
4. Precision is true positives divided by all observed predictions.
5. Recall is true positives divided by all visible reference boxes.
6. F1 combines precision and recall through their count formula.

An undefined denominator is reported as `null`, rather than an invented perfect or zero score. Partial cars are included in these counts. Hidden cars are excluded. Counts and the number of aligned samples must accompany rates.

## Sparse tracking checks

Visible ground truth coverage is the fraction of reference boxes that match an observed prediction with a nonempty tracker ID. It measures presence of a tracked observation at labelled times, not uninterrupted tracking between them.

An identity switch is counted when the same reference car has a different matched tracker ID in successive comparable labelled frames within the same clip and camera shot. The report includes the number of actual comparable identity transitions as well as the switch count. A detection miss, hidden label, missing tracker ID, unaligned sample, or camera cut breaks the comparison chain. IDs on either side of an unknown interval are not asserted to be continuous.

Sparse labels cannot establish full MOT metrics, complete switch counts, or identity quality inside unlabelled intervals. In particular, these reports do not claim HOTA, IDF1, MOTA, or a full per-frame benchmark. Review the annotated replay to find failures between labelled samples.

## Screen separation and role review

Assign lead and chase roles manually from a reliable frame. Their IDs must differ and both must occur in the run. For each processed frame where both assigned IDs are actually observed, separation is the Euclidean distance between box centres divided by the mean box width. Missing detections leave an unknown value. The pipeline does not interpolate the position of a hidden car or fabricate separation during smoke.

The value is a perspective-dependent image proxy. It cannot be converted to metres, speed, drift angle, judging proximity, or driver skill without a separate calibrated method and validation. A tracker swap can corrupt the meaning of assigned roles even if both IDs remain present, so inspect role assignments during playback.

## Analyze an uploaded local clip

Open **Process a clip → Analyze your own clip**, choose **Upload my video**, and select an MP4, MOV, MKV, AVI or WebM file of at most 200 MB. Upload intake verifies readable video metadata and stores the file by its content hash under `data/uploads`. Choose a source interval longer than zero and no longer than 60 seconds, an analysis profile and a new result name, then select **Process clip locally**. CPU processing can be slower than playback. Start with a short continuous view and consult the measured processing speed.

Open the saved result in **Shot review** to inspect its replay, readable review, review events, sample images and exports. The report initially describes observations and missing role evidence. Use **Correct roles within bounded clip intervals** to supply visual lead and chase decisions. Intervals use clip relative seconds with included starts and excluded ends, must remain ordered and nonoverlapping, and cannot carry assigned identities across a detected camera cut. Each assigned ID must occur inside its interval. Empty IDs and uncovered time remain unknown. **Mark all clip roles unknown** clears the accepted role mapping without inventing hidden locations.

Saving a correction records an unverified user assignment and rebuilds the replay, frame metrics, analysis JSON and readable report together, with rollback on publication failure. This role review is separate from annotation review and does not make model detections expert verified. A successful upload or a readable report establishes neither general accuracy on arbitrary footage nor valid physical telemetry. Evaluation of a new clip needs its own source labels, held out groups and reviewed identities.

## Run the evaluator

From the project folder:

Reproduce all inspected clips, the six tracker comparisons, and the combined diagnostic report:

```powershell
.venv\Scripts\python.exe tools\run_catalog.py --comparisons
.venv\Scripts\python.exe tools\build_evidence.py
```

Use `--force` on the catalogue command to replace existing results after a processing change. To recreate reviewed example roles, run `tools\prepare_demo_roles.py`; those presets are valid only for the pinned model, settings, tracker adapter and current source. Ambiguous identity swaps remain unassigned.

The commands below illustrate evaluating a separate single clip label file:

```powershell
.venv\Scripts\python.exe -m driftlens.evaluation data\annotations\run01.json outputs\run01\observations.csv outputs\run01\evaluation.json
```

To specify a frame manifest or a different matching tolerance:

```powershell
.venv\Scripts\python.exe -m driftlens.evaluation data\annotations\run01.json outputs\run01\observations.csv outputs\run01\evaluation.json --frames outputs\run01\frames.csv --min-iou 0.5 --max-time-delta 0.1
```

To process an interval with either supported tracker:

```powershell
.venv\Scripts\python.exe -m driftlens longbeach2024_action.mp4 outputs\example_bytetrack --start 30 --end 38 --tracker bytetrack --size 416 --fps 10
.venv\Scripts\python.exe -m driftlens longbeach2024_action.mp4 outputs\example_botsort --start 30 --end 38 --tracker botsort --size 416 --fps 10
```

The timestamps above are command examples only. Choose visually checked single-shot intervals. The interval, model weights, detector thresholds, image size, sampled rate, package versions, source dimensions, and measured processing speed must be recorded with each real run. Compare trackers on the same source frames with identical detector settings.

To run the logic checks:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Original complete run demonstration

`data/full_run_catalog.json` covers source seconds 1904.5 through 1931.2 with four contiguous camera shots and 801 source frames. The independently sampled replay has 268 frames at 10 fps and lasts 26.8 seconds. `run_time` preserves source elapsed time; `playback_time` includes the fractional camera boundary rounding in the exported replay. Native player jumps use whole seconds with less than one second of pre-roll.

Recreate the full run with the independent environment:

```powershell
.venv\Scripts\python.exe -m driftlens.full_run data\full_run_catalog.json --tracker botsort --size 640
```

Existing complete child results are cached only when source path, source file size and modification time, model checksum, source range, pipeline revision, tracker, sampled rate, inference size and suppression setting match. Add `--force` for fresh inference. A freshly processed source requires visual confirmation of the saved role intervals because tracker IDs are not intrinsic car identities. Preset review intervals are withheld when the source or model checksum differs from the reviewed manifest. To reassemble already reviewed child results without inference, use `--assemble-only`.

The complete run uses class agnostic suppression to remove overlapping car and truck predictions on the same vehicle. The original diagnostic comparison remains 416 pixels with its original suppression setting. Original scores are not accuracy estimates for this different demonstration profile. Complete run material overlaps the existing test broadcast and is not additional held out evidence.

Role intervals are half open seconds relative to each camera shot. An AI assistant inspected the exported observations against source livery and travel order, marking swaps, parked candidates, merged boxes and ambiguous spans. Unassigned intervals remain unknown even when a detector box exists. Measurements require two accepted current observations. Charts break at camera cuts, local identity mapping changes and missing samples. No automatic association across cameras is claimed, and no human expert validation has occurred.

In **Complete run → Review roles for a camera shot**, **Role edit scope** defaults to **Bounded intervals**. Edit ordered, nonoverlapping start and end times within the selected shot, observed lead/chase IDs and the review basis. One role can remain known while the other is empty. Uncovered time stays unknown; **Mark the entire camera shot as role unknown** clears the map. **Save reviewed intervals and rebuild run** rebuilds shot and complete exports with the written conclusions, restoring prior outputs on publication failure.

**Entire camera shot** remains available for a pair that has been checked throughout a view. It replaces that shot's interval map with the chosen whole shot IDs. Both edit modes record an unverified user assignment, preserve missing observations and do not associate identities across camera views.

Complete outputs are `annotated.mp4`, `timeline.csv`, `shots.json` and `summary.json` under `outputs/full_runs/full_run01`. Each child shot retains raw detections, observations and source frame indices. `outputs/full_run_report.json` records the numeric summary. Browser checks verify the actual 26.8 second video, jumps, chart rendering and downloaded exports.

## Broader evaluation

Reserve entire tandem runs for evaluation. Consecutive or near-identical frames from one run must not appear on both sides of a training or tuning split. If no detector training has occurred, still separate the clips used to choose thresholds from the final comparison clips. Add a second event or camera style before claiming robustness across venues.

Label clear shots, partial smoke, and overlapping cars separately. Report their sample counts and failure cases, not just an aggregate rate. A detector-only baseline can show whether the main failures come from the detector or from association. Compare ByteTrack and BoTSORT under the same sampling and detector settings and preserve unsuccessful examples.

Wall-clock throughput includes detection and tracking; export time may be recorded separately. A sampled analysis rate does not imply real-time throughput. The review must disclose CPU hardware, sample rate, and the number of frames processed.

## Completed adaptation experiment

The completed twelve epoch CPU experiment trained on 24 frames with 59 approximate boxes from two tuning battle groups and validated on eight frames with 20 boxes from a third tuning group. It loaded all 355 pretrained tensors from a fresh unfused checkpoint. The best validation checkpoint was selected before testing once against the frozen 24 frame, 48 box reference set. Original weights and frozen labels were preserved. No new training is needed to inspect this experiment.

`outputs/finetuning_report.json` records a 640 pixel detector comparison with confidence 0.15, suppression IoU 0.5, class agnostic suppression, COCO car and truck classes and reference matching IoU 0.5. It differs from `outputs/evaluation_report.json`, which retains the original 416 pixel sparse detector and tracker evaluation. Do not substitute the newer detector scores for the old tracking results or claim that they measure full run pair coverage.

Baseline and candidate validation F1 were 73.7% and 77.8%. On frozen test references, baseline precision/recall/F1 were 97.8%/91.7%/94.6%, compared with 95.6%/89.6%/92.5% for the candidate. The promotion gate failed; the pretrained detector stays active. This negative finding uses approximate assistant labels without human expert review. Repeated tuning after examining this test result would require a new independent holdout before an unbiased final comparison.

## Separate human annotation review workflow

The frozen `data/annotations/labels.json` remains the historical reference. Human corrections belong in `outputs/annotation_review/queue.json` and separate exports. The prepared queue starts with all 28 samples pending; no box or identity is declared human reviewed until a person explicitly confirms it. Source frames remain private local assets.

In the app, open **Evidence → Review reference labels for future training**. Inspect the original frame beside current reference boxes, edit the source pixel coordinate table and choose visibility, class, smoke and overlap tags. Enter the actual reviewer's name and explicitly confirm image inspection, visible box checks and shot local identity checks, then choose **Save my frame review**. The frame dimensions must match the queue's source dimensions before a review can be saved. **Export the review queue** preserves pending drafts and confirmed records separately; **Export confirmed labels** contains only saved human self attestations. The delivered queue remains at zero human confirmations. New uploaded clip samples are not automatically added to this frozen reference review queue.

The queue schema stores each frame's clip, camera shot, source pixel dimensions, relative image path, source video ID, battle group, split, original assistant cars, editable cars, smoke and overlap tags, and provenance. Identities such as `car_a` belong to `clip_id:shotN`, separate from numeric tracker IDs. Supported classes are COCO 2 for car and 7 for truck. Boxes must have positive area inside the source frame. `hidden` cars require a null box. Smoke tags are clear, light, heavy or unknown; overlap tags are none, partial, severe or unknown. These tags are review choices, never inferred as measured smoke severity from a shot caption.

A confirmed sample records a reviewer name, ISO timestamp with timezone, three literal true confirmations for image inspection, box/class/visibility checks and shot local identity checks, and a fingerprint of the reviewed content. Changing content invalidates that confirmation. This is a human self attestation; it does not establish reviewer expertise or independent expert certification. The original annotation method and original cars remain attached to the queue. Its original reference fingerprint uses canonical JSON content, rather than byte formatting; historical raw file hashes remain in the original evidence manifests.

`battle` grouping keeps every shot from the same video and battle group in one split. `video` grouping keeps all frames of one source video in one split. Imports reject changed source provenance, frame context, split policy or unknown frames. The existing same broadcast dataset uses battle grouping; video grouping rejects its mixed tuning and test split. Neither policy makes related frames independent, and a second event is still needed to test broader generalisation.

Inspect the already prepared queue:

```powershell
.venv\Scripts\python.exe tools\annotation_review.py status
```

If the queue is absent, create it once with `tools\annotation_review.py queue`. Creation refuses to replace an existing queue. It does not run inference or training. For a separate new reference set, pass `--labels data/annotations/new_reference.json --catalog data/new_catalog.json --out outputs/annotation_review/new_queue.json`. The reference JSON follows the evaluator format and adds source `width`, `height` and an honest `annotation_method`; the catalogue supplies source identity and complete battle groups. Keep new source labels separate from the frozen references.

A reviewer can inspect an image, edit its source coordinate boxes and save a JSON file containing `cars` and `tags`. The following is a format illustration, not an actual review:

```json
{
  "cars": [{"identity": "car_a", "class_id": 2, "visibility": "partial", "bbox": [100, 200, 220, 280]}],
  "tags": {"smoke": "light", "overlap": "partial"}
}
```

After actually checking the source frame, the human reviewer can record the explicit confirmations. The sample ID comes from `status`; the name below is a placeholder:

```powershell
.venv\Scripts\python.exe tools\annotation_review.py confirm --sample clip01_shot0_0.300000 --reviewer "Your name" --edits outputs/annotation_review/edits.json --confirm-image --confirm-boxes --confirm-identities
```

Save and transfer the queue as JSON. Validated merging preserves the source and frame metadata and never promotes pending assistant drafts:

```powershell
.venv\Scripts\python.exe tools\annotation_review.py import --input outputs/annotation_review/returned_queue.json
.venv\Scripts\python.exe tools\annotation_review.py export --split test --out outputs/annotation_review/reviewed_test.json
.venv\Scripts\python.exe tools\annotation_review.py export --format detector --split tune --out outputs/annotation_review/reviewed_detector_tune.json
```

Exports contain confirmed frames only and fail when none are available. Evaluator exports default to the target car class and work with `driftlens.evaluation`; class and review metadata stay attached. Detector exports retain car and truck boxes, omit hidden positions, carry source dimensions and grouping metadata, and require an explicit split. Test exports record `training_allowed=false`. These JSON exports prepare later training data; they do not themselves create a trained model or certify dense tracking quality. New train and validation labels must cover whole distinct video or battle groups, and any newly reviewed test set must remain outside training.

The UI can use `build_review_queue`, `load_review_queue`, `confirm_sample`, `save_review_queue`, `import_review_queue`, `export_reviewed_labels` and `export_detector_labels` from `driftlens.annotation_review`. Meaningful tests exercise the real evaluator on exported labels, hidden boxes, bounds, classes, tags, split leakage, stale attestations and preservation of frozen source context.

## Enhanced demo and own clip review

`data/inference_profiles.json` selects the enhanced upload profile and the default complete replay. `data/full_run_enhanced_catalog.json` freezes its separate inference settings and assistant reviewed role intervals. `outputs/visibility_report.json` compares the same source interval with the preserved original run. It is descriptive demonstration evidence, not a held out detector score. `tools/finalize_enhanced_demo.py` rebuilds this reviewed demonstration from cached current detections; its explicit maps are historical assistant visual decisions, not automatic role inference. `tools/verify_project.py` validates both saved full runs, actual box associations, source and replay time, separation geometry and recovered class provenance.

Browser checks in `tools/browser_improvements_qa.cjs` exercise a real private raw source excerpt upload, automatic reports, actual replay playback, report downloads and rejection of incomplete human review without changing the queue. `outputs/browser_improvements_qa.json` records the successful run. `tools/verify_upload_workflow.py` additionally applies source reviewed bounded roles to a private verification copy and rebuilds every export, then clears roles again while preserving source metadata and detector observations. Its own IDs are inspected independently of the complete run's IDs. `outputs/upload_workflow_qa.json` records that check. This reused test clip does not establish general accuracy on personal footage.

In Process a clip select Upload my video. Inputs remain under `data/uploads` and results under `outputs/runs`, outside public Git exports. Files are probed as actual video before acceptance and capped at 200 MB. Analyze up to 60 seconds per result. Automatic reports describe detections and unknown measurements. In Shot review open Correct roles within bounded clip intervals, assign distinct observed IDs within each camera view, then explicitly save. Replay, metrics, analysis JSON, readable report and summary publish together; failed publication restores previous exports. Uncovered time stays unknown, and intervals spanning a detected camera cut are rejected.

R means a current nonvehicle class box matched conservatively to a recent confirmed vehicle within the same view. Recovery requires geometry and color appearance agreement and expires after 0.5 seconds. Original class, confidence and appearance evidence remain in CSV exports. Its confidence is the original class score, not vehicle confidence. Orientation and appearance recovery must receive separate validation on new footage before any general accuracy claim.

## V1 uploaded battle regression

The first 28 second Thorne versus Olsen upload was outside the original catalogue and demo windows but came from the same Long Beach 2024 broadcast, source seconds 3025 through 3053. Its enhanced profile processed 280 samples in 611.995 seconds. The automatic safeguard reported one view and missed three assistant reviewed cuts at clip times 5.7, 14.8 and 23.3 seconds. Postprocessing camera annotations and assistant visual roles yielded 173 accepted paired samples, or 61.8 percent, without changing current boxes or IDs. Those annotations did not reset tracking at missed cuts. There are no independent detector or identity reference labels for this excerpt.

Preserve `outputs/unseen_clip_test/automatic_output_before_review` and the original `review_check.json`. Once a clip informs a correction it is a regression example. Check the repaired camera boundaries against the visibly reviewed source samples, verify tracker and appearance anchors reset at actual cuts, and inspect any extra detected boundary inside continuous footage. A detector improvement on this known clip is different from generalisation to another event. Reprocessing changes local track IDs, so historical role maps must never be copied without fresh source and replay inspection.

Background vehicles and false boxes can remain unassigned observations; they must not enter tandem separation merely because they have IDs. Withhold merged boxes, uncertain liveries and missing participants. Compare final results under fresh independently reviewed intervals, preserving the original counts and settings. A reviewed pair rate is descriptive accepted observation availability and cannot be reported as generic model accuracy. New event validation requires a separate untouched source and independent reviewed labels.

The final direct scans contain 280 samples and 279 adjacent comparisons for the uploaded battle, and 267 samples with 266 comparisons for the original continuous 26.7 second source interval. The revised safeguard finds cut times 5.7, 14.8 and 23.3 seconds in the uploaded excerpt and 8.2, 13.1 and 17.5 seconds in the original interval, with no extra flags in either scan. The historical original replay contains 268 frames because four child views were sampled separately. These counts refer to different sampling procedures and must remain distinct.

Screening saved appearance recovery seeds retained the one inspected real yellow car recovery and rejected two curb artifacts in the uploaded battle. In the enhanced demo it retained 33 inspected car recoveries and rejected one known bin artifact out of 34 saved recovered observations. This is candidate screening on cached known examples, not a new detector benchmark, full retracking or replacement pair score. Current observed box, recent vehicle anchor, geometry, appearance and conservative crop acceptance remain required. Native background predictions can remain candidates without entering the reviewed pair.

The final visibility implementation revision is recorded separately from the unchanged historical data schema. Preserve old evidence metadata and role maps with their original outputs. Fresh processing under changed cut and recovery logic must record the new revision and receive fresh role review; do not promote historical roles merely because a source interval matches.

Inspect `outputs/v1_camera_cut_regression/new_upload.json` and `default_demo.json` with their corresponding comparison CSVs for the full numerical camera scan. `outputs/recovery_quality_evidence.json` contains the saved candidate probe. `outputs/v1_closeout_verification.json` holds final fresh upload, runtime, browser, provenance, README and test checks. Private fixture video, inspected source images and browser artifacts belong under `outputs/upload_validation/v1_closeout`; they are separate from numerical public evidence.

## Source provenance

The selected broadcast is the official [Long Beach 2024 Top 16 ALL ACTION video](https://www.youtube.com/watch?v=nobounLesY4). Keep its publisher, URL, downloaded filename, source timestamp range, and permission status with clip metadata. Record permission status as unverified unless the rights holder has supplied authorisation or an applicable licence has been verified.

A downloaded local copy does not by itself authorise redistribution of the source or derived video. Public portfolio evidence can include project code, methods, and numerical summaries while footage distribution remains subject to its own rights. The source code license is separate from footage rights. The detection backend's published [licensing page](https://www.ultralytics.com/license) explains its AGPL option.


Final runtime verification used a fresh browser upload of a four second excerpt spanning a known cut. Revision 3 detected that boundary, reset track IDs and recovery scope, and exported only current detector observations. Original model weights and frozen historical pair totals were retained. Static role maps from visibility revision 2 are not reused automatically after fresh revision 3 processing because changed detections can renumber IDs; review the new result. Evidence and browser checks are in `outputs/v1_closeout_verification.json`. The test suite passed 122 tests and 80 subtests.
