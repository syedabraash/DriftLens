# Evaluation protocol

DriftLens reports evidence about a specific pretrained detector and tracker on specific local video intervals. It does not currently establish accuracy across Formula Drift events. The first experiments use excerpts from one broadcast, so their shared venue and camera style limit generalisation.

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
3. An unmatched visible human box is a false negative.
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

## Complete run demonstration

`data/full_run_catalog.json` covers source seconds 1904.5 through 1931.2 with four contiguous camera shots and 801 source frames. The independently sampled replay has 268 frames at 10 fps and lasts 26.8 seconds. `run_time` preserves source elapsed time; `playback_time` includes the fractional camera boundary rounding in the exported replay. Native player jumps use whole seconds with less than one second of pre-roll.

Recreate the full run with the independent environment:

```powershell
.venv\Scripts\python.exe -m driftlens.full_run data\full_run_catalog.json --tracker botsort --size 640
```

Existing complete child results are cached only when source path, source file size and modification time, model checksum, source range, pipeline revision, tracker, sampled rate, inference size and suppression setting match. Add `--force` for fresh inference. A freshly processed source requires visual confirmation of the saved role intervals because tracker IDs are not intrinsic car identities. Preset review intervals are withheld when the source or model checksum differs from the reviewed manifest. To reassemble already reviewed child results without inference, use `--assemble-only`.

The complete run uses class agnostic suppression to remove overlapping car and truck predictions on the same vehicle. The original diagnostic comparison remains 416 pixels with its original suppression setting. Original scores are not accuracy estimates for this different demonstration profile. Complete run material overlaps the existing test broadcast and is not additional held out evidence.

Role intervals are half open seconds relative to each camera shot. An AI assistant inspected the exported observations against source livery and travel order, marking swaps, parked candidates, merged boxes and ambiguous spans. Unassigned intervals remain unknown even when a detector box exists. Measurements require two accepted current observations. Charts break at camera cuts, local identity mapping changes and missing samples. No automatic association across cameras is claimed, and no human expert validation has occurred.

The role form replaces a selected shot's interval map with a whole shot ID assignment. It stages updated replay and metrics, rebuilds the combined exports and restores all previous files if publication fails. Simple whole shot overrides require checking the entire shot for swaps.

Complete outputs are `annotated.mp4`, `timeline.csv`, `shots.json` and `summary.json` under `outputs/full_runs/full_run01`. Each child shot retains raw detections, observations and source frame indices. `outputs/full_run_report.json` records the numeric summary. Browser checks verify the actual 26.8 second video, jumps, chart rendering and downloaded exports.

## Broader evaluation

Reserve entire tandem runs for evaluation. Consecutive or near-identical frames from one run must not appear on both sides of a training or tuning split. If no detector training has occurred, still separate the clips used to choose thresholds from the final comparison clips. Add a second event or camera style before claiming robustness across venues.

Label clear shots, partial smoke, and overlapping cars separately. Report their sample counts and failure cases, not just an aggregate rate. A detector-only baseline can show whether the main failures come from the detector or from association. Compare ByteTrack and BoTSORT under the same sampling and detector settings and preserve unsuccessful examples.

Wall-clock throughput includes detection and tracking; export time may be recorded separately. A sampled analysis rate does not imply real-time throughput. The review must disclose CPU hardware, sample rate, and the number of frames processed.

## Source provenance

The selected broadcast is the official [Long Beach 2024 Top 16 ALL ACTION video](https://www.youtube.com/watch?v=nobounLesY4). Keep its publisher, URL, downloaded filename, source timestamp range, and permission status with clip metadata. Record permission status as unverified unless the rights holder has supplied authorisation or an applicable licence has been verified.

A downloaded local copy does not by itself authorise redistribution of the source or derived video. Public portfolio evidence can include project code, methods, and numerical summaries while footage distribution remains subject to its own rights. The source code license is separate from footage rights. The detection backend's published [licensing page](https://www.ultralytics.com/license) explains its AGPL option.
