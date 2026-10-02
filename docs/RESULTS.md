# DriftLens measured results

## Enhanced demonstration and uploads on 3 October 2026

The default `full_run02` reprocesses the same 26.7 second broadcast interval as the preserved `full_run01`, with the original YOLOv8n weights. Reviewed pair availability rises from 120/268 samples (44.8 percent) to 191/268 (71.3 percent). Per shot accepted pairs are 61/82, 38/49, 24/45 and 68/92. These descriptive results include source reviewed role intervals chosen on this demonstration, not an independent accuracy evaluation.

In the final shot's last 1.8 seconds, both participants now have accepted current observations in all 18 sampled frames, compared with zero before. The green car's late detector boxes were originally classified as bottle, and two orange car boxes as cell phone. Conservative same camera appearance recovery preserves those source classes and scores and marks recovered boxes R in the replay. Thirty four exported observations use recovery; 23 accepted paired samples include it. Those scores are not car or identity probabilities.

The profile uses 640 pixel inference with native and 90/270 degree orientation passes, class agnostic suppression, no global motion compensation, and low confidence tracking without score fusion. It took 284.888 seconds to process 268 samples on this CPU with uncontrolled load, compared with 109.624 seconds for the original replay. This gain costs processing time. A larger model and higher resolution probes did not reliably improve the late case and were not activated.

Visible misses remain around the final hairpin at local times 3.0 to 4.1 seconds, with the orange car still untracked at 4.2 to 4.8 seconds. Merged boxes, parked vehicles and uncertain overlaps are excluded from pair geometry. No hidden positions are generated. The previous fine tuned candidate remains inactive because it regressed on its saved comparison.

Uploads accept private local files up to 200 MB and intervals up to 60 seconds, returning current observations, an annotated replay, a readable review and CSV/JSON exports. Lead and chase remain unknown until visually assigned within bounded intervals. Camera cuts require separate decisions. Human annotation review is available in Evidence, with zero confirmed human labels so far. Stronger training needs diverse reviewed examples and a separate untouched event level test.

The matched numerical report is `outputs/visibility_report.json`. Earlier measurements below retain their original profile and reference set.

The local project opens with a complete tandem run from the official Formula DRIFT Long Beach 2024 Top 16 ALL ACTION broadcast. It also retains twelve diagnostic camera shots and six BoTSORT comparison exports. A completed twelve epoch detector adaptation experiment is recorded separately below. The original pretrained detector remains active because the candidate regressed on the frozen diagnostic test references.

## Original 640 pixel complete run demonstration

The source interval is 1904.5 through 1931.2 seconds, spanning 26.7 seconds and 801 original frames from launch and initiation through the visible finish. Four contiguous camera views produce 268 sampled frames and a 26.8 second combined replay at 10 fps. The timeline retains both exact source elapsed time and encoded playback time. Native player jumps use whole seconds with less than one second of pre-roll.

This qualitative replay uses BoTSORT at 640 pixel inference with class agnostic suppression to reduce duplicate car and truck boxes on the same vehicle. It reuses footage from the existing test broadcast. It is not an additional independent evaluation, and the 416 pixel diagnostic scores below are not accuracy estimates for this profile.

| Camera shot | Sampled frames | Frames with both accepted observations | Pair availability |
| --- | ---: | ---: | ---: |
| Launch and initiation |82|58|70.7%|
| Bridge transition |49|34|69.4%|
| Outer zone and transition |45|10|22.2%|
| Aerial hairpin and finish |92|18|19.6%|
| Complete replay |268|120|44.8%|

Pair availability means that both visually accepted roles had current tracker observations. It does not establish accuracy or physical identity. The source footage covers the complete run even where tracking is absent. Unknown measurements remain gaps instead of being filled from hidden or interpolated positions.

The AI assistant visually reviewed local track IDs against source car livery and travel order. Explicit half open role intervals exclude uncertain or merged boxes. Shot01 loses chase ID 2, and later ID 12 changes from a green or merged box to orange blue chase. In shot02, green lead fragments from ID 1 to ID 4; final merged boxes are excluded. In shot03, ID 14 changes from green lead to orange blue chase before green reacquires ID 17. Shot04 ID 29 also changes cars, while IDs 21, 24 and 32 are parked or off-track candidates. The late aerial pair is green lead ID 48 and orange blue chase ID 47. Tracking disappears again before the visible finish. These are curated AI visual decisions without human expert validation, not automatic identity guarantees.

Charts break at camera cuts, mapping changes and unknown samples. Local IDs are never joined automatically across views. Video overlays clear trajectories when a reviewed role mapping changes. Users can edit ordered, nonoverlapping role intervals within a camera shot, preserve a single known role or mark an entire shot unknown. Uncovered time stays unknown. An optional whole shot assignment replaces that shot's interval map. Both modes rebuild complete exports and written conclusions with rollback on failure, recording an unverified user assignment rather than expert validation.

The four detector jobs processed 268 frames in 109.624 seconds, an aggregate 2.445 fps on the inspected CPU. This includes decoding, inference and tracking, excluding model loading, role review and replay export. It is an offline demonstration with local workload variability.

`data/full_run_catalog.json` records source frame boundaries, livery references, processing context and review intervals. `outputs/full_run_report.json` records the completed numerical summary. Full replay media and visual inspection sheets remain local and ignored by Git.

## Local clip intake and readable review

In **Process a clip**, **Analyze your own clip** accepts **Upload my video** or a saved local source. Supported video extensions are MP4, MOV, MKV, AVI and WebM. Uploads are limited to 200 MB and each analysis interval to 60 seconds. The selected analysis profile records its detector and tracker settings; this interface does not silently adopt the unsuccessful trained candidate.

Processing creates sampled frame metadata, observed vehicle boxes, local tracker IDs, an annotated replay, a readable review, analysis JSON and numerical exports. Open the resulting selection in **Shot review**. A written review can explain unavailable measurements before roles are assigned; it does not infer physical identities, judging scores or driver skill. Timestamped review events and exact annotated sample images provide locations to inspect, rather than proving the event's cause automatically.

Use **Correct roles within bounded clip intervals** after checking visible livery and travel order. Assigned track IDs must have been observed inside their intervals, and assigned intervals must stop at detected camera cuts. Leave uncertain IDs empty or use **Mark all clip roles unknown**. Saving rebuilds the replay, frame metrics and reports together. Pair measurements require two accepted current observations; a detector box alone does not establish a valid lead or chase role.

Uploaded sources and derivatives remain private local files. Successfully uploading or processing a clip is not an accuracy evaluation on that footage. The current reference evidence remains limited to the selected Long Beach broadcast and approximate assistant labels. New events and personal videos need their own independent review and evaluation.

## What was measured

The test set has 24 sparse frames and 48 approximate visible car boxes across three battles. Six tuning shots from three other battles are separate from the six test shots. Four tuning frames are labelled for the first demo and excluded from the test totals. The twelve shots total 53.4 seconds, with 28.2 seconds for tuning and 25.2 seconds for testing.

The AI assistant drafted the visual reference boxes and identities from source imagery independently of detector predictions. No human expert has validated them. Partial boxes describe visible extents, which makes their boundaries subjective. These results support a small diagnostic comparison, not a certified ground truth benchmark or a claim of general Formula Drift accuracy.

Matching uses descending IoU with one match per box, IoU at least 0.5 and maximum frame alignment error of 0.1 seconds. All 24 test references aligned to processed frames. Counts include partial visibility; no fully hidden positions were invented.

## Original 416 pixel aggregate comparison

The raw detector produced 41 matched boxes, seven unmatched predictions and seven missed reference boxes. Its precision, recall and F1 were each 85.4 percent. Raw detections are saved independently of confirmed tracker observations.

| Confirmed observations | Precision | Recall and visible reference coverage | F1 | Sparse ID changes | Comparable transitions | Processing fps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ByteTrack | 100.0% | 83.3% | 90.9% | 4 | 25 | 3.770 |
| BoTSORT | 100.0% | 75.0% | 85.7% | 0 | 22 | 2.842 |

ByteTrack matched 40 of 48 reference boxes, missed eight and had zero unmatched confirmed predictions at the labelled times. BoTSORT matched 36, missed twelve and also had zero unmatched confirmed predictions. Those precision figures cover only this small sparse set; they do not establish that all detections throughout the videos are correct.

BoTSORT's zero sampled identity changes comes with lower coverage and fewer comparable transitions. A miss breaks the identity comparison chain. Changes inside gaps or between unlabelled frames can therefore be uncounted. The evidence does not establish a universal better tracker.

## Completed detector adaptation experiment

`outputs/finetuning_report.json` records one completed twelve epoch CPU transfer learning experiment. Twenty four frames with 59 approximate boxes from two tuning battle groups were used for training. Eight frames with 20 boxes from a third tuning battle group were used for validation. The original 24 test frames with 48 boxes from separate battles remained frozen. The completed training took 203.637 seconds. The final run loaded 355 of 355 pretrained tensors from a fresh unfused checkpoint; earlier interrupted setup attempts were not selected.

The candidate was selected by its best validation checkpoint before the final test comparison. These labels were drafted by the AI assistant without human expert review. The same broadcast, correlated frames and subjective visible extents limit what the experiment establishes. It is an actual training experiment, but it does not establish broad Formula Drift accuracy.

Both detectors were evaluated at 640 pixel inference, confidence 0.15, suppression IoU 0.5, class agnostic suppression, COCO car and truck classes and matching IoU 0.5. This setting differs from the original 416 pixel comparison above, so the two sets of scores must not be combined.

| Diagnostic comparison | Matched boxes | Unmatched predictions | Missed boxes | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline validation, 8 frames and 20 boxes | 14 | 4 | 6 | 77.8% | 70.0% | 73.7% |
| Candidate validation, same references | 14 | 2 | 6 | 87.5% | 70.0% | 77.8% |
| Baseline frozen test, 24 frames and 48 boxes | 44 | 1 | 4 | 97.8% | 91.7% | 94.6% |
| Candidate frozen test, same references | 43 | 2 | 5 | 95.6% | 89.6% | 92.5% |

Validation F1 improved by 4.1 percentage points, while test recall fell by 2.1 points and precision fell by 2.2 points. The candidate failed the promotion gate and remains an inspection artifact at `models/driftlens_finetuned.pt`. Original `models/yolov8n.pt` stays active; its baseline checksum and the frozen reference checksum are recorded in the report. Detector scores do not measure role correctness, full run pair availability or tracking accuracy. No automatic promotion or further training occurred as part of the review interface improvements.

## Human annotation review readiness

The separate review queue at `outputs/annotation_review/queue.json` starts with the 28 frozen reference samples as pending assistant drafts. Open **Evidence → Review reference labels for future training** to inspect original frames, edit source pixel boxes, classes and visibility, and record smoke and overlap tags. A name and three explicit confirmations are required to save a human self attestation. Creating the queue does not validate its labels. The delivered queue has zero human confirmations.

`driftlens/annotation_review.py` and `tools/annotation_review.py` support queue creation, a named reviewer with a timezone timestamp and explicit image, box and shot local identity confirmations, validated import, and exports containing only confirmed samples. Review records preserve the original boxes and annotation method, while changes invalidate an existing confirmation. Bounds, car and truck classes, visibility, smoke and overlap tags, shot local identities and grouped splits are checked. Hidden cars cannot have invented boxes.

The default battle grouping preserves the historical tuning and test separation within the source video. Video grouping can instead require an entire video to stay in one split; it rejects this broadcast's existing mixed split. Different battles from one broadcast still share camera and venue characteristics. Reviewed evaluator exports are separate from the frozen reference file, and detector exports retain split metadata, treating test samples as unavailable for training. Procedures are in `docs/EVALUATION.md`. Independent human checks, denser labels and a new event remain necessary before broader claims or another meaningful training experiment.

## Per shot diagnostics

Each shot has four labelled frames and eight reference boxes. Coverage counts only matched boxes with confirmed tracker IDs. The change count is limited to comparable sparse samples.

| Shot | Condition | ByteTrack coverage | ByteTrack changes | BoTSORT coverage | BoTSORT changes |
| --- | --- | ---: | ---: | ---: | ---: |
| clip07 | Clear then overlap | 100.0% | 1 | 100.0% | 0 |
| clip08 | Smoke and overlap | 87.5% | 1 | 87.5% | 0 |
| clip09 | Smoke and overlap | 75.0% | 0 | 75.0% | 0 |
| clip10 | Smoke and overlap | 62.5% | 0 | 12.5% | 0 |
| clip11 | Clear then overlap | 100.0% | 1 | 100.0% | 0 |
| clip12 | Smoke and overlap | 75.0% | 1 | 75.0% | 0 |

The clear then overlap shots have 16 reference boxes: both trackers matched all sixteen, while ByteTrack recorded two sampled identity changes. The four smoke and overlap shots have 32 boxes: ByteTrack matched 24 and BoTSORT matched 20. Conditions are coarse shot descriptions rather than frame specific visibility strata.

## Failure examples to inspect

In clip08, the green reference car matches ByteTrack ID 1 at 0.3 and 1.3 seconds, then ID 2 at 2.3 and 3.3 seconds. The other car had previously matched ID 2. The second car is missed at 2.3 seconds and later receives ID 4. This illustrates why retaining two IDs does not guarantee correct roles through overlap.

In clip12, the yellow reference car changes its ByteTrack ID during smoke. Ambiguous baseline results remain unassigned instead of presenting a reliable lead and chase separation curve. The first demo and selected BoTSORT comparisons have visually reviewed starting assignments, with their assistant attribution recorded in each summary. They still require replay inspection.

In clip10, BoTSORT matches only one of eight reference boxes and has no comparable identity transitions. Its zero change count is uninformative in that shot. In clip09, both trackers lose the second car in some labelled frames, so separation must remain unknown during those missing observations.

## Hardware and timing

The inspected computer has an Intel Core i3 2350M at 2.30 GHz, approximately 11.9 GB memory and Intel HD Graphics 3000. Inference uses CPU PyTorch with two threads. YOLOv8n uses 416 pixel inference, confidence threshold 0.15, detection IoU threshold 0.5 and COCO car and truck classes. BoTSORT uses the pinned backend's default configuration without appearance reidentification.

The source is 1280 by 720 at 30 fps. Processing samples ten frames per second. Each tracker processes 252 frames over the six test shots. Measured processing time is 66.842 seconds for ByteTrack and 88.675 seconds for BoTSORT. Aggregate throughput is total frames divided by total measured seconds.

Timing includes source decoding, inference and tracking. It excludes model loading and replay encoding. Other local verification was running during this measurement, so throughput varies with workload. These are observed offline speeds, not a controlled hardware benchmark or a claim of real time analysis.

## Measurement limits and implementation checks

Separation is image centre distance divided by mean bounding box width. Perspective, zoom, overlap and changing box shape affect it. The project does not estimate metres, speed, drift angle, judging scores or driver skill. Missing observations create empty values and chart gaps. A camera cut resets tracker state; the cut safeguard is a heuristic, and selected clips were also inspected visually as continuous shots.

Raw detections are captured before tracking. A local adapter preserves their original indices through the pinned tracker's high and low confidence filtering. Every confirmed observation must map to one unique current detector box. The project validator checks this across all eighteen saved results. Replay encoding uses H264 with browser compatible pixel format.

The unit suite covers matching, duplicate boxes, frame alignment, sparse identity comparisons, missing measurements, failed decoding, failed export recovery and actual ByteTrack and BoTSORT association with high and low confidence detections. Browser checks cover playback, navigation, charts and real exports; saved evidence is under `outputs/screenshots`.

## Reproduce and inspect

Run `launch.cmd` to open Complete run. Shot review retains the shorter examples and uploaded clip results with readable reports. Evidence contains separate saved comparisons, clip metadata, this report and the original JSON evaluation. Process a clip accepts a private upload or another saved local source interval. Full run reproduction and annotation review commands are in the evaluation notes.

The exact source, weights and label hashes are in `outputs/reproducibility_manifest.json`. Direct dependencies are pinned in `requirements.txt`, with installed versions recorded in `requirements.lock.txt`. The evaluation procedure and catalogue commands are in `docs/EVALUATION.md`.

Raw footage, inspection images and video derivatives stay local. Reuse and redistribution permission remains unverified. The project source is independently licensed under AGPL3; that license does not grant rights to the footage.
