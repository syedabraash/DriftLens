# DriftLens measured results

The finished local project contains twelve processed camera shots from the official Formula DRIFT Long Beach 2024 Top 16 ALL ACTION broadcast. The six test shots also have BoTSORT comparison exports. No detector training was performed. This is an offline review tool for a pretrained baseline.

## What was measured

The test set has 24 sparse frames and 48 approximate visible car boxes across three battles. Six tuning shots from three other battles are separate from the six test shots. Four tuning frames are labelled for the first demo and excluded from the test totals. The twelve shots total 53.4 seconds, with 28.2 seconds for tuning and 25.2 seconds for testing.

The AI assistant drafted the visual reference boxes and identities from source imagery independently of detector predictions. No human expert has validated them. Partial boxes describe visible extents, which makes their boundaries subjective. These results support a small diagnostic comparison, not a certified ground truth benchmark or a claim of general Formula Drift accuracy.

Matching uses descending IoU with one match per box, IoU at least 0.5 and maximum frame alignment error of 0.1 seconds. All 24 test references aligned to processed frames. Counts include partial visibility; no fully hidden positions were invented.

## Aggregate comparison

The raw detector produced 41 matched boxes, seven unmatched predictions and seven missed reference boxes. Its precision, recall and F1 were each 85.4 percent. Raw detections are saved independently of confirmed tracker observations.

| Confirmed observations | Precision | Recall and visible reference coverage | F1 | Sparse ID changes | Comparable transitions | Processing fps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ByteTrack | 100.0% | 83.3% | 90.9% | 4 | 25 | 3.770 |
| BoTSORT | 100.0% | 75.0% | 85.7% | 0 | 22 | 2.842 |

ByteTrack matched 40 of 48 reference boxes, missed eight and had zero unmatched confirmed predictions at the labelled times. BoTSORT matched 36, missed twelve and also had zero unmatched confirmed predictions. Those precision figures cover only this small sparse set; they do not establish that all detections throughout the videos are correct.

BoTSORT's zero sampled identity changes comes with lower coverage and fewer comparable transitions. A miss breaks the identity comparison chain. Changes inside gaps or between unlabelled frames can therefore be uncounted. The evidence does not establish a universal better tracker.

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

Run `launch.cmd` and select the prepared first replay. The Evidence tab contains tracker comparisons, clip metadata, this report and the JSON evaluation. The Process a clip tab accepts another local continuous interval.

The exact source, weights and label hashes are in `outputs/reproducibility_manifest.json`. Direct dependencies are pinned in `requirements.txt`, with installed versions recorded in `requirements.lock.txt`. The evaluation procedure and catalogue commands are in `docs/EVALUATION.md`.

Raw footage, inspection images and video derivatives stay local. Reuse and redistribution permission remains unverified. The project source is independently licensed under AGPL3; that license does not grant rights to the footage.
