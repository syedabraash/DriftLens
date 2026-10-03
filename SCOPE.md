# DriftLens scope

Updated on 3 October 2026.

DriftLens V1 is an implemented local, review assisted tandem tracking and run review project. Its independent root is `G:\DriftLens`, with its own Python environment. The user authorised implementation after supplying `longbeach2024_action.mp4`. PitWall and the banking project remain separate.

## Purpose

Convert a complete real tandem run into a combined replay, an explicitly segmented observation timeline, an image separation chart and reproducible diagnostic evidence. The technical question is how consistently a small pretrained detector and tracker preserve the two vehicle identities through smoke and overlap.

V1 includes private bounded uploads, observed detection and tracking, replay review, explicit role corrections, seed based appearance continuity and CSV, JSON and readable report exports. A reviewer identifies a reliable initial pair; conservative automatic proposals can then assign later observed IDs across camera views or tracker fragmentation. The tool does not infer lead and chase from tracker numbering, provide physical telemetry or guarantee identities on arbitrary footage. Users can test their own videos locally, but personal footage and other events have not yet been independently evaluated. `docs/DEMO_GUIDE.md` explains testing and recording the finished result.

The default demonstration covers 1904.5 through 1931.2 seconds of the source, with four contiguous camera shots. It includes 26.7 seconds of broadcast coverage from launch and initiation through the visible finish. Independent sampling of the four shots produces a 26.8 second replay with 268 frames. Exact source and replay timestamps remain available separately.

## Implemented workflow

1. Select a local video and a bounded source interval in the Streamlit interface or command line.
2. Run pretrained YOLOv8n detection on the CPU and associate observations using ByteTrack or BoT SORT.
3. Inspect the annotated replay and seed two different observed tracker IDs as lead and chase. Propose later role assignments using appearance, inspect the evidence and correct unsuitable matches.
4. Display bounding boxes, IDs, assigned roles, confidence and observed trajectories.
5. Compute detected image center separation divided by the mean detected box width when both selected cars are observed.
6. Leave missing observations and separation values unknown rather than drawing hidden positions as measured facts.
7. Export the annotated MP4, observations CSV, sampled frame manifest, frame metrics and run summary.
8. Compare trackers on the same source interval and detector settings.
9. Evaluate saved observations against the supplied sparse visual reference labels.
10. Maintain a separate human annotation review queue, confirm corrected source boxes and shot local identities, tag smoke and overlap, and export only confirmed samples without changing frozen references.

The original diagnostic comparison uses offline CPU inference sampling 10 frames per second at 416 pixel inference size. The complete run uses BoTSORT at 640 pixel inference with class agnostic suppression to reduce duplicate boxes on the same car. The analysis sampling rate is different from measured processing throughput. Actual settings and timing belong in each run summary. The first setup needs internet access for free Python packages and model weights; subsequent analysis runs locally.

## Selected footage and shot library

The source is the official [Formula DRIFT Long Beach Top 16 ALL ACTION video from 2024](https://www.youtube.com/watch?v=nobounLesY4), supplied locally by the user as `longbeach2024_action.mp4`. The inspected local file has 1280 by 720 frames at 30 frames per second.

`data/clip_catalog.json` records the source, source timestamps, shot conditions, split, run group and thumbnail. The current library contains 12 continuous camera shots totalling 53.4 seconds. Six tuning shots total 28.2 seconds and six test shots total 25.2 seconds. Individual shots last 3.4 to 5.6 seconds because broadcast camera cuts limit continuous views.

Shots from the same battle are grouped on one side of the split. Three battle groups are used for tuning and three different groups for testing. This avoids placing alternate runs or adjacent broadcast views of one battle across the tuning and test boundary. The selected intervals exclude replay footage.

Raw footage and video derivatives remain private local files. The source URL and the unverified reuse status are recorded in the manifest. No footage redistribution permission has been established by this project.

## Diagnostic evaluation

The sparse reference set contains 24 test frames with 48 approximate visible vehicle boxes, plus four tuning frames with eight boxes. The reference labels were drafted by an AI assistant using visual inspection of source frames. They have not received human expert review.

These are diagnostic visual references rather than an expert verified benchmark. The evaluator checks detection precision, recall, F1, tracked reference coverage and identity changes across comparable labelled observations. It reports the actual aligned sample counts and comparable transitions. Hidden cars are excluded; their positions are not invented.

Sparse samples do not prove continuous tracking quality between labelled frames and cannot establish full HOTA, IDF1 or MOTA results. All footage comes from one event and venue. Broader robustness requires more events, denser annotations and independent human review. The saved annotation method and these limitations must accompany any numerical result.

`outputs/evaluation_report.json` and `docs/RESULTS.md` hold the original measured comparison. A separate completed twelve epoch CPU detector adaptation used 24 training frames and eight validation frames from tuning battles. Its candidate reduced precision and recall on the frozen 24 frame test reference set at 640 pixel inference with class agnostic suppression. `outputs/finetuning_report.json` records the settings, counts and unsuccessful promotion gate. The original pretrained detector remains active. This is detector only diagnostic evidence using approximate assistant labels, not a verified improvement in tracking or run analysis.

`outputs/annotation_review/queue.json` keeps proposed corrections separate from `data/annotations/labels.json`. A review needs a named reviewer, a timezone timestamp and explicit image, box and shot local identity confirmations. This records a self attestation, not expert certification. Pending records retain assistant draft provenance. Source bounds, classes, visibility, smoke and overlap tags, shot local identity scopes, fingerprints and video or battle split grouping are validated before export. The queue contains no human confirmations until a person actually supplies them. Frozen historical scores are not silently replaced by reviewed exports.

## Measurement boundaries

The separation chart measures a perspective dependent image proxy. It does not measure physical gap, vehicle speed, drift angle, judging proximity or driver skill. Such quantities need a separately calibrated and validated method.

The application does not produce official Formula Drift scores or identify driver names. The initial lead and chase pair is reviewer supplied. Later role continuity proposals are attributed as automatic appearance matches, with their seed, settings and uncertainty retained separately from manual review. Track IDs and similarity scores do not establish physical identity or calibrated identity probability, especially after smoke or overlap.

Prefer one continuous camera shot for an initial upload test. The full run manifest joins ordered, contiguous shots into a complete review. Every view has independently reviewed roles and local IDs. Explicit visual role intervals exclude ambiguous observations and allow known fragments to be reviewed without pretending uninterrupted identity. Cuts, unknown gaps and role mapping changes break chart lines. The cut safeguard resets tracking across detected cuts. The separate continuity step compares visible participant appearance against reviewed seeds to propose roles for later current IDs. It withholds weak or competing matches and does not reconnect hidden positions. Unassigned candidates can include native background vehicle predictions and false boxes; only accepted observations of the reviewed pair enter tandem measurements.

The complete run reuses material from the existing test broadcast and is a demonstration rather than additional independent evaluation. Saved role intervals apply only to the pinned source, weights and BoTSORT 640 pixel profile. Other tracker or inference settings need fresh role review.

A new 28 second Thorne versus Olsen battle excerpt from source seconds 3025 through 3053 was uploaded outside the original selected windows. Its first reviewed output contained 173 accepted pairs in 280 samples, after assistant visual roles and manual camera annotation. The initial automatic safeguard missed all three reviewed cuts. It is from the same broadcast and has no independent expert labels. Once inspected to guide final camera fixes, this excerpt became a regression case; rechecks measure repair of this known example rather than new held out event accuracy. The original automatic output remains preserved. Background false detections, overlapping boxes and fragmented IDs require review.

The final camera safeguard detects all six visually reviewed boundaries with no additional flags in direct scans of the original complete interval and the new battle excerpt. This is regression evidence on two known examples. It does not establish arbitrary camera cut accuracy. The saved reviewed replays, local IDs and pair totals remain preserved, and reprocessed clips require fresh role review. Optional appearance recovery now applies an additional conservative crop check to reject inspected curb and bin artifacts. This may withhold recovery of monochrome cars. Native detector false candidates and unfamiliar appearance errors remain possible.

## Tools and storage

The implementation uses free local Python tools, OpenCV, PyTorch, Ultralytics YOLOv8n, ByteTrack, BoT SORT, Streamlit, CSV and JSON. Browser compatible video export uses the bundled FFmpeg dependency. CVAT Community is an optional future annotation tool, not a prerequisite installed by this implementation.

Source code, the detector and its dependencies have their own licences; those do not grant rights to the footage. Follow the applicable open source terms when publishing project code. Do not introduce a paid service or API without a new explicit user request.

Every project README must contain no dash characters. Validate that rule when creating or editing README files. Keep this project's code, data, environment, weights and outputs separate from PitWall and the banking project.

## Role continuity evidence boundary

The continuity feature is a bounded heuristic built from observed crop appearance, not a new trained reidentification network. Similarity and ambiguity gates reduce unsupported assignments but cannot eliminate appearance mistakes or participant swaps. New view decisions remain inspectable and editable, and unknown or missing observations do not contribute to tandem separation.

The user's uploaded `test` clip exposed the camera and identity issue and now guides the implementation. It is a regression and tuning example, not an untouched independent evaluation. Historical complete demo results, sparse labels and unsuccessful training evidence remain frozen. No new expert review or broader event accuracy is established by adding this feature.
