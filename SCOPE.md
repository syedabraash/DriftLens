# DriftLens scope

Updated on 2 October 2026.

DriftLens is an implemented local Formula Drift tandem tracking and run review project. Its independent root is `G:\DriftLens`, with its own Python environment. The user authorised implementation after supplying `longbeach2024_action.mp4`. PitWall and the banking project remain separate.

## Purpose

Convert short real tandem shots into a numbered replay, observed vehicle trajectories, an image separation chart and reproducible diagnostic evidence. The technical question is how consistently a small pretrained detector and tracker preserve the two vehicle identities through smoke and overlap.

## Implemented workflow

1. Select a local video and a bounded source interval in the Streamlit interface or command line.
2. Run pretrained YOLOv8n detection on the CPU and associate observations using ByteTrack or BoT SORT.
3. Inspect the annotated replay and assign two different tracker IDs to lead and chase.
4. Display bounding boxes, IDs, assigned roles, confidence and observed trajectories.
5. Compute detected image center separation divided by the mean detected box width when both selected cars are observed.
6. Leave missing observations and separation values unknown rather than drawing hidden positions as measured facts.
7. Export the annotated MP4, observations CSV, sampled frame manifest, frame metrics and run summary.
8. Compare trackers on the same source interval and detector settings.
9. Evaluate saved observations against the supplied sparse visual reference labels.

The project uses offline CPU inference with a small model, nominally sampling 10 frames per second at 416 pixel inference size. The analysis sampling rate is different from measured processing throughput. Actual settings and timing belong in each run summary. The first setup needs internet access for free Python packages and model weights; subsequent analysis runs locally.

## Selected footage and shot library

The source is the official [Formula DRIFT Long Beach Top 16 ALL ACTION video from 2024](https://www.youtube.com/watch?v=nobounLesY4), supplied locally by the user as `longbeach2024_action.mp4`. The inspected local file has 1280 by 720 frames at 30 frames per second.

`data/clip_catalog.json` records the source, source timestamps, shot conditions, split, run group and thumbnail. The current library contains 12 continuous camera shots totalling 53.4 seconds. Six tuning shots total 28.2 seconds and six test shots total 25.2 seconds. Individual shots last 3.4 to 5.6 seconds because broadcast camera cuts limit continuous views.

Shots from the same battle are grouped on one side of the split. Three battle groups are used for tuning and three different groups for testing. This avoids placing alternate runs or adjacent broadcast views of one battle across the tuning and test boundary. The selected intervals exclude replay footage.

Raw footage and video derivatives remain private local files. The source URL and the unverified reuse status are recorded in the manifest. No footage redistribution permission has been established by this project.

## Diagnostic evaluation

The sparse reference set contains 24 test frames with 48 approximate visible vehicle boxes, plus four tuning frames with eight boxes. The reference labels were drafted by an AI assistant using visual inspection of source frames. They have not received human expert review.

These are diagnostic visual references rather than an expert verified benchmark. The evaluator checks detection precision, recall, F1, tracked reference coverage and identity changes across comparable labelled observations. It reports the actual aligned sample counts and comparable transitions. Hidden cars are excluded; their positions are not invented.

Sparse samples do not prove continuous tracking quality between labelled frames and cannot establish full HOTA, IDF1 or MOTA results. All footage comes from one event and venue. Broader robustness requires more events, denser annotations and independent human review. The saved annotation method and these limitations must accompany any numerical result.

`outputs/evaluation_report.json` and `docs/RESULTS.md` hold the measured comparison when generated. The portfolio contribution is the documented shot selection, reproducible pipeline, useful local review interface, diagnostic comparison and honest failure analysis. No detector fine tuning is claimed.

## Measurement boundaries

The separation chart measures a perspective dependent image proxy. It does not measure physical gap, vehicle speed, drift angle, judging proximity or driver skill. Such quantities need a separately calibrated and validated method.

The application does not produce official Formula Drift scores or identify driver names. Lead and chase are reviewer supplied roles. Initial visual role suggestions are attributed in the manifest and need review. Track IDs do not establish physical identity on their own, especially after smoke or overlap.

Use one continuous camera shot per analysis clip. A conservative cut safeguard resets tracking across detected cuts, but it does not replace visual shot selection or reconnect identities across views.

## Tools and storage

The implementation uses free local Python tools, OpenCV, PyTorch, Ultralytics YOLOv8n, ByteTrack, BoT SORT, Streamlit, CSV and JSON. Browser compatible video export uses the bundled FFmpeg dependency. CVAT Community is an optional future annotation tool, not a prerequisite installed by this implementation.

Source code, the detector and its dependencies have their own licences; those do not grant rights to the footage. Follow the applicable open source terms when publishing project code. Do not introduce a paid service or API without a new explicit user request.

Every project README must contain no dash characters. Validate that rule when creating or editing README files. Keep this project's code, data, environment, weights and outputs separate from PitWall and the banking project.
