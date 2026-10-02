# Sources and provenance

Updated on 2 October 2026.

## Selected footage

The actual source is [Formula DRIFT Long Beach Top 16 ALL ACTION](https://www.youtube.com/watch?v=nobounLesY4), published by the official Formula DRIFT channel on 16 April 2024. The user supplied the local file `longbeach2024_action.mp4` for this project. The source metadata records Long Beach 2024, 1280 by 720 frames and 30 frames per second.

`data/clip_catalog.json` records each selected source interval, condition, tuning or test split, battle group and thumbnail. Twelve continuous shots total 53.4 seconds. Six tuning shots and six test shots are grouped by complete battle so views of the same battle do not cross the split.

The local recording, thumbnails and video derivatives stay private. Reuse and redistribution permission are unverified. Availability on an official public channel is provenance information, not a footage licence recorded by this project. Project code and numerical summaries can be prepared separately from footage distribution.

The earlier 2025 broadcast proposal is not part of the implemented dataset.

## Reference annotations

`data/annotations/labels.json` contains the sparse visual reference set. An AI assistant drafted approximate visible vehicle boxes and identity labels from inspected source frames. There has been no human expert review.

The set has 24 test frames with 48 boxes and four tuning frames with eight boxes. It supports diagnostic matching at those labelled times. It does not establish accurate hidden positions, uninterrupted identities, or a complete tracking benchmark. The saved annotation method and review status must remain with the results.

## Implemented tools and methods

1. [Ultralytics tracking documentation](https://docs.ultralytics.com/modes/track/) describes the tracking interface and the ByteTrack and BoT SORT options. The implementation uses the package version recorded in `requirements.txt` and pretrained YOLOv8n weights, not necessarily the newest models shown in current documentation.
2. [Ultralytics source repository](https://github.com/ultralytics/ultralytics) contains the detector and tracker implementation. Consult the installed version for the exact configuration used in a run.
3. [Ultralytics licensing information](https://www.ultralytics.com/license) describes its open source AGPL option. Code and model terms are separate from footage rights.
4. [OpenCV documentation](https://docs.opencv.org/4.x/) covers local frame decoding, image operations and video processing.
5. [PyTorch documentation](https://pytorch.org/docs/stable/index.html) documents the CPU inference runtime used by the detector.
6. [Streamlit documentation](https://docs.streamlit.io/) documents the local review interface, media display and exports.
7. [imageio FFmpeg dependency](https://github.com/imageio/imageio-ffmpeg) supplies the free local video encoder used for browser compatible output.
8. [CVAT Community repository](https://github.com/cvat-ai/cvat) is an optional future annotation reference. CVAT is not required or installed for this project's current sparse labels.

The implementation uses free local tools and no paid API. Exact dependency versions, detector settings, sample rate, source interval and measured processing time belong with each reproducible run.

## Interpretation

Separation is computed from observed box centers and widths in the image. The project does not infer physical distance, vehicle speed, drift angle or official Formula Drift scores. Reference matching, sparse identity checks and measured processing speed are described in `docs/EVALUATION.md` and the generated results report.
