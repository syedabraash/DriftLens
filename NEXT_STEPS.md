# DriftLens V1 handoff

Updated on 3 October 2026.

V1 provides local private uploads, observed vehicle tracking, replay review, bounded lead and chase corrections, reviewed seed appearance continuity across local ID and camera changes, image separation with explicit gaps, a readable report and numerical exports. It uses the original pretrained detector. The twelve epoch candidate regressed on the saved diagnostic test and remains inactive.

## Run and test locally

1. Open `G:\DriftLens` and double click `launch.cmd`.
2. Keep its terminal open and open [the dashboard](http://127.0.0.1:8510).
3. In **Process a clip**, choose **Upload my video**. Begin with a ten to fifteen second continuous view showing both cars, under 200 MB. The analysis interval is capped at 60 seconds.
4. Enter a new result name and select **Process clip locally**. Let processing finish before recording a demo.
5. Keep **Automatically identify and follow tandem** enabled during processing, or choose **Automatically find and follow tandem** on a saved result. The app can reuse a saved starting pair without repeating local IDs, and older camera tracking is repaired separately. If the app finds participants but cannot determine reliable travel order, confirm one clear starting pair using **Follow lead and chase across views**. Inspect **MATCH** labels and unknown spans, and use the bounded editor only for corrections involving IDs present in each interval.
6. Read the replay, image separation chart, missing observations and report together. Export the report and numerical evidence.

Use [the demo guide](docs/DEMO_GUIDE.md) for recording steps, a 45 to 60 second walkthrough and paste ready LinkedIn wording. A single car video can exercise upload and detection, but paired tandem measurements require two observed participants and reviewed roles.

## What the current results establish

The reused enhanced complete demo has 191 accepted pairs in 268 samples, or 71.3 percent, under assistant visual review. Its original settings and counts remain separately preserved. The first new 28 second battle upload provided 173 accepted pairs in 280 samples, or 61.8 percent, after assistant visual roles and manual camera annotation. The automatic safeguard missed three cuts. This same broadcast example now informs regression checks, so a repair result cannot become a new independent accuracy claim. Full settings, limitations and final check evidence belong in `docs/RESULTS.md`.

The final safeguard found all six reviewed cuts with no other flags across direct scans of these two complete intervals. Historical stitched replay counts, IDs and pair totals remain preserved. Screening the saved appearance recovery candidates rejected the inspected curb and bin artifacts while retaining inspected car recoveries; native false candidates still need review. These checks address known failures and do not establish accuracy on arbitrary footage.

The user's `test` upload now guides the continuity repair and is a tuning and regression case. Appearance continuity is not a new trained detector or an independent identity benchmark. No human expert labels or independent personal video evaluation have been completed. A reviewer must still inspect camera boundaries, identities, background false detections and overlaps. The chart reports a perspective dependent image proxy, without physical distance, throttle, brake, speed, drift angle or official scores.

## Future quality work

1. Test the user's own footage and preserve failures, actual settings, processing time and source permissions with each result.
2. Obtain named human review of visible boxes and shot local identities in the separate annotation queue. Its current 28 samples remain pending assistant drafts with zero human confirmations.
3. Add diverse footage from another event or camera style, with entire videos or battle groups held in one split. Keep a new independent test untouched while choosing fixes and settings.
4. Evaluate camera cut false positives and misses, participant coverage and identity errors across that expanded set before claiming broader reliability. Denser labels are needed for continuous tracking conclusions.
5. Consider additional training only with a larger reviewed dataset and independent validation. Preserve the unsuccessful candidate and frozen historical results.

These are later quality milestones. The V1 portfolio claim remains a review assisted local analysis workflow with measured limitations.

## Portfolio files and privacy

Publish source code, setup instructions, methods and numerical summaries. Uploaded videos stay in `data/uploads`; results, frame images and replays stay in private outputs. The supplied official broadcast and its visual derivatives remain private while reuse permission is unverified. A public recording can use footage the user owns or is authorised to publish, or show numerical diagrams and dashboard sections without broadcast imagery.

The launcher and setup use the dedicated `G:\DriftLens\.venv`. Keep PitWall and the banking project separate. First setup needs internet for free dependencies and weights; normal processing then runs locally. Every README must contain no ASCII hyphens, Unicode dash punctuation or the Unicode minus sign.
