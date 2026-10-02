# DriftLens next steps

The enhanced replay, readable run analysis, private video uploads, bounded role corrections and human label review workflow are implemented. The enhanced demo has 191 accepted pairs in 268 samples and all eighteen final samples contain both observed participants. This is a reused demonstration with assistant role review. It does not establish accuracy on other events. The next quality milestone requires actual human reviewed diverse footage, rather than another training run on the same small reference set.

Updated on 3 October 2026.

The offline CPU pipeline and local review interface have been implemented in `G:\DriftLens`. The footage source is now the supplied official Long Beach 2024 Top 16 ALL ACTION recording, `longbeach2024_action.mp4`. Footage selection and implementation are no longer planning tasks.

## Open the project

1. Double click `launch.cmd` in the project folder.
2. Open `http://127.0.0.1:8510` if the browser does not open automatically.
3. Watch the Complete run tab, covering 26.7 seconds from launch through the visible finish across four camera shots.
4. Jump between shots and inspect the reviewed role intervals against visible car livery.
5. Review the separation curve together with missing or unassigned observation markers.
6. Export the complete replay, timeline, shot metadata or summary.

The source span produces a 26.8 second sampled replay with 268 frames. The twelve shorter diagnostic examples remain in Shot review. Role intervals in the original demonstration are AI visual review decisions without human expert validation. In **Review roles for a camera shot**, choose **Bounded intervals** to correct specific times, retain a single known role or clear the entire shot to unknown. Saving rebuilds the complete exports and written conclusions. **Entire camera shot** is an explicit alternative that replaces the selected shot's interval map after the whole view has been checked.

The launcher uses `G:\DriftLens\.venv` and starts only a local Streamlit server. Keep the source recording in this project folder. The pipeline does not use PitWall or banking project environments.

## Process another interval

Use **Process a clip → Analyze your own clip → Upload my video** to select a private MP4, MOV, MKV, AVI or WebM of at most 200 MB, or choose a saved local source. Set start and end times within a positive interval of at most 60 seconds, inspect the selected **Analysis profile**, and use a new result name. Choose one continuous camera shot and inspect both cars near its start. Begin with a short interval on this CPU; the measured speed, rather than the sampling setting, determines waiting time.

Open the saved result in **Shot review** to inspect the annotated replay, written conclusions, review events, exact sample images and exports. In **Correct roles within bounded clip intervals**, supply visual lead/chase decisions only where supported. Each assigned track ID must occur inside its interval, and assigned intervals must stop at detected camera cuts. Empty IDs and uncovered time remain unknown. **Save clip intervals and rebuild exports** refreshes all derived outputs. Processing your video does not itself establish accurate identities, physical telemetry or generalisation to new footage.

The shot catalogue contains 12 visually selected intervals totalling 53.4 seconds, split into six tuning and six test shots. Whole battle groups remain within one split. Use an existing result name with the other tracker to compare the same source interval. Review each tracker result's roles because ID numbers can differ.

## Read the current evidence

The Evidence tab shows the saved tracker comparison and annotation method. `outputs/evaluation_report.json` records the original 416 pixel diagnostic evaluation. `outputs/finetuning_report.json` records the completed twelve epoch CPU adaptation experiment evaluated separately at 640 pixels with class agnostic suppression. Its candidate regressed on frozen test precision and recall, so the original detector remains active. `docs/RESULTS.md` explains both comparisons, their counts and failures.

The reference set contains 24 sparse test frames with 48 approximate vehicle boxes and four tuning frames with eight boxes. An AI assistant drafted the labels from visual source inspection; no human expert has reviewed them. Treat these numbers as a small diagnostic evaluation. They do not establish a full tracking benchmark or general accuracy across Formula Drift events.

## Strengthen the portfolio

1. Open **Evidence → Review reference labels for future training** to inspect source frames and correct boxes, visibility, classes and shot local identities. A named reviewer must explicitly confirm the image, visible boxes and shot local identities; the delivered queue contains zero human confirmations. See `docs/EVALUATION.md` and `tools/annotation_review.py` for validated imports and exports. New uploaded clip samples are not automatically added to this original queue.
2. Export confirmed samples to a new file under `outputs/annotation_review`. Keep the frozen historical labels and scores. Add denser source labels in a separate reference set so identity quality inside currently unlabelled intervals can be assessed.
3. Add a different event or camera style while preserving complete battle groups in each dataset split.
4. Repeat the same detector and tracker settings on the expanded held out set, reporting sample counts and failures with the rates. Do not launch another training experiment from the existing small assistant labels merely to improve the displayed score.
5. Publish project code, methods and numerical summaries separately from footage. Raw recordings and video derivatives remain private while reuse permission is unverified.

The image separation chart does not measure physical distance, speed or drift angle and does not provide official judging scores. Detector confidence does not establish correct identity.

## Computer and maintenance notes

The inspected computer has an Intel Core i3 2350M, approximately 11.9 GB installed memory and Intel HD Graphics 3000. Analysis uses the CPU. Consult each run summary for measured processing speed; a 10 fps sampling setting does not promise 10 fps throughput.

Use the independent environment and the versions recorded by the project. Initial setup downloads free packages and model weights; normal analysis then runs locally. Keep unsuccessful examples so future changes can be assessed against actual errors.

Every README must contain no ASCII hyphens or Unicode dash punctuation. Continue validating that rule when updating documentation.
