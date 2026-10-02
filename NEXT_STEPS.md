# DriftLens next steps

Updated on 2 October 2026.

The offline CPU pipeline and local review interface have been implemented in `G:\DriftLens`. The footage source is now the supplied official Long Beach 2024 Top 16 ALL ACTION recording, `longbeach2024_action.mp4`. Footage selection and implementation are no longer planning tasks.

## Open the project

1. Double click `launch.cmd` in the project folder.
2. Open `http://127.0.0.1:8510` if the browser does not open automatically.
3. Choose a prepared shot from the library and inspect its numbered replay.
4. Check lead and chase assignments against visible travel order.
5. Review the separation curve together with red missing observation markers.
6. Export the video, observations, frame metrics or summary from the review page.

The launcher uses `G:\DriftLens\.venv` and starts only a local Streamlit server. Keep the source recording in this project folder. The pipeline does not use PitWall or banking project environments.

## Process another interval

Use the Process a clip tab to select the local source, start and end times, tracker and output name. Choose one continuous camera shot and inspect both cars near its start. Begin with short intervals, 416 pixel inference size and 10 sampled frames per second on this computer. The pipeline accepts at most 60 seconds per job.

The shot catalogue contains 12 visually selected intervals totalling 53.4 seconds, split into six tuning and six test shots. Whole battle groups remain within one split. Use an existing result name with the other tracker to compare the same source interval. Review each tracker result's roles because ID numbers can differ.

## Read the current evidence

The Evidence tab shows the saved tracker comparison and annotation method. `outputs/evaluation_report.json` records diagnostic evaluation and `docs/RESULTS.md` explains the measured results and failures when generated.

The reference set contains 24 sparse test frames with 48 approximate vehicle boxes and four tuning frames with eight boxes. An AI assistant drafted the labels from visual source inspection; no human expert has reviewed them. Treat these numbers as a small diagnostic evaluation. They do not establish a full tracking benchmark or general accuracy across Formula Drift events.

## Strengthen the portfolio

1. Ask a human reviewer to check the reference boxes, visibility labels and car identities, especially during smoke and overlap.
2. Add denser labels so identity quality inside currently unlabelled intervals can be assessed.
3. Add a different event or camera style while preserving complete battle groups in each dataset split.
4. Repeat the same detector and tracker settings on the expanded held out set, reporting sample counts and failures with the rates.
5. Publish project code, methods and numerical summaries separately from footage. Raw recordings and video derivatives remain private while reuse permission is unverified.

The image separation chart does not measure physical distance, speed or drift angle and does not provide official judging scores. Detector confidence does not establish correct identity.

## Computer and maintenance notes

The inspected computer has an Intel Core i3 2350M, approximately 11.9 GB installed memory and Intel HD Graphics 3000. Analysis uses the CPU. Consult each run summary for measured processing speed; a 10 fps sampling setting does not promise 10 fps throughput.

Use the independent environment and the versions recorded by the project. Initial setup downloads free packages and model weights; normal analysis then runs locally. Keep unsuccessful examples so future changes can be assessed against actual errors.

Every README must contain no ASCII hyphens or Unicode dash punctuation. Continue validating that rule when updating documentation.
