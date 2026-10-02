# Test your video and record DriftLens

Updated on 3 October 2026.

DriftLens V1 runs locally with free tools and its own Python environment. It produces an annotated replay and observed measurements for a reviewer to check. Processing your video is a test of that video, not evidence that the system works accurately on every event.

## Open the dashboard

1. Open `G:\DriftLens` in File Explorer and double click `launch.cmd`.
2. Keep the terminal open. In Edge or Chrome, open [the local dashboard](http://127.0.0.1:8510).
3. If the launcher says its environment is missing, run `setup.cmd` from the same folder and then launch again. The current environment is already installed; setup only needs to be repeated if it is absent. Initial setup downloads free dependencies and model weights.
4. To stop the app when finished, close the launcher terminal or press Ctrl+C in it.

## Upload your first test

1. Use an existing video you own or have permission to use. For a public demonstration, also confirm permission to publish the footage. Prefer a clear continuous view with both participating cars visible.
2. Open **Process a clip → Analyze your own clip → Upload my video**. Select an MP4, MOV, MKV, AVI or WebM of at most 200 MB.
3. Start with ten to fifteen seconds. Set **Start seconds** and **End seconds** inside the file. Intervals must be longer than zero and no longer than 60 seconds. A long file can be uploaded within the file limit while only a bounded interval is analysed.
4. Keep **Enhanced observed car recovery** for an initial comparison with the prepared demonstration. It uses the CPU and can be slow. The **Original 416 pixel diagnostic** profile is a simpler alternative, with different settings and without experimental appearance recovery; its results need their own review.
5. Enter a new **Result name**, such as `my_first_test`, and select **Process clip locally**. Keep the terminal open until processing completes.
6. Open the saved result in **Shot review**. Watch the replay and check whether boxes represent the participants, background vehicles, overlaps or false detections. A single car clip can test detection, but tandem measurements require two observed cars and accepted roles.
7. Open **Correct roles within bounded clip intervals**. Check visible livery and travel order, then assign different observed IDs to lead and chase only within supported intervals. Make separate decisions for each camera view. Unknown IDs and uncovered times stay empty; any recovered class marked R needs inspection.
8. Select **Save clip intervals and rebuild exports**. Read the chart and report against the replay. A missing participant or uncertain role should appear as an unavailable measurement.
9. Export the readable report and CSV or JSON results. The private upload is under `G:\DriftLens\data\uploads` and its result under `G:\DriftLens\outputs\runs`.

Inspect camera boundaries against the actual video. The cut safeguard is a heuristic; it may miss or add a boundary on unfamiliar footage. If a view change is not represented correctly, use a shorter continuous source interval for the first test and keep the failure as evidence. Tracker IDs are local observations, not guaranteed physical identity through a cut. Role review does not repair missed detections or turn assistant decisions into expert labels.

The original 28 second uploaded test took 611.995 seconds of CPU processing before export and visual review, about ten minutes, under its recorded workload and settings. Your wait can differ. Complete processing and role review before recording, and say the demonstration shows a prepared result. A 10 fps sampling setting does not mean real time analysis.

## Record the finished app

First open the completed result in a normal browser and test a short recording. Close unrelated windows and notifications, make the text readable and keep the replay, chart and report ready. Use an existing authorised clip for the walkthrough.

If Xbox Game Bar is available and can record the browser on this computer, press **Windows+G** to open its controls, then **Windows+Alt+R** to start or stop. **Windows+Alt+M** toggles the microphone. Recordings are MP4 files under **Videos → Captures**. These controls and the save location are documented by [Microsoft Support](https://support.microsoft.com/en-us/accessibility/windows/use-a-screen-reader-to-record-your-screen-with-xbox-game-bar). Availability and capture support depend on the computer, so verify the trial before your full take.

If you already have a compatible OBS Studio installation, use its **Auto Configuration Wizard**, add a **Window Capture** source for the browser, verify the audio meters and select **Start Recording**. Review a short trial for readable text, smooth replay and audible narration. [The official OBS guide](https://obsproject.com/kb/quick-start-guide) describes these steps.

Suggested 45 to 60 second walkthrough:

1. **0 to 8 seconds:** Introduce DriftLens as a local computer vision project for review assisted tandem analysis. Show the completed result name.
2. **8 to 25 seconds:** Play the annotated clip. Point out boxes, local IDs and reviewed lead and chase roles.
3. **25 to 40 seconds:** Show the image separation chart and one missing or uncertain interval. Explain that unavailable observations remain gaps.
4. **40 to 50 seconds:** Show the readable report and numerical export controls.
5. **50 to 60 seconds:** State that processing happens offline, roles require review and independent evaluation on more events is future work.

Optional narration:

> I built DriftLens to review tandem drifting videos with local computer vision. After offline processing, I check the observed cars and assign roles within each camera view. The replay and image separation chart make both accepted observations and missing measurements visible. I can export the report and data for review. The current version needs manual role checks, and broader event validation is the next step.

## Footage for LinkedIn

The supplied official Formula Drift broadcast has unverified redistribution permission. Keep that source, screenshots, thumbnails, replays and other visual derivatives private. For a public demo, use footage you own or are authorised to publish. If that is unavailable, show code, numerical charts, methods and dashboard sections without broadcast images. Keep these rights separate from the project code licence.

Paste ready post text:

> I built DriftLens, a local computer vision tool for reviewing tandem drifting videos.
>
> It combines vehicle detection and tracking with replay review, manual lead and chase corrections, an image separation timeline and downloadable reports. Missing or uncertain observations remain explicit gaps.
>
> Built with Python, OpenCV, YOLO, BoTSORT and Streamlit. The current V1 supports private video uploads and offline processing. It needs visual role review, and evaluation across more events is future work.
>
> #ComputerVision #Python #MachineLearning #PortfolioProject

Only describe your own footage as tested once you have processed and inspected it. The saved 71.3 percent demo and 61.8 percent first uploaded test are reviewed pair availability on selected examples, not generic accuracy figures. Do not present either as a model accuracy score in a post.
