# Test your video and record DriftLens

Updated on 3 October 2026.

DriftLens V1 runs locally with free tools and its own Python environment. It produces an annotated replay and observed measurements for a reviewer to check. Processing your video is a test of that video, not evidence that the system works accurately on every event.

## Open the dashboard

1. Open `G:\DriftLens` in File Explorer and double click `launch.cmd`.
2. Keep the terminal open. In Edge or Chrome, open [the local dashboard](http://127.0.0.1:8510).
3. If the launcher says its environment is missing, run `setup.cmd` from the same folder and then launch again. The current environment is already installed; setup only needs to be repeated if it is absent. Initial setup downloads free dependencies and model weights.
4. To stop the app when finished, close the launcher terminal or press Ctrl+C in it.

## Launch manually in PowerShell

Open PowerShell and run:

```powershell
cd G:\DriftLens
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8510 --browser.gatherUsageStats false
```

This uses the dedicated project environment without activating it. Keep PowerShell open while using the dashboard and press **Ctrl+C** to stop it. If another DriftLens session occupies port 8510, stop that session or choose `--server.port 8511` and open `http://127.0.0.1:8511`.

## Upload your first test

1. Use an existing video you own or have permission to use. For a public demonstration, also confirm permission to publish the footage. Prefer a clear continuous view with both participating cars visible.
2. Open **Process a clip → Analyze your own clip → Upload my video**. Select an MP4, MOV, MKV, AVI or WebM of at most 200 MB.
3. Start with ten to fifteen seconds. Set **Start seconds** and **End seconds** inside the file. Intervals must be longer than zero and no longer than 60 seconds. A long file can be uploaded within the file limit while only a bounded interval is analysed.
4. Keep **Enhanced observed car recovery** for an initial comparison with the prepared demonstration. It uses the CPU and can be slow. The **Original 416 pixel diagnostic** profile is a simpler alternative, with different settings and without experimental appearance recovery; its results need their own review.
5. Enter a new **Result name**, such as `my_first_test`, and select **Process clip locally**. Keep the terminal open until processing completes.
6. Open the saved result in **Shot review**. Watch the replay and check whether boxes represent the participants, background vehicles, overlaps or false detections. A single car clip can test detection, but tandem measurements require two observed cars and accepted roles.
7. Watch a clear initial interval. Check livery and travel order and note the different observed lead and chase IDs. A prior manual interval save is not required. Any recovered class marked R needs inspection.
8. Open **Follow lead and chase across views** and enter **Seed start seconds**, **Seed end seconds**, **Seed lead ID** and **Seed chase ID**. Confirm **I have checked lead and chase in the seed interval**, then select **Follow this pair through the clip**. Inspect the replay and matching evidence across camera changes and ID fragmentation. Later role labels display **MATCH** and are algorithmic, unverified assignments. Scores and margins are not identity probabilities.
9. The continuity action rebuilds the exports. Read the chart and report against the replay. Correct a poor match in **Correct roles within bounded clip intervals**, or use another reliable seed. Manual role edits override the automatic assignments and deactivate continuity status while retaining match diagnostics. A missing participant or uncertain role should appear as an unavailable measurement.
10. Export the readable report and CSV or JSON results. The private upload is under `G:\DriftLens\data\uploads` and its result under `G:\DriftLens\outputs\runs`.

Inspect camera boundaries against the actual video. The cut safeguard is a heuristic; it may miss or add a boundary on unfamiliar footage. Tracker IDs are local observations, not guaranteed physical identity through a cut. The continuity step uses the reviewed seed's appearance to propose roles for later observed IDs, including when an ID changes inside a view. It cannot recover an invisible car or prove an appearance match correct. Inspect unmatched and uncertain spans, and keep failures as evidence. Role review does not repair missed detections or turn automatic proposals into expert labels.

The original 28 second uploaded test took 611.995 seconds of CPU processing before export and visual review, about ten minutes, under its recorded workload and settings. Your wait can differ. Complete processing and role review before recording, and say the demonstration shows a prepared result. A 10 fps sampling setting does not mean real time analysis.

## Record the finished app

First open the completed result in a normal browser and test a short recording. Close unrelated windows and notifications, make the text readable and keep the replay, chart and report ready. Use an existing authorised clip for the walkthrough.

If Xbox Game Bar is available and can record the browser on this computer, press **Windows+G** to open its controls, then **Windows+Alt+R** to start or stop. **Windows+Alt+M** toggles the microphone. Recordings are MP4 files under **Videos → Captures**. These controls and the save location are documented by [Microsoft Support](https://support.microsoft.com/en-us/accessibility/windows/use-a-screen-reader-to-record-your-screen-with-xbox-game-bar). Availability and capture support depend on the computer, so verify the trial before your full take.

If you already have a compatible OBS Studio installation, use its **Auto Configuration Wizard**, add a **Window Capture** source for the browser, verify the audio meters and select **Start Recording**. Review a short trial for readable text, smooth replay and audible narration. [The official OBS guide](https://obsproject.com/kb/quick-start-guide) describes these steps.

Suggested 45 to 60 second walkthrough:

1. **0 to 8 seconds:** Introduce DriftLens as a local computer vision project for review assisted tandem analysis. Show the completed result name.
2. **8 to 25 seconds:** Play the annotated clip across a camera change. Show that a reviewed seed supports later automatic role proposals, and point out the local IDs and remaining unknowns.
3. **25 to 40 seconds:** Show the image separation chart and one missing or uncertain interval. Explain that unavailable observations remain gaps.
4. **40 to 50 seconds:** Show the readable report and numerical export controls.
5. **50 to 60 seconds:** State that processing happens offline, roles require review and independent evaluation on more events is future work.

Optional narration:

> I built DriftLens to review tandem drifting videos with local computer vision. After offline processing, I review a reliable lead and chase pair. The app then compares visible car appearance to propose later roles when the camera or local track ID changes. I inspect those matches, and uncertain or missing observations remain gaps. The replay, image separation chart and exported report support that review. Independent evaluation across more events is still future work.

## Footage for LinkedIn

The supplied official Formula Drift broadcast has unverified redistribution permission. Keep that source, screenshots, thumbnails, replays and other visual derivatives private. For a public demo, use footage you own or are authorised to publish. If that is unavailable, show code, numerical charts, methods and dashboard sections without broadcast images. Keep these rights separate from the project code licence.

Paste ready post text:

> I built DriftLens, a local computer vision tool for reviewing tandem drifting videos.
>
> It combines vehicle detection and tracking with a reviewed initial lead and chase pair, appearance based role continuity across camera changes, editable intervals, an image separation timeline and downloadable reports. Missing or uncertain observations remain explicit gaps.
>
> Built with Python, OpenCV, YOLO, BoTSORT and Streamlit. V1 supports private video uploads and offline processing. Appearance matches still need visual checks, and independent evaluation across more events is future work.
>
> #ComputerVision #Python #MachineLearning #PortfolioProject

Only describe your own footage as tested once you have processed and inspected it. The saved 71.3 percent demo and 61.8 percent first uploaded test are reviewed pair availability on selected examples, not generic accuracy figures. Do not present either as a model accuracy score in a post.
