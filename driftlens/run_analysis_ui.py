"""Run conclusions, review events and exact annotated sample inspection."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import uuid

import pandas as pd
import streamlit as st

from .run_analysis import build_run_analysis, load_evidence, playback_seek, report_payloads


STATUS_LABELS = {
    "pair_observed": "Paired image measurement accepted",
    "role_unknown": "Role unknown",
    "lead_track_missing": "Assigned lead ID not observed",
    "chase_track_missing": "Assigned chase ID not observed",
    "selected_tracks_missing": "Both assigned IDs not observed",
    "measurement_unavailable": "Pair geometry unavailable",
    "no_detector_candidates": "No selected vehicle candidate boxes",
    "detections_without_track_ids": "Selected candidate boxes, no tracked IDs",
    "tracked_observations_present": "Tracked observations present",
    "no_tracked_observations_detector_unknown": "No tracked IDs; detector export unavailable",
    "evidence_unavailable": "Detector/tracker exports unavailable",
}


def render_run_analysis(root: Path, directory: Path, summary: dict, metrics,
                        shots: list[dict] | None = None) -> dict | None:
    """Show reproducible conclusions and return a selected event for playback."""
    try:
        evidence = load_evidence(root, directory, shots)
        if not shots:
            shot_id = str(summary.get("clip_id", summary.get("run_id", directory.name)))
            evidence = {shot_id: evidence.get(directory.name, {})}
        report = build_run_analysis(summary, metrics, shots=shots, evidence_by_shot=evidence)
    except (OSError, ValueError, TypeError, KeyError) as error:
        st.info(f"Run conclusions are unavailable for this saved timeline: {error}")
        return None
    key = str(directory.resolve())
    st.markdown("**Measured run conclusions**")
    for sentence in report["conclusions"]:
        st.write(sentence)
    counts = report["measurement_status_counts"]
    columns = st.columns(3)
    columns[0].metric("Accepted pair samples", f"{report['accepted_paired_samples']} / {report['sample_count']}")
    columns[1].metric("Role unknown samples", counts.get("role_unknown", 0))
    missing = sum(counts.get(name, 0) for name in ("lead_track_missing", "chase_track_missing", "selected_tracks_missing"))
    columns[2].metric("Assigned ID missing samples", missing)
    st.caption("Role uncertainty, absent assigned IDs and detector candidate counts are separate observations. None establishes a hidden car's position or verified tracking accuracy.")
    with st.expander("Sample status and unknown intervals"):
        if report["frame_statuses"]:
            frames = pd.DataFrame(report["frame_statuses"])[["frame_index", "playback_time", "source_time", "shot_id", "lead_status", "chase_status", "measurement_status", "detector_candidate_count", "tracked_observation_count", "recovered_selected_box_count", "recovered_tracked_box_count", "pair_uses_recovered_box", "observation_status"]]
            frames["measurement_status"] = frames["measurement_status"].map(STATUS_LABELS)
            frames["observation_status"] = frames["observation_status"].map(STATUS_LABELS)
            st.dataframe(frames, hide_index=True, width="stretch", column_config={
                "detector_candidate_count": st.column_config.NumberColumn("Selected input boxes"),
                "tracked_observation_count": st.column_config.NumberColumn("Current tracked boxes"),
                "recovered_selected_box_count": st.column_config.NumberColumn("Recovered input boxes"),
                "recovered_tracked_box_count": st.column_config.NumberColumn("Recovered tracked boxes"),
                "pair_uses_recovered_box": st.column_config.CheckboxColumn("Pair includes recovery"),
            })
        if report["box_provenance"]["recovered_source_classes"]:
            st.markdown("**Original predicted classes of recovered boxes**")
            st.dataframe(pd.DataFrame(report["box_provenance"]["recovered_source_classes"]), hide_index=True, width="stretch")
            st.caption("These are the original detector classes. Recovery reuses a current observed box; its score remains confidence in that source class rather than car confidence.")
        if report["unknown_intervals"]:
            intervals = pd.DataFrame(report["unknown_intervals"])[["shot_id", "local_shot_index", "start_playback_seconds", "end_playback_seconds", "start_source_seconds", "last_sample_source_seconds", "sample_count", "sampled_duration_seconds"]]
            st.dataframe(intervals, hide_index=True, width="stretch")
        st.caption("Unknown spans stop at camera cuts, role mapping changes and unsampled gaps. Sampled durations describe replay frames; source timestamps remain separate.")
    payloads = report_payloads(report)
    for column, (name, payload) in zip(st.columns(2), payloads.items()):
        column.download_button("Download readable report" if name == "report.txt" else "Download analysis JSON",
                               payload, file_name=f"{report['run_id']}_{name}",
                               mime="text/plain" if name.endswith(".txt") else "application/json",
                               key=f"analysis_download_{key}_{name}", width="stretch")
    events = report["events"]
    if not events:
        return None
    st.markdown("**Replay review events**")
    event_rows = [{"Replay seconds": item["playback_time"], "Source seconds": item["source_time"],
                   "Frame": item["frame_index"], "Shot": item["shot_id"], "Event": item["label"]}
                  for item in events]
    with st.expander(f"All {len(events)} timestamped review events"):
        st.dataframe(pd.DataFrame(event_rows), hide_index=True, width="stretch")
    selection_key = f"analysis_event_{key}"
    options = [None] + list(range(len(events)))
    if st.session_state.get(selection_key) not in options:
        st.session_state[selection_key] = None
    selected = st.selectbox("Jump to a review event", options, key=selection_key,
                            format_func=lambda value: "Use the camera shot or clip playback selection" if value is None else f"{events[value]['playback_time']:.2f}s · {events[value]['shot_id']} · {events[value]['label']}")
    return events[selected] if selected is not None else None


@st.cache_data(show_spinner=False, max_entries=8)
def _annotated_frame(path: str, modified_ns: int, frame_index: int):
    import cv2
    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            return None
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        # Reject an inaccurate decoder seek instead of labelling a different
        # sample as exact. Annotated replay frame numbers start at zero.
        if abs(capture.get(cv2.CAP_PROP_POS_FRAMES) - frame_index) > .5:
            return None
        success, frame = capture.read()
        if not success or abs(capture.get(cv2.CAP_PROP_POS_FRAMES) - (frame_index + 1)) > .5:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def render_review_player(root: Path, video: Path, start_seconds: float = 0,
                         event: dict | None = None) -> None:
    """Native replay with an honest whole-second seek and exact sample still."""
    video = Path(video).resolve()
    video.relative_to(Path(root).resolve())
    seek = playback_seek(event["playback_time"] if event else start_seconds)
    st.video(str(video), start_time=seek["player_start_seconds"])
    if event:
        st.caption(f"Selected event: replay {seek['target_seconds']:.2f}s · source {event['source_time']:.2f}s · annotated frame {event['frame_index']}. The native player starts at {seek['player_start_seconds']}s ({seek['preroll_seconds']:.2f}s before the event).")
        with st.expander("Inspect the exact annotated event frame", expanded=True):
            try:
                frame = _annotated_frame(str(video), video.stat().st_mtime_ns, int(event["frame_index"]))
            except (OSError, ImportError, ValueError):
                frame = None
            if frame is None:
                st.info("The exact annotated sample could not be decoded. Use the displayed event time in the replay.")
            else:
                st.image(frame, caption=f"Annotated sample {event['frame_index']} · replay {event['playback_time']:.2f}s · source {event['source_time']:.2f}s", width="stretch")
    else:
        st.caption(f"Native player start: {seek['player_start_seconds']}s. Requested offset {seek['target_seconds']:.2f}s; whole-second playback can include up to one second before the selected shot.")


def _shot_report(run_dir: Path, summary: dict) -> dict:
    run_dir = Path(run_dir).resolve()
    with (run_dir / "frame_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        metrics = list(csv.DictReader(handle))
    evidence = load_evidence(run_dir, run_dir)
    shot_id = str(summary.get("clip_id", run_dir.name))
    return build_run_analysis(summary, metrics, evidence_by_shot={shot_id: evidence.get(run_dir.name, {})})


def save_shot_analysis(run_dir: Path, summary: dict) -> dict:
    """Publish readable/JSON reports with rollback; do not rerun inference."""
    run_dir = Path(run_dir).resolve()
    report = _shot_report(run_dir, summary)
    revised = dict(summary)
    revised["files"] = {**summary.get("files", {}), "analysis": "analysis.json", "report": "report.txt"}
    payloads = {**report_payloads(report), "summary.json": (json.dumps(revised, indent=2, allow_nan=False) + "\n").encode("utf-8")}
    stage = run_dir / f".analysis_{uuid.uuid4().hex}"
    stage.mkdir()
    existing, published = {}, []
    try:
        for name, content in payloads.items():
            destination = run_dir / name
            if destination.is_file():
                shutil.copyfile(destination, stage / f"prior_{name}")
                existing[name] = stage / f"prior_{name}"
            (stage / name).write_bytes(content)
        try:
            for name in payloads:
                (stage / name).replace(run_dir / name)
                published.append(name)
        except Exception:
            for name in published:
                if name in existing:
                    shutil.copyfile(existing[name], run_dir / name)
                else:
                    (run_dir / name).unlink()
            raise
    finally:
        resolved = stage.resolve()
        resolved.relative_to(run_dir)
        if resolved.parent != run_dir or not resolved.name.startswith(".analysis_"):
            raise ValueError("Analysis staging cleanup escaped the shot directory.")
        shutil.rmtree(resolved)
    return revised


def render_shot_analysis(run_dir: Path, summary: dict) -> None:
    """Render an existing clip report and offer replay navigation on demand."""
    run_dir = Path(run_dir).resolve()
    try:
        with (run_dir / "frame_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
            metrics = list(csv.DictReader(handle))
    except (OSError, ValueError, csv.Error):
        st.info("Run conclusions require the saved frame metrics export.")
        return
    event = render_run_analysis(run_dir, run_dir, summary, metrics)
    video = (run_dir / str(summary.get("files", {}).get("video", "annotated.mp4"))).resolve()
    try:
        video.relative_to(run_dir)
    except ValueError:
        return
    if event is not None and video.is_file():
        render_review_player(run_dir, video, event=event)
