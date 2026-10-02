"""Bounded manual role decisions for uploaded clips and fragmented tracks."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import shutil
import uuid

import pandas as pd
import streamlit as st

from .review import derive_frame_metrics, read_rows, validate_role_intervals, write_metrics


def _finite(value, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number.") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number.")
    return number


def _id(value) -> int | None:
    if value is None or str(value).strip().lower() in {"", "none", "null", "nan"}:
        return None
    number = _finite(value, "Track ID")
    if isinstance(value, bool) or number < 0 or not number.is_integer():
        raise ValueError("Role IDs must be observed nonnegative integers or empty.")
    return int(number)


def _observed(row: dict) -> bool:
    return str(row.get("observed", True)).strip().lower() in {"true", "1", "1.0", "yes"}


def _validated_inputs(summary: dict, frames: list[dict], observations: list[dict], intervals: list[dict]) -> list[dict]:
    if summary.get("status") != "complete":
        raise ValueError("Only a completed clip can receive reviewed roles.")
    duration = _finite(summary.get("duration_seconds"), "Clip duration")
    if duration <= 0 or not frames:
        raise ValueError("Role review needs a positive clip duration and saved sampled frames.")
    if summary.get("frame_count") is not None and _finite(summary["frame_count"], "Saved frame count") != len(frames):
        raise ValueError("Saved sampled frames disagree with the completed clip frame count.")
    if not isinstance(intervals, list):
        raise ValueError("Role intervals must be a list.")
    canonical = []
    for interval in intervals:
        if not isinstance(interval, dict):
            raise ValueError("Every role interval must contain start, end and role decisions.")
        canonical.append({"start_clip_seconds": _finite(interval.get("start_clip_seconds"), "Interval start"),
                          "end_clip_seconds": _finite(interval.get("end_clip_seconds"), "Interval end"),
                          "lead_id": _id(interval.get("lead_id")), "chase_id": _id(interval.get("chase_id")),
                          "review_basis": str(interval.get("review_basis") or "User reviewed interval; no expert validation.")})
    frame_map, cuts, previous = {}, [], None
    for frame in frames:
        index = _id(frame.get("frame_index"))
        shot = _id(frame.get("shot_index", 0))
        time = _finite(frame.get("clip_time"), "Frame clip time")
        source = _finite(frame.get("source_time"), "Frame source time")
        if index is None or index != len(frame_map) or shot is None or time < 0 or source < 0 or time >= duration + .001 or index in frame_map:
            raise ValueError("Saved frames must have unique indices and timestamps inside the clip.")
        if previous and (index <= previous[0] or time <= previous[1] or source <= previous[3]):
            raise ValueError("Saved sampled frames must be strictly ordered.")
        if previous and shot != previous[2]:
            cuts.append(time)
        frame_map[index] = (time, shot)
        previous = index, time, shot, source
    available, visible = set(), []
    for observation in observations:
        if not _observed(observation):
            continue
        identity = _id(observation.get("track_id"))
        index = _id(observation.get("frame_index"))
        if identity is None:
            continue
        if index not in frame_map:
            raise ValueError("A tracked observation does not belong to a saved frame.")
        frame_time, frame_shot = frame_map[index]
        if observation.get("clip_time") is not None and abs(_finite(observation["clip_time"], "Observation time") - frame_time) > .001:
            raise ValueError("An observation timestamp disagrees with its saved sampled frame.")
        if _id(observation.get("shot_index", 0)) != frame_shot:
            raise ValueError("An observation belongs to a different camera view than its saved frame.")
        available.add(identity)
        visible.append((identity, frame_time))
    validate_role_intervals(canonical, duration, available)
    for interval in canonical:
        start, end = interval["start_clip_seconds"], interval["end_clip_seconds"]
        if end > duration:
            raise ValueError("Role intervals must end inside the clip duration.")
        roles = [identity for identity in (interval["lead_id"], interval["chase_id"]) if identity is not None]
        if roles and any(start < cut < end for cut in cuts):
            raise ValueError("Split assigned role intervals at each camera cut. IDs and roles stay local to one view.")
        for identity in roles:
            if not any(observed_id == identity and start <= time < end for observed_id, time in visible):
                raise ValueError(f"Track ID {identity} was not observed within interval {start:.3f}s to {end:.3f}s.")
    return canonical


def update_clip_role_intervals(run_dir: Path, intervals: list[dict]) -> dict:
    """Rebuild every derived clip export from observed boxes, with rollback."""
    from .pipeline import render_run
    from .run_analysis_ui import save_shot_analysis

    run_dir = Path(run_dir).resolve()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8-sig"))
    frames, observations = read_rows(run_dir / "frames.csv"), read_rows(run_dir / "observations.csv")
    canonical = _validated_inputs(summary, frames, observations, intervals)
    metrics = derive_frame_metrics(observations, frames, None, None, canonical)
    revised = copy.deepcopy(summary)
    first_pair = next((interval for interval in canonical if interval["lead_id"] is not None and interval["chase_id"] is not None), {})
    revised.update(lead_id=first_pair.get("lead_id"), chase_id=first_pair.get("chase_id"),
                   role_intervals=canonical, role_review_status="user_assignment_unverified",
                   role_assignment_method="User assigned bounded local track intervals after inspecting the clip; no human expert validation or association across cuts is asserted.",
                   paired_frames=sum(row["pair_observed"] for row in metrics),
                   pair_coverage=sum(row["pair_observed"] for row in metrics) / len(metrics))
    revised["files"] = {**revised.get("files", {}), "video": "annotated.mp4", "metrics": "frame_metrics.csv",
                        "analysis": "analysis.json", "report": "report.txt", "summary": "summary.json"}
    stage = (run_dir / f".clip_roles_{uuid.uuid4().hex}").resolve()
    stage.relative_to(run_dir)
    stage.mkdir()
    names = ("annotated.mp4", "frame_metrics.csv", "analysis.json", "report.txt", "summary.json")
    preserve_stage = False
    try:
        # Reports inspect the same current box evidence as the renderer. Source
        # paths and timestamps stay in the original metadata, not this stage.
        for name in ("frames.csv", "observations.csv", "detections.csv"):
            original = (run_dir / name).resolve()
            original.relative_to(run_dir)
            if original.is_file():
                shutil.copyfile(original, stage / name)
        write_metrics(stage / "frame_metrics.csv", metrics)
        render_run(stage, revised, observations, frames)
        revised = save_shot_analysis(stage, revised)
        prior = stage / "prior"
        prior.mkdir()
        existing, published = set(), []
        for name in names:
            original = (run_dir / name).resolve()
            original.relative_to(run_dir)
            if original.is_file():
                shutil.copyfile(original, prior / name)
                existing.add(name)
        try:
            for name in names:
                (stage / name).replace(run_dir / name)
                published.append(name)
        except Exception:
            restore_errors = []
            for name in published:
                try:
                    if name in existing:
                        shutil.copyfile(prior / name, run_dir / name)
                    else:
                        (run_dir / name).unlink()
                except OSError as error:
                    restore_errors.append(str(error))
            if restore_errors:
                preserve_stage = True
                raise RuntimeError(f"Publication failed and some exports could not be restored. Original backups are preserved in {prior}: {'; '.join(restore_errors)}")
            raise
    finally:
        if not preserve_stage:
            resolved = stage.resolve()
            resolved.relative_to(run_dir)
            if resolved.parent != run_dir or not resolved.name.startswith(".clip_roles_"):
                raise ValueError("Clip role staging cleanup escaped the result directory.")
            shutil.rmtree(resolved)
    return revised


def render_clip_role_editor(run_dir: Path, summary: dict, observations) -> None:
    """Offer an explicitly saved interval map for one existing clip result."""
    run_dir = Path(run_dir).resolve()
    duration = _finite(summary.get("duration_seconds", 0), "Clip duration")
    if duration <= 0:
        return
    try:
        frames = read_rows(run_dir / "frames.csv")
    except (OSError, ValueError):
        st.info("Saved sampled frames are needed for bounded role editing.")
        return
    raw_rows = observations.to_dict("records") if hasattr(observations, "to_dict") else list(observations)
    ids = sorted({identity for row in raw_rows if _observed(row) and (identity := _id(row.get("track_id"))) is not None})
    cuts = [float(row["clip_time"]) for previous, row in zip(frames, frames[1:])
            if str(previous.get("shot_index", 0)) != str(row.get("shot_index", 0))]
    with st.expander("Correct roles within bounded clip intervals", expanded=False):
        st.caption("Use visual livery and travel order to select lead and chase. Tracker numbers alone do not establish roles or associate cars across cuts.")
        st.caption(f"Clip duration {duration:.3f}s. Available observed IDs: {', '.join(map(str, ids)) or 'none'}. Start is included and end is excluded. Each assigned ID must occur inside its interval; uncovered time stays unknown.")
        if cuts:
            st.caption("Detected camera cuts within clip: " + ", ".join(f"{time:.3f}s" for time in cuts) + ". Split assigned intervals at these boundaries and review roles independently.")
        intervals = summary.get("role_intervals")
        if intervals is None:
            intervals = [{"start_clip_seconds": 0.0, "end_clip_seconds": duration,
                          "lead_id": summary.get("lead_id"), "chase_id": summary.get("chase_id"), "review_basis": ""}] if not cuts else [
                              {"start_clip_seconds": start, "end_clip_seconds": end, "lead_id": None, "chase_id": None, "review_basis": ""}
                              for start, end in zip([0.0] + cuts, cuts + [duration])]
        initial = pd.DataFrame([{"Start within clip (s)": row.get("start_clip_seconds"), "End within clip (s)": row.get("end_clip_seconds"),
                                 "Lead ID": row.get("lead_id"), "Chase ID": row.get("chase_id"), "Review basis": row.get("review_basis", "")}
                                for row in intervals], columns=["Start within clip (s)", "End within clip (s)", "Lead ID", "Chase ID", "Review basis"])
        for name in ("Start within clip (s)", "End within clip (s)", "Lead ID", "Chase ID"):
            initial[name] = pd.to_numeric(initial[name], errors="coerce").astype(float)
        with st.form(f"clip_interval_form_{run_dir}"):
            edited = st.data_editor(initial, hide_index=True, num_rows="dynamic", width="stretch", key=f"clip_interval_table_{run_dir}",
                                    column_config={
                                        "Start within clip (s)": st.column_config.NumberColumn(min_value=0.0, max_value=duration, step=.01, format="%.3f", required=True),
                                        "End within clip (s)": st.column_config.NumberColumn(min_value=0.0, max_value=duration, step=.01, format="%.3f", required=True),
                                        "Lead ID": st.column_config.NumberColumn(min_value=0, step=1, format="%d"),
                                        "Chase ID": st.column_config.NumberColumn(min_value=0, step=1, format="%d"),
                                        "Review basis": st.column_config.TextColumn(help="Record visible livery and travel order, or the reason a role remains unknown."),
                                    })
            clear = st.checkbox("Mark all clip roles unknown", key=f"clip_interval_clear_{run_dir}")
            submitted = st.form_submit_button("Save clip intervals and rebuild exports", type="primary", width="stretch")
        st.caption("A saved correction is an unverified user assignment. The replay, frame metrics, readable report and analysis JSON are rebuilt together; failed publication restores the saved exports.")
        if not submitted:
            return
        try:
            revised = [] if clear else [{"start_clip_seconds": row["Start within clip (s)"],
                                        "end_clip_seconds": row["End within clip (s)"],
                                        "lead_id": None if pd.isna(row["Lead ID"]) else row["Lead ID"],
                                        "chase_id": None if pd.isna(row["Chase ID"]) else row["Chase ID"],
                                        "review_basis": row.get("Review basis", "")}
                                       for row in edited.to_dict("records")]
            with st.spinner("Rebuilding clip roles, replay and every report export…"):
                update_clip_role_intervals(run_dir, revised)
            st.session_state["notice"] = "Bounded clip roles saved as an unverified user assignment. Replay, metrics and reports were rebuilt."
            st.rerun()
        except Exception as error:
            st.error(f"Could not save bounded clip roles: {error}")
