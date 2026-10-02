"""Review an assembled complete run while preserving camera shot boundaries."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from .run_analysis_ui import render_run_analysis, render_review_player

ACCENT = "#d7ff60"


def _number(value: Any, default: float = 0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _clock(value: Any) -> str:
    seconds = max(0, _number(value))
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


def _path(root: Path, value: str | Path, base: Path | None = None) -> Path | None:
    try:
        candidate = Path(value)
        resolved = (candidate if candidate.is_absolute() else (base or root) / candidate).resolve()
        resolved.relative_to(root)
        return resolved
    except (OSError, ValueError, TypeError):
        return None


def _json(root: Path, value: Path) -> dict:
    safe = _path(root, value)
    if safe is None:
        return {}
    try:
        result = json.loads(safe.read_text(encoding="utf-8-sig"))
        return result if isinstance(result, dict) else {}
    except (OSError, ValueError):
        return {}


def _file(root: Path, directory: Path, summary: dict, key: str, fallback: str) -> Path | None:
    value = (summary.get("files") or {}).get(key, fallback)
    safe = _path(root, value, directory)
    return safe if safe and safe.is_file() else None


def _bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(str).str.lower().isin(["true", "1", "1.0", "yes"])


def _role_editor(root: Path, directory: Path, summary: dict, shot_options: dict[str, dict], jump: str | None) -> None:
    if not shot_options:
        return
    run_id = str(summary["run_id"])
    selection_key = f"full_role_shot_{run_id}"
    previous_jump_key = f"full_role_previous_jump_{run_id}"
    if jump is not None and st.session_state.get(previous_jump_key) != jump:
        st.session_state[selection_key] = jump
    if st.session_state.get(selection_key) not in shot_options:
        st.session_state[selection_key] = jump if jump in shot_options else next(iter(shot_options))
    st.session_state[previous_jump_key] = jump
    with st.expander("Review roles for a camera shot", expanded=False):
        selected = st.selectbox("Camera shot to review", list(shot_options), key=selection_key,
                                format_func=lambda value: str(shot_options[value].get("label", value)))
        shot = shot_options[selected]
        child = _path(root, directory / "shots" / selected)
        if child is None or not child.is_dir():
            st.info("The saved tracking result for this camera shot is unavailable.")
            return
        child_summary = _json(root, child / "summary.json")
        intervals = child_summary.get("role_intervals", shot.get("role_intervals", [])) or []
        if intervals:
            st.markdown("**Current reviewed role intervals**")
            interval_rows = [{"Start within shot (s)": interval.get("start_clip_seconds"),
                              "End within shot (s)": interval.get("end_clip_seconds"),
                              "Lead ID": interval.get("lead_id"), "Chase ID": interval.get("chase_id"),
                              "Review basis": interval.get("review_basis", "Not recorded")}
                             for interval in intervals]
            st.dataframe(pd.DataFrame(interval_rows), hide_index=True, width="stretch")
            st.caption("Intervals include their start and exclude their end. Unreviewed time outside the mapping remains unknown.")
        ids = set()
        for value in child_summary.get("track_ids", []):
            try:
                numeric = float(value)
                if math.isfinite(numeric) and numeric >= 0 and numeric.is_integer():
                    ids.add(int(numeric))
            except (ValueError, TypeError):
                continue
        if len(ids) < 2:
            st.info("Fewer than two tracked IDs are available for this shot. A paired role assignment is unavailable; intervals can still preserve one known role or explicitly unknown time.")
        if shot.get("lead_description") or shot.get("chase_description"):
            lead_reference, chase_reference = st.columns(2)
            lead_reference.caption(f"Lead reference: {shot.get('lead_description') or 'No livery description saved'}")
            chase_reference.caption(f"Chase reference: {shot.get('chase_description') or 'No livery description saved'}")
        reference = shot.get("role_reference") or {}
        image = _path(root, reference.get("image", "")) if isinstance(reference, dict) else None
        if image and image.is_file():
            st.image(str(image), caption="Visual reference for this camera shot. Inspect livery and travel order in the replay.", width="stretch")
        st.caption("Select the numbered tracks belonging to the lead and chase cars in this shot. Missing observations remain gaps; assigning roles does not reconnect an ID after a loss or across a camera cut.")
        options = [None] + sorted(ids)
        current_lead = child_summary.get("lead_id", shot.get("lead_id"))
        current_chase = child_summary.get("chase_id", shot.get("chase_id"))
        mode = st.radio("Role edit scope", ["Bounded intervals", "Entire camera shot"],
                        key=f"full_role_scope_{run_id}_{selected}", horizontal=True)
        if mode == "Bounded intervals":
            _interval_editor(root, directory, child_summary, selected, run_id, ids,
                             summary.get("manifest_path", "data/full_run_catalog.json"))
            return
        st.caption("Applying the pair below assigns those two IDs to the entire camera shot and replaces any reviewed interval mapping.")
        with st.form(f"full_role_form_{run_id}_{selected}"):
            lead_column, chase_column = st.columns(2)
            lead_id = lead_column.selectbox("Lead track ID", options,
                                            index=options.index(current_lead) if current_lead in options else 0,
                                            format_func=lambda value: "Choose an ID" if value is None else f"ID {value}",
                                            key=f"full_lead_id_{run_id}_{selected}")
            chase_id = chase_column.selectbox("Chase track ID", options,
                                              index=options.index(current_chase) if current_chase in options else 0,
                                              format_func=lambda value: "Choose an ID" if value is None else f"ID {value}",
                                              key=f"full_chase_id_{run_id}_{selected}")
            submitted = st.form_submit_button("Apply shot roles and rebuild run", type="primary", width="stretch")
        st.caption("A saved correction is recorded as an unverified user assignment. It does not imply expert review.")
        if submitted:
            if lead_id is None or chase_id is None or lead_id == chase_id:
                st.error("Choose two different observed track IDs for this camera shot.")
                return
            manifest = _path(root, summary.get("manifest_path", "data/full_run_catalog.json"))
            if manifest is None or not manifest.is_file():
                st.error("The full run manifest is missing. Restore it before rebuilding role assignments.")
                return
            try:
                from driftlens.full_run import update_shot_roles

                with st.spinner("Updating this shot's roles and rebuilding the complete replay…"):
                    update_shot_roles(manifest_path=manifest, run_dir=directory, shot_id=selected,
                                      lead_id=int(lead_id), chase_id=int(chase_id))
                st.session_state["full_run_notice"] = "Camera shot roles saved as an unverified user assignment. The complete replay and timeline were rebuilt."
                st.rerun()
            except Exception as error:
                st.error(f"Could not update the camera shot roles: {error}")


def _interval_editor(root: Path, directory: Path, child_summary: dict,
                     selected: str, run_id: str, ids: set[int], manifest_value: str) -> None:
    """Replace a half-open role map after validating every edited interval."""
    from .review import validate_role_intervals

    duration = _number(child_summary.get("duration_seconds"))
    if duration <= 0:
        st.error("This shot has no valid duration for bounded role editing.")
        return
    existing = child_summary.get("role_intervals")
    if existing is None:
        existing = [{"start_clip_seconds": 0.0, "end_clip_seconds": duration,
                     "lead_id": child_summary.get("lead_id"), "chase_id": child_summary.get("chase_id"),
                     "review_basis": "Existing whole shot assignment; visually review the interval before saving."}]
    records = [{"Start within shot (s)": interval.get("start_clip_seconds"),
                "End within shot (s)": interval.get("end_clip_seconds"),
                "Lead ID": interval.get("lead_id"), "Chase ID": interval.get("chase_id"),
                "Review basis": interval.get("review_basis", "")}
               for interval in existing]
    initial = pd.DataFrame(records, columns=["Start within shot (s)", "End within shot (s)", "Lead ID", "Chase ID", "Review basis"])
    for name in ("Start within shot (s)", "End within shot (s)", "Lead ID", "Chase ID"):
        initial[name] = pd.to_numeric(initial[name], errors="coerce").astype(float)
    st.caption(f"Shot duration {duration:.3f}s. Enter ordered, nonoverlapping intervals. Starts are included; ends are excluded. Leave a role ID empty when uncertain. Uncovered time stays unknown. Available IDs: {', '.join(str(value) for value in sorted(ids)) or 'none'}.")
    with st.form(f"full_interval_form_{run_id}_{selected}"):
        edited = st.data_editor(initial, hide_index=True, num_rows="dynamic", width="stretch",
                                key=f"full_interval_table_{run_id}_{selected}",
                                column_config={
                                    "Start within shot (s)": st.column_config.NumberColumn(min_value=0.0, max_value=duration, step=.01, format="%.3f", required=True),
                                    "End within shot (s)": st.column_config.NumberColumn(min_value=0.0, max_value=duration, step=.01, format="%.3f", required=True),
                                    "Lead ID": st.column_config.NumberColumn(min_value=0, step=1, format="%d"),
                                    "Chase ID": st.column_config.NumberColumn(min_value=0, step=1, format="%d"),
                                    "Review basis": st.column_config.TextColumn(help="Record the visible livery and travel-order evidence, or why a role remains unknown."),
                                })
        clear = st.checkbox("Mark the entire camera shot as role unknown", key=f"full_interval_clear_{run_id}_{selected}")
        submitted = st.form_submit_button("Save reviewed intervals and rebuild run", type="primary", width="stretch")
    st.caption("Saving records an unverified user assignment and rebuilds the shot replay, frame metrics and complete exports together. If rebuilding fails, the saved outputs are restored.")
    if not submitted:
        return
    manifest = _path(root, manifest_value)
    if manifest is None or not manifest.is_file():
        st.error("The full run manifest is missing. Restore it before rebuilding role assignments.")
        return
    try:
        intervals = []
        if not clear:
            for row in edited.to_dict("records"):
                role_values = []
                for name in ("Lead ID", "Chase ID"):
                    value = row.get(name)
                    if value is None or pd.isna(value):
                        role_values.append(None)
                    else:
                        number = float(value)
                        if not math.isfinite(number) or not number.is_integer() or int(number) not in ids:
                            raise ValueError("Each role must be empty or one of the observed integer track IDs.")
                        role_values.append(int(number))
                intervals.append({"start_clip_seconds": float(row["Start within shot (s)"]),
                                  "end_clip_seconds": float(row["End within shot (s)"]),
                                  "lead_id": role_values[0], "chase_id": role_values[1],
                                  "review_basis": str(row.get("Review basis") or "User reviewed interval; no expert validation.")})
        validate_role_intervals(intervals, duration, ids)
        from .full_run import update_shot_roles
        with st.spinner("Saving bounded roles and rebuilding every complete run export…"):
            update_shot_roles(manifest_path=manifest, run_dir=directory, shot_id=selected, role_intervals=intervals)
        st.session_state["full_run_notice"] = "Reviewed role intervals saved as an unverified user assignment. The complete exports and run conclusions were rebuilt."
        st.rerun()
    except Exception as error:
        st.error(f"Could not save the reviewed role intervals: {error}")


def _charts(metrics: pd.DataFrame, shots: list[dict], duration: float) -> None:
    if metrics.empty or "run_time" not in metrics:
        st.info("The combined frame timeline is not available for this run.")
        return
    data = metrics.copy()
    use_playback = "playback_time" in data
    if use_playback:
        data["run_time"] = data["playback_time"]
    axis_title = "Playback time (seconds)" if use_playback else "Time within complete run (seconds)"
    tooltip_title = "Playback time" if use_playback else "Run time"
    data["run_time"] = pd.to_numeric(data["run_time"], errors="coerce")
    data = data.loc[data["run_time"].notna()].sort_values("run_time").reset_index(drop=True)
    for name in ("lead_observed", "chase_observed", "pair_observed"):
        data[name] = _bool(data[name]) if name in data else False
    if "shot_id" not in data:
        data["shot_id"] = "unassigned"
    if "local_shot_index" not in data:
        data["local_shot_index"] = 0
    proxy = pd.to_numeric(data.get("separation_proxy", pd.Series(index=data.index, dtype=float)), errors="coerce")
    data["separation_proxy"] = proxy.where(data["pair_observed"])
    valid = data["pair_observed"] & data["separation_proxy"].notna()
    boundary = data["shot_id"].ne(data["shot_id"].shift()) | data["local_shot_index"].ne(data["local_shot_index"].shift())
    if "camera_cut" in data:
        boundary = boundary | _bool(data["camera_cut"])
    if "role_assignment_cut" in data:
        boundary = boundary | _bool(data["role_assignment_cut"])
    data["segment"] = ((~valid) | boundary).cumsum()
    cut_records = [{"run_time": _number(shot.get("run_start_seconds")), "label": shot.get("label", str(shot.get("shot_id", "Camera shot")))}
                   for shot in shots if _number(shot.get("run_start_seconds")) > 0]
    if "camera_cut" in data:
        for _, row in data.loc[_bool(data["camera_cut"]) & (data["run_time"] > 0)].iterrows():
            if not any(abs(record["run_time"] - row["run_time"]) < .001 for record in cut_records):
                cut_records.append({"run_time": row["run_time"], "label": f"Camera cut within {row['shot_id']}"})
    cut_data = pd.DataFrame(cut_records, columns=["run_time", "label"])
    domain = alt.Scale(domain=[0, max(duration, _number(data["run_time"].max()))])
    cuts = alt.Chart(cut_data).mark_rule(color="#76858f", strokeDash=[4, 4], strokeWidth=1).encode(
        x=alt.X("run_time:Q", scale=domain), tooltip=[alt.Tooltip("label:N", title="Camera shot"), alt.Tooltip("run_time:Q", title=tooltip_title, format=".2f")],
    )
    st.markdown("**Image separation across the run**")
    st.caption("Center separation divided by mean detected box width. This is a view dependent image measurement. Dashed lines mark camera shot boundaries.")
    if valid.any():
        separation = alt.Chart(data.loc[valid]).mark_line(color=ACCENT, strokeWidth=2.5).encode(
            x=alt.X("run_time:Q", title=axis_title, scale=domain),
            y=alt.Y("separation_proxy:Q", title="Separation / mean box width", scale=alt.Scale(zero=True)),
            detail=["shot_id:N", "local_shot_index:N", "segment:N"],
            tooltip=[alt.Tooltip("run_time:Q", title=tooltip_title, format=".2f"), alt.Tooltip("source_time:Q", title="Source time", format=".2f"), alt.Tooltip("shot_id:N", title="Shot"), alt.Tooltip("separation_proxy:Q", title="Image separation", format=".3f")],
        )
        chart = alt.layer(separation, cuts).properties(height=240).configure_axis(gridColor="#253139", labelColor="#adbcc7", titleColor="#adbcc7").configure_view(stroke=None)
        st.altair_chart(chart, width="stretch")
    else:
        st.info("No paired image measurements are available. Review shot roles and missing observations in the shot table.")
    st.markdown("**Observed cars and gaps**")
    interval = data["run_time"].diff().loc[lambda values: values > 0].median()
    step = _number(interval, .1)
    data["end_time"] = data["run_time"] + step
    status_frames = []
    for role in ("lead", "chase"):
        assigned = pd.to_numeric(data[f"{role}_track_id"], errors="coerce").notna() if f"{role}_track_id" in data else pd.Series(False, index=data.index)
        frames = data[["run_time", "end_time", "shot_id"]].copy()
        frames["role"] = role.title()
        frames["status"] = "Role unknown"
        frames.loc[assigned, "status"] = "Assigned ID not observed"
        frames.loc[assigned & data[f"{role}_observed"], "status"] = "Observed"
        status_frames.append(frames)
    availability = pd.concat(status_frames, ignore_index=True)
    bars = alt.Chart(availability).mark_rect().encode(
        x=alt.X("run_time:Q", title=axis_title, scale=domain), x2="end_time:Q",
        y=alt.Y("role:N", title=None, sort=["Lead", "Chase"]),
        color=alt.Color("status:N", title=None, scale=alt.Scale(domain=["Observed", "Assigned ID not observed", "Role unknown"], range=[ACCENT, "#ff705f", "#788898"])),
        tooltip=[alt.Tooltip("run_time:Q", title=tooltip_title, format=".2f"), "shot_id:N", "role:N", "status:N"],
    )
    chart = alt.layer(bars, cuts).properties(height=90).configure_axis(gridColor="#253139", labelColor="#adbcc7", titleColor="#adbcc7").configure_view(stroke=None)
    st.altair_chart(chart, width="stretch")
    st.caption("The chart distinguishes unknown roles from assigned IDs absent in current observations. Hidden positions are not filled in; line segments stop at gaps, role changes and camera cuts.")


def render_full_run(root: Path) -> None:
    """Render existing assembled results, with no processing or filesystem mutation."""
    root = Path(root).resolve()
    collection = root / "outputs" / "full_runs"
    available = []
    if collection.exists():
        for directory in collection.iterdir():
            safe = _path(root, directory)
            if safe is None or not safe.is_dir():
                continue
            summary = _json(root, safe / "summary.json")
            if summary.get("status", "complete") == "complete" and summary.get("run_id"):
                available.append((safe, summary))
    available.sort(key=lambda pair: (-_number(pair[1].get("duration_seconds")), pair[0].name))
    st.subheader("The complete tandem run")
    st.caption("Watch the whole selected run in order, then jump into the camera shots to inspect tracking and observation gaps.")
    if notice := st.session_state.pop("full_run_notice", None):
        st.success(notice)
    if not available:
        st.info("No complete run replay is prepared yet. Shot review remains available while the complete run is assembled.")
        return
    options = {str(directory): (directory, summary) for directory, summary in available}
    if st.session_state.get("full_run_selected") not in options:
        preferred = _json(root, root / "data" / "inference_profiles.json").get("default_full_run_id")
        st.session_state["full_run_selected"] = next((key for key, (_, saved) in options.items()
                                                      if saved.get("run_id") == preferred), next(iter(options)))
    selected = st.selectbox("Complete run", list(options), key="full_run_selected",
                            format_func=lambda key: f"{options[key][1].get('title', options[key][1]['run_id'])} · {_clock(options[key][1].get('duration_seconds'))}")
    directory, summary = options[selected]
    shots = summary.get("shots", [])
    shots_path = _file(root, directory, summary, "shots", "shots.json")
    if not shots and shots_path:
        try:
            saved_shots = json.loads(shots_path.read_text(encoding="utf-8-sig"))
            shots = saved_shots if isinstance(saved_shots, list) else saved_shots.get("shots", [])
        except (OSError, ValueError, AttributeError):
            shots = []
    duration = _number(summary.get("duration_seconds"))
    reviewed = int(_number(summary.get("roles_reviewed_shots")))
    metrics_path = _file(root, directory, summary, "metrics", "timeline.csv")
    try:
        metrics = pd.read_csv(metrics_path) if metrics_path else pd.DataFrame()
    except (OSError, ValueError):
        metrics = pd.DataFrame()
    columns = st.columns(4)
    columns[0].metric("Complete run duration", f"{duration:.1f} seconds")
    columns[1].metric("Camera shots", int(_number(summary.get("shot_count"), len(shots))))
    columns[2].metric("Pair observed", f"{100 * _number(summary.get('pair_coverage')):.1f}%" if reviewed else "Roles pending")
    columns[3].metric("Sampled frames", int(_number(summary.get("frame_count"), len(metrics))))
    st.caption(f"Source interval {_clock(summary.get('source_start_seconds'))} to {_clock(summary.get('source_end_seconds'))} · {_number(summary.get('total_processing_seconds')):.1f}s measured processing · {reviewed} camera shots with reviewed role assignments")
    source_duration = _number(summary.get("source_duration_seconds"), _number(summary.get("source_end_seconds")) - _number(summary.get("source_start_seconds")))
    if abs(source_duration - duration) > .01:
        st.caption(f"Source run {source_duration:.1f}s · compiled replay {duration:.1f}s. The exported timeline retains source timestamps and playback offsets.")
    status = str(summary.get("role_review_status", "unassigned"))
    if status == "assistant_visual_review_no_human_expert":
        st.caption("Current shot roles were visually reviewed by an AI assistant and have not received human expert review.")
    else:
        st.caption(f"Recorded role review status: {status.replace('_', ' ')}")
    st.caption("Track IDs reset for each camera shot. Lead and chase assignments are reviewed separately for each shot; this replay does not establish continuous identity association between camera views.")
    if (summary.get("inference_profile") or {}).get("recover_vehicle_classes"):
        st.caption("R marks a current detector box recovered through a recent vehicle appearance match. Original predicted classes and recovery evidence remain in each shot's observations CSV. The displayed confidence belongs to the original class, not a vehicle identity probability.")
    shot_options = {str(shot.get("shot_id", index)): shot for index, shot in enumerate(shots)}
    jump_key = f"full_run_jump_{summary['run_id']}"
    jump = st.selectbox("Jump to camera shot", [None] + list(shot_options), key=jump_key,
                        format_func=lambda value: "Play the complete run from the beginning" if value is None else f"{shot_options[value].get('label', value)} · {_clock(shot_options[value].get('run_start_seconds'))}")
    start_time = _number(shot_options[jump].get("run_start_seconds")) if jump is not None else 0
    event = render_run_analysis(root, directory, summary, metrics, shots=shots)
    video = _file(root, directory, summary, "video", "annotated.mp4")
    if video:
        render_review_player(root, video, start_seconds=start_time, event=event)
    else:
        st.warning("The combined annotated video is missing from this saved run.")
    _role_editor(root, directory, summary, shot_options, jump)
    _charts(metrics, shots, duration)
    st.markdown("**Camera shot details**")
    if shots:
        rows = [{"Shot": shot.get("shot_id"), "View": shot.get("label"), "Playback start seconds": shot.get("run_start_seconds"),
                 "Playback end seconds": shot.get("run_end_seconds"), "Source start seconds": shot.get("start_seconds"),
                 "Source end seconds": shot.get("end_seconds"), "Lead ID": shot.get("lead_id"), "Chase ID": shot.get("chase_id"),
                 "Role status": shot.get("role_status", "unassigned"),
                 "Role mapping": f"{len(shot['role_intervals'])} reviewed intervals" if shot.get("role_intervals") else "One pair for entire shot" if shot.get("lead_id") is not None and shot.get("chase_id") is not None else "Unassigned",
                 "Sampled frames": shot.get("frame_count"),
                 "Paired frames": shot.get("paired_frames"), "Pair observed %": _number(shot.get("pair_coverage")) * 100 if shot.get("role_intervals") or (shot.get("lead_id") is not None and shot.get("chase_id") is not None) else None}
                for shot in shots]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.caption("No camera shot metadata is saved.")
    st.markdown("**Export the complete run**")
    downloads = [
        ("Complete annotated replay", video, "video/mp4"),
        ("Complete timeline CSV", metrics_path, "text/csv"),
        ("Complete run summary", _file(root, directory, summary, "summary", "summary.json"), "application/json"),
        ("Camera shots JSON", shots_path, "application/json"),
    ]
    for column, (label, file_path, mime) in zip(st.columns(4), downloads):
        if file_path:
            with file_path.open("rb") as handle:
                column.download_button(label, handle, file_name=f"{summary['run_id']}_{file_path.name}", mime=mime,
                                       key=f"full_export_{summary['run_id']}_{label}", width="stretch")
    st.caption("Image separation is not physical vehicle distance, speed, drift angle or official judging. Pair availability measures observations of assigned IDs, not verified tracking accuracy.")
