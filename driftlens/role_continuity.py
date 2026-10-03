"""Continue a reviewed pair with explicit, unverified appearance hypotheses."""
from __future__ import annotations

import json
from pathlib import Path

from .clip_role_review import _publish_clip_role_intervals, _validated_inputs
from .review import read_rows


def follow_clip_roles(run_dir: Path, seed: dict) -> dict:
    """Use an explicit reviewed interval; publish matches and exports together."""
    from .role_identity import suggest_role_intervals

    run_dir = Path(run_dir).resolve()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8-sig"))
    frames = read_rows(run_dir / "frames.csv")
    observations = read_rows(run_dir / "observations.csv")
    seeds = _validated_inputs(summary, frames, observations, [seed])
    if seeds[0]["lead_id"] is None or seeds[0]["chase_id"] is None:
        raise ValueError("Choose two different observed seed IDs after checking lead and chase.")
    result = suggest_role_intervals(Path(summary["source_path"]), frames, observations, seeds,
                                   float(summary["duration_seconds"]))
    continuation = {**result["summary"], "active": True, "seed_intervals": seeds,
                    "method": "reviewed_seed_appearance_role_continuation_v1",
                    "review_status": "automatic_appearance_unverified",
                    "score_note": "Appearance similarity and margins are heuristic scores, not calibrated identity probabilities.",
                    "scope": "Only current observed boxes may receive roles. Uncertain, missing or merged candidates remain unassigned; no hidden positions or expert labels are created."}
    return _publish_clip_role_intervals(run_dir, result["intervals"],
                                       continuation=continuation, decisions=result["decisions"])


def render_role_continuity(run_dir: Path, summary: dict) -> None:
    import streamlit as st

    frames = read_rows(run_dir / "frames.csv")
    observations = read_rows(run_dir / "observations.csv")
    duration = float(summary.get("duration_seconds", 0))
    ids = sorted({int(row["track_id"]) for row in observations if row.get("track_id") not in (None, "")})
    if not frames or not ids or duration <= 0:
        return
    cuts = [float(current["clip_time"]) for previous, current in zip(frames, frames[1:])
            if int(previous.get("shot_index", 0)) != int(current.get("shot_index", 0))]
    saved_seeds = (summary.get("role_continuation") or {}).get("seed_intervals", [])
    first = next((row for row in saved_seeds or summary.get("role_intervals", [])
                  if row.get("lead_id") is not None and row.get("chase_id") is not None), {})
    seed_start = float(first.get("start_clip_seconds", 0))
    seed_end = min(float(first.get("end_clip_seconds", duration)),
                   next((cut for cut in cuts if cut > seed_start), duration))
    lead = first.get("lead_id", summary.get("lead_id"))
    chase = first.get("chase_id", summary.get("chase_id"))
    if lead in ids and chase in ids:
        jointly_observed = []
        by_frame = {}
        for row in observations:
            by_frame.setdefault(int(row["frame_index"]), set()).add(int(row["track_id"]))
        for frame in frames:
            t = float(frame["clip_time"])
            if seed_start <= t < seed_end and {int(lead), int(chase)}.issubset(by_frame.get(int(frame["frame_index"]), set())):
                jointly_observed.append(t)
        if jointly_observed:
            seed_end = min(seed_end, jointly_observed[-1] + 1 / float(summary.get("sampled_fps", 10)))
    if seed_end <= seed_start:
        seed_start, seed_end = 0.0, min(cuts[0] if cuts else duration, 3.0)
    choices = [None] + ids
    with st.expander("Follow lead and chase across views", expanded=False):
        st.caption("Check the first pair in a clear interval. The app will match their appearance to current boxes after camera changes or new track IDs. Similar cars, overlap and smoke can leave roles unknown.")
        with st.form(f"role_continuation_{run_dir}"):
            left, right = st.columns(2)
            start = left.number_input("Seed start seconds", min_value=0.0, max_value=duration,
                                      value=min(seed_start, duration), step=.1)
            end = right.number_input("Seed end seconds", min_value=0.0, max_value=duration,
                                     value=min(seed_end, duration), step=.1)
            lead_id = left.selectbox("Seed lead ID", choices, index=choices.index(lead) if lead in choices else 0,
                                     format_func=lambda value: "Choose an observed ID" if value is None else str(value))
            chase_id = right.selectbox("Seed chase ID", choices, index=choices.index(chase) if chase in choices else 0,
                                       format_func=lambda value: "Choose an observed ID" if value is None else str(value))
            checked = st.checkbox("I have checked lead and chase in the seed interval")
            submitted = st.form_submit_button("Follow this pair through the clip", type="primary", width="stretch")
        if submitted:
            if not checked:
                st.error("Check the seed pair in the replay and confirm the checkbox first.")
            else:
                try:
                    with st.spinner("Matching the reviewed pair and rebuilding the replay and reports…"):
                        follow_clip_roles(run_dir, {"start_clip_seconds": start, "end_clip_seconds": end,
                                                   "lead_id": lead_id, "chase_id": chase_id,
                                                   "review_basis": "User confirmed the seed pair; later roles use unverified appearance matching."})
                    st.session_state["notice"] = "Pair matching saved. Review automatic roles and unknown intervals in the replay."
                    st.rerun()
                except Exception as error:
                    st.error(f"Could not continue this pair: {error}")
        continuation = summary.get("role_continuation") or {}
        if continuation.get("active"):
            st.caption("Later roles are automatic appearance matches and need visual review. Similarity scores are not identity probabilities. Saving manual interval corrections overrides this result.")
            evidence = run_dir / "role_matches.json"
            if evidence.is_file():
                st.download_button("Download role match evidence", evidence.read_bytes(),
                                   file_name=f"{run_dir.name}_role_matches.json", mime="application/json",
                                   key=f"role_matches_{run_dir}")
