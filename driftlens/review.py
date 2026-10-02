"""Manual role assignment and observed image measurements."""
from __future__ import annotations

import csv
import json
import math
import shutil
import uuid
from pathlib import Path

METRIC_FIELDS = ["frame_index", "clip_time", "source_time", "shot_index", "lead_observed", "chase_observed", "pair_observed", "separation_proxy", "lead_confidence", "chase_confidence", "lead_track_id", "chase_track_id"]


def validate_role_intervals(intervals: list[dict], duration: float, available_ids: set[int] | None = None) -> None:
    """Reject overlapping role decisions and IDs not observed in this shot."""
    previous_end = 0.0
    for entry in intervals:
        start, end = float(entry["start_clip_seconds"]), float(entry["end_clip_seconds"])
        if not math.isfinite(start) or not math.isfinite(end) or start < previous_end or not 0 <= start < end <= duration + 0.001:
            raise ValueError("Role intervals must be finite, ordered, nonoverlapping and inside the shot.")
        lead, chase = entry.get("lead_id"), entry.get("chase_id")
        if lead is not None and chase is not None and int(lead) == int(chase):
            raise ValueError("Lead and chase must be different track IDs.")
        for identifier in (lead, chase):
            if identifier is not None and (int(identifier) != identifier or (available_ids is not None and int(identifier) not in available_ids)):
                raise ValueError("Role intervals must refer to observed integer track IDs.")
        previous_end = end


def roles_at(summary: dict, clip_time: float) -> tuple[int | None, int | None]:
    """An explicit interval map leaves all uncovered time unassigned."""
    if "role_intervals" in summary:
        for entry in summary["role_intervals"]:
            if float(entry["start_clip_seconds"]) <= clip_time < float(entry["end_clip_seconds"]):
                return entry.get("lead_id"), entry.get("chase_id")
        return None, None
    return summary.get("lead_id"), summary.get("chase_id")


def _observed(row: dict) -> bool:
    return str(row.get("observed", True)).lower() not in {"false", "0", "", "none"}


def derive_frame_metrics(observations: list[dict], frames: list[dict], lead_id: int | None, chase_id: int | None, role_intervals: list[dict] | None = None) -> list[dict]:
    """Measure only detected, assigned cars in each sampled frame.

    Pixel separation normalized by mean box width is a perspective dependent
    screen measurement. No values are interpolated across missing detections.
    """
    if lead_id is not None and chase_id is not None and int(lead_id) == int(chase_id):
        raise ValueError("Lead and chase must be different track IDs.")
    if role_intervals is not None:
        validate_role_intervals(role_intervals, max((float(f["clip_time"]) for f in frames), default=0) + 1)
    role_summary = {"lead_id": lead_id, "chase_id": chase_id}
    if role_intervals is not None:
        role_summary["role_intervals"] = role_intervals
    indexed = {}
    for observation in observations:
        if not _observed(observation) or observation.get("track_id") in (None, ""):
            continue
        key = (int(observation["frame_index"]), int(observation.get("shot_index", 0)), int(observation["track_id"]))
        previous = indexed.get(key)
        if previous is None or float(observation["confidence"]) > float(previous["confidence"]):
            indexed[key] = observation
    metrics = []
    for frame in frames:
        index, shot = int(frame["frame_index"]), int(frame.get("shot_index", 0))
        active_lead, active_chase = roles_at(role_summary, float(frame["clip_time"]))
        lead = indexed.get((index, shot, int(active_lead))) if active_lead is not None else None
        chase = indexed.get((index, shot, int(active_chase))) if active_chase is not None else None
        proxy = None
        if lead is not None and chase is not None:
            lx1, ly1, lx2, ly2 = [float(lead[k]) for k in ("x1", "y1", "x2", "y2")]
            cx1, cy1, cx2, cy2 = [float(chase[k]) for k in ("x1", "y1", "x2", "y2")]
            width = ((lx2 - lx1) + (cx2 - cx1)) / 2
            if width > 0:
                proxy = math.hypot((lx1 + lx2 - cx1 - cx2) / 2, (ly1 + ly2 - cy1 - cy2) / 2) / width
        metrics.append({**{key: frame[key] for key in ("frame_index", "clip_time", "source_time", "shot_index")}, "lead_observed": lead is not None, "chase_observed": chase is not None, "pair_observed": proxy is not None, "separation_proxy": proxy, "lead_confidence": float(lead["confidence"]) if lead else None, "chase_confidence": float(chase["confidence"]) if chase else None, "lead_track_id": active_lead, "chase_track_id": active_chase})
    return metrics


def read_rows(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_metrics(path: Path, metrics: list[dict]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(metrics)


def assign_roles(run_dir: Path, lead_id: int, chase_id: int) -> dict:
    """Assign observed IDs to roles and regenerate exports from the source."""
    from .pipeline import render_run
    run_dir = Path(run_dir).resolve()
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    lead_id, chase_id = int(lead_id), int(chase_id)
    if lead_id == chase_id:
        raise ValueError("Choose different IDs for lead and chase.")
    available = {int(value) for value in summary["track_ids"]}
    if lead_id not in available or chase_id not in available:
        raise ValueError("Both assigned IDs must occur in this run.")
    if summary.get("shot_count", 1) > 1:
        raise ValueError("Trim a single camera shot before assigning roles.")
    frames = read_rows(run_dir / "frames.csv")
    observations = read_rows(run_dir / "observations.csv")
    metrics = derive_frame_metrics(observations, frames, lead_id, chase_id)
    summary.update(lead_id=lead_id, chase_id=chase_id, paired_frames=sum(row["pair_observed"] for row in metrics), pair_coverage=sum(row["pair_observed"] for row in metrics) / len(metrics) if metrics else 0)
    summary.pop("role_intervals", None)
    # Inherit the project directory's Windows permissions. TemporaryDirectory
    # creates a private ACL; replacing exports from it can make them unreadable
    # when the project was prepared with an elevated setup process.
    stage = (run_dir / f".role_update_{uuid.uuid4().hex}").resolve()
    stage.relative_to(run_dir)
    stage.mkdir()
    try:
        write_metrics(stage / "frame_metrics.csv", metrics)
        render_run(stage, summary, observations, frames)
        (stage / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        for name in ("annotated.mp4", "frame_metrics.csv", "summary.json"):
            (stage / name).replace(run_dir / name)
    finally:
        resolved_stage = stage.resolve()
        resolved_stage.relative_to(run_dir)
        if resolved_stage.parent != run_dir or not resolved_stage.name.startswith(".role_update_"):
            raise ValueError("Role staging cleanup escaped the run directory.")
        shutil.rmtree(resolved_stage)
    return summary
