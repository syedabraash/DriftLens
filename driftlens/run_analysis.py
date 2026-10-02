"""Deterministic conclusions from observed, camera scoped image measurements.

This module does not estimate physical telemetry or infer detector failure from
unassigned roles. A detector count is evidence about candidates in one frame,
not evidence that either physical tandem car was visible or missed.
"""
from __future__ import annotations

from collections import Counter
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable


LIMITATIONS = [
    "Separation is image center distance divided by mean detected box width. It is view dependent, not physical distance, speed, drift angle, driver skill or an official score.",
    "Statistics are confined to one camera view. Track IDs and roles do not establish identity across camera cuts.",
    "Only observed, assigned pairs with finite nonnegative separation are accepted. Missing measurements are not interpolated.",
    "Unknown intervals describe sampled replay frames, not continuous visibility between samples. Their ends exclude the next frame.",
    "An empty detector candidate list or absent track ID does not establish a missed physical car. Role uncertainty alone gives no detector failure evidence.",
    "Role assignments need visual review and are not expert verified tracking accuracy.",
]


def _number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _flag(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "1.0", "yes"}


def _identifier(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None and number >= 0 and number.is_integer() else None


def _records(rows: Iterable[dict] | Any) -> list[dict]:
    # Accept a DataFrame without requiring pandas for reports or CLI exports.
    if hasattr(rows, "to_dict"):
        return rows.to_dict("records")
    return [dict(row) for row in rows]


def _counts(rows: list[dict] | None, tracked: bool = False) -> dict[int, int] | None:
    if rows is None:
        return None
    result = Counter()
    for row in rows:
        index = _identifier(row.get("frame_index"))
        if index is None:
            raise ValueError("Evidence requires nonnegative integer frame indices.")
        if not _flag(row.get("observed", True)):
            continue
        if tracked and _identifier(row.get("track_id")) is None:
            continue
        result[index] += 1
    return dict(result)


def _recovered(row: dict) -> bool:
    """Class recovery provenance remains distinct from a vehicle prediction."""
    method = str(row.get("detection_method") or "").lower()
    source_class = _identifier(row.get("source_class_id"))
    return "recover" in method or (source_class is not None and source_class not in {2, 7})


def _recovery_frames(rows: list[dict] | None, tracked: bool = False) -> tuple[dict | None, dict]:
    if rows is None:
        return None, {}
    counts, ids = Counter(), {}
    for row in rows:
        if not _flag(row.get("observed", True)) or not _recovered(row):
            continue
        index, identifier = _identifier(row.get("frame_index")), _identifier(row.get("track_id"))
        if index is None or (tracked and identifier is None):
            continue
        counts[index] += 1
        if identifier is not None:
            ids.setdefault(index, set()).add(identifier)
    return dict(counts), ids


def _provenance_totals(evidence: dict[str, dict], rows: list[dict], class_names: dict) -> dict:
    """Count exported current boxes, never certify their physical identity."""
    allowed = {}
    for row in rows:
        allowed.setdefault(row["shot_id"], set()).add(row["local_frame_index"])
    result = {"detector_exports_available": 0, "tracker_exports_available": 0,
              "selected_box_count": 0, "tracked_box_count": 0,
              "native_vehicle_prediction_boxes": 0, "orientation_vehicle_prediction_boxes": 0,
              "recovered_selected_boxes": 0, "recovered_tracked_boxes": 0,
              "unspecified_selected_box_provenance": 0, "unspecified_tracked_box_provenance": 0,
              "recovered_source_classes": []}
    source_classes = {}
    for shot_id, values in evidence.items():
        frame_ids = allowed.get(str(shot_id), set())
        if not frame_ids:
            continue
        for name, tracked in (("detections", False), ("observations", True)):
            exported = values.get(name)
            if exported is None:
                continue
            result["tracker_exports_available" if tracked else "detector_exports_available"] += 1
            for row in exported:
                index = _identifier(row.get("frame_index"))
                if index not in frame_ids or not _flag(row.get("observed", True)):
                    continue
                if tracked and _identifier(row.get("track_id")) is None:
                    continue
                result["tracked_box_count" if tracked else "selected_box_count"] += 1
                method = str(row.get("detection_method") or "")
                source_class = _identifier(row.get("source_class_id", row.get("class")))
                if _recovered(row):
                    result["recovered_tracked_boxes" if tracked else "recovered_selected_boxes"] += 1
                    key = source_class
                    counts = source_classes.setdefault(key, {"selected_boxes": 0, "tracked_boxes": 0})
                    counts["tracked_boxes" if tracked else "selected_boxes"] += 1
                elif method in {"native_detector", "orientation_detector"} and source_class in {2, 7}:
                    if not tracked:
                        result["native_vehicle_prediction_boxes" if method == "native_detector" else "orientation_vehicle_prediction_boxes"] += 1
                else:
                    result["unspecified_tracked_box_provenance" if tracked else "unspecified_selected_box_provenance"] += 1
    for source_class in sorted(source_classes, key=lambda value: -1 if value is None else value):
        name = class_names.get(str(source_class), class_names.get(source_class)) if source_class is not None else None
        result["recovered_source_classes"].append({"source_class_id": source_class,
                                                   "source_class_name": str(name) if name is not None else "not recorded",
                                                   **source_classes[source_class]})
    return result


def _roles(row: dict, shot: dict) -> tuple[int | None, int | None]:
    if "lead_track_id" in row or "chase_track_id" in row:
        return _identifier(row.get("lead_track_id")), _identifier(row.get("chase_track_id"))
    if "role_intervals" in shot:
        clip_time = _number(row.get("clip_time"))
        if clip_time is None:
            source, start = _number(row.get("source_time")), _number(shot.get("start_seconds"))
            clip_time = source - start if source is not None and start is not None else None
        for interval in shot.get("role_intervals", []):
            start, end = _number(interval.get("start_clip_seconds")), _number(interval.get("end_clip_seconds"))
            if clip_time is not None and start is not None and end is not None and start <= clip_time < end:
                return _identifier(interval.get("lead_id")), _identifier(interval.get("chase_id"))
        return None, None
    return _identifier(shot.get("lead_id")), _identifier(shot.get("chase_id"))


def _stamp(row: dict) -> dict:
    return {key: row[key] for key in ("frame_index", "local_frame_index", "playback_time", "source_time", "shot_id", "local_shot_index")}


def _event(kind: str, label: str, row: dict, **details: Any) -> dict:
    return {"kind": kind, "label": label, **_stamp(row), **details}


def _interval(rows: list[dict]) -> dict:
    first, last = rows[0], rows[-1]
    return {
        **_stamp(first), "start_playback_seconds": first["playback_time"],
        "end_playback_seconds": last["end_playback_time"],
        "start_source_seconds": first["source_time"], "last_sample_source_seconds": last["source_time"],
        "end_source_seconds": last["end_source_time"],
        "last_frame_index": last["frame_index"], "sample_count": len(rows),
        "sampled_duration_seconds": round(sum(row["end_playback_time"] - row["playback_time"] for row in rows), 6),
        "measurement_statuses": sorted({row["measurement_status"] for row in rows}),
        "observation_statuses": sorted({row["observation_status"] for row in rows}),
    }


def build_run_analysis(summary: dict, timeline_rows: Iterable[dict] | Any,
                       shots: list[dict] | None = None,
                       evidence_by_shot: dict[str, dict] | None = None) -> dict:
    """Build a JSON serializable report from a full timeline or clip metrics.

    Optional evidence maps a local shot ID to complete ``detections`` and
    ``observations`` CSV rows. ``None`` means unavailable; an empty list means
    a saved complete export containing no candidates/observations. Counts are
    indexed by local frame, so repeated tracker IDs in other shots never join.
    """
    shot_list = shots if shots is not None else summary.get("shots", [])
    default_id = str(summary.get("clip_id", summary.get("run_id", "shot")))
    shot_map = {str(shot.get("shot_id", shot.get("id", index))): dict(shot) for index, shot in enumerate(shot_list)}
    if not shot_map:
        shot_map[default_id] = dict(summary, shot_id=default_id)
    evidence = {}
    for shot_id, values in (evidence_by_shot or {}).items():
        recovered_detections, _ = _recovery_frames(values.get("detections"))
        recovered_tracks, recovered_ids = _recovery_frames(values.get("observations"), tracked=True)
        evidence[str(shot_id)] = (_counts(values.get("detections")), _counts(values.get("observations"), tracked=True),
                                  recovered_detections, recovered_tracks, recovered_ids)
    fps = _number(summary.get("sampled_fps"))
    period = 1 / fps if fps is not None and fps > 0 else None
    rows, previous_time, previous_frame = [], None, None
    for position, raw in enumerate(_records(timeline_rows)):
        playback = next((_number(raw.get(key)) for key in ("playback_time", "clip_time", "run_time") if _number(raw.get(key)) is not None), None)
        source = _number(raw.get("source_time"))
        frame = _identifier(raw.get("frame_index", position))
        if playback is None or playback < 0 or source is None or source < 0 or frame is None:
            raise ValueError("Analysis requires finite nonnegative playback/source times and integer frame indices.")
        if previous_time is not None and (playback <= previous_time or frame <= previous_frame):
            raise ValueError("Analysis frames and replay times must be unique and strictly ordered.")
        shot_id = str(raw.get("shot_id", default_id))
        shot = shot_map.get(shot_id, {})
        lead_id, chase_id = _roles(raw, shot)
        assigned = lead_id is not None and chase_id is not None and lead_id != chase_id
        lead = _flag(raw.get("lead_observed")) and lead_id is not None
        chase = _flag(raw.get("chase_observed")) and chase_id is not None
        proxy = _number(raw.get("separation_proxy"))
        accepted = assigned and lead and chase and _flag(raw.get("pair_observed")) and proxy is not None and proxy >= 0
        status = "pair_observed" if accepted else "role_unknown" if not assigned else "selected_tracks_missing" if not lead and not chase else "lead_track_missing" if not lead else "chase_track_missing" if not chase else "measurement_unavailable"
        local_index = _identifier(raw.get("local_frame_index", frame))
        local_shot = _identifier(raw.get("local_shot_index", raw.get("shot_index", 0)))
        if local_index is None or local_shot is None:
            raise ValueError("Local frame and shot indices must be nonnegative integers.")
        detector_counts, track_counts, recovered_detections, recovered_tracks, recovered_ids = evidence.get(shot_id, (None, None, None, None, {}))
        detection_count = detector_counts.get(local_index, 0) if detector_counts is not None else None
        track_count = track_counts.get(local_index, 0) if track_counts is not None else None
        observation_status = ("no_detector_candidates" if detection_count == 0 else
                              "detections_without_track_ids" if detection_count is not None and detection_count > 0 and track_count == 0 else
                              "tracked_observations_present" if track_count is not None and track_count > 0 else
                              "no_tracked_observations_detector_unknown" if track_count == 0 else "evidence_unavailable")
        rows.append({"frame_index": frame, "local_frame_index": local_index, "playback_time": playback,
                     "source_time": source, "shot_id": shot_id, "local_shot_index": local_shot,
                     "camera_cut": _flag(raw.get("camera_cut")), "role_assignment_cut": _flag(raw.get("role_assignment_cut")),
                     "lead_track_id": lead_id, "chase_track_id": chase_id, "assigned_pair": assigned,
                     "accepted": accepted, "separation_proxy": proxy if accepted else None,
                     "lead_status": "role_unknown" if lead_id is None else "observed" if lead else "selected_track_missing",
                     "chase_status": "role_unknown" if chase_id is None else "observed" if chase else "selected_track_missing",
                     "measurement_status": status, "observation_status": observation_status,
                     "detector_candidate_count": detection_count, "tracked_observation_count": track_count,
                     "recovered_selected_box_count": recovered_detections.get(local_index, 0) if recovered_detections is not None else None,
                     "recovered_tracked_box_count": recovered_tracks.get(local_index, 0) if recovered_tracks is not None else None,
                     "pair_uses_recovered_box": (accepted and bool({lead_id, chase_id} & recovered_ids.get(local_index, set()))) if track_counts is not None else None})
        previous_time, previous_frame = playback, frame
    if period is None:
        differences = [b["playback_time"] - a["playback_time"] for a, b in zip(rows, rows[1:])]
        period = median(differences) if differences else .1
    # Sample durations never bridge an unsampled gap or overrun a view boundary.
    for index, row in enumerate(rows):
        shot = shot_map.get(row["shot_id"], {})
        end = row["playback_time"] + period
        if index + 1 < len(rows):
            end = min(end, rows[index + 1]["playback_time"])
        shot_end = _number(shot.get("run_end_seconds"))
        if shot_end is not None and shot_end >= row["playback_time"]:
            end = min(end, shot_end)
        row["end_playback_time"] = round(end, 6)
        source_end = _number(shot.get("end_seconds"))
        end_source = row["source_time"] + (end - row["playback_time"])
        row["end_source_time"] = round(min(end_source, source_end) if source_end is not None and source_end >= row["source_time"] else end_source, 6)
    unknown, current, events = [], [], []
    prior = None
    for row in rows:
        boundary = prior is not None and (row["shot_id"] != prior["shot_id"] or row["local_shot_index"] != prior["local_shot_index"] or row["camera_cut"])
        role_change = prior is not None and ((row["lead_track_id"], row["chase_track_id"]) != (prior["lead_track_id"], prior["chase_track_id"]) or row["role_assignment_cut"])
        gap = prior is not None and row["playback_time"] - prior["end_playback_time"] > .000001
        if current and (row["accepted"] or boundary or role_change or gap):
            unknown.append(_interval(current))
            current = []
        if boundary:
            events.append(_event("camera_cut", "Camera view begins", row))
        elif role_change:
            events.append(_event("role_change", "Reviewed role mapping changes", row))
        if not row["accepted"]:
            current.append(row)
        prior = row
    if current:
        unknown.append(_interval(current))
    for interval in unknown:
        events.append({"kind": "unknown_interval", "label": "Pair measurement unknown", **interval})
    views = []
    for shot_id, local_shot in dict.fromkeys((row["shot_id"], row["local_shot_index"]) for row in rows):
        view_rows = [row for row in rows if row["shot_id"] == shot_id and row["local_shot_index"] == local_shot]
        pairs = [row for row in view_rows if row["accepted"]]
        values = [row["separation_proxy"] for row in pairs]
        view_gaps = [interval for interval in unknown if interval["shot_id"] == shot_id and interval["local_shot_index"] == local_shot]
        view = {"shot_id": shot_id, "local_shot_index": local_shot, "label": shot_map.get(shot_id, {}).get("label", shot_id),
                "sample_count": len(view_rows), "assigned_pair_samples": sum(row["assigned_pair"] for row in view_rows),
                "accepted_paired_samples": len(pairs), "pair_coverage": len(pairs) / len(view_rows),
                "start_playback_seconds": view_rows[0]["playback_time"], "end_playback_seconds": view_rows[-1]["end_playback_time"],
                "measurement_status_counts": dict(sorted(Counter(row["measurement_status"] for row in view_rows).items())),
                "observation_status_counts": dict(sorted(Counter(row["observation_status"] for row in view_rows).items())),
                "longest_unknown_interval": max(view_gaps, key=lambda item: item["sampled_duration_seconds"], default=None),
                "separation": None}
        if pairs:
            minimum = min(pairs, key=lambda row: row["separation_proxy"])
            maximum = max(pairs, key=lambda row: row["separation_proxy"])
            view["separation"] = {"minimum": min(values), "median": median(values), "maximum": max(values),
                                  "minimum_sample": _stamp(minimum), "maximum_sample": _stamp(maximum)}
            events.append(_event("minimum_separation", "Smallest image separation within this view", minimum, separation_proxy=minimum["separation_proxy"]))
            if maximum["frame_index"] != minimum["frame_index"]:
                events.append(_event("maximum_separation", "Largest image separation within this view", maximum, separation_proxy=maximum["separation_proxy"]))
        views.append(view)
    events.sort(key=lambda item: (item["playback_time"], item["frame_index"], item["kind"]))
    paired = sum(row["accepted"] for row in rows)
    assigned = sum(row["assigned_pair"] for row in rows)
    report = {"schema_version": 1, "run_id": str(summary.get("run_id", summary.get("clip_id", "run"))),
              "title": str(summary.get("title", summary.get("clip_id", "Run review"))),
              "role_review_status": summary.get("role_review_status", "unassigned"),
              "processing_profile": {key: summary.get(key) for key in ("tracker", "model", "imgsz", "sampled_fps", "processing_seconds", "processing_fps", "total_processing_seconds", "agnostic_nms", "inference_profile", "tracker_options", "orientations", "recover_vehicle_classes")},
              "sample_count": len(rows), "accepted_paired_samples": paired, "assigned_pair_samples": assigned,
              "pair_coverage": paired / len(rows) if rows else 0,
              "assigned_pair_coverage": paired / assigned if assigned else None,
              "measurement_status_counts": dict(sorted(Counter(row["measurement_status"] for row in rows).items())),
              "observation_status_counts": dict(sorted(Counter(row["observation_status"] for row in rows).items())),
              "longest_unknown_interval": max(unknown, key=lambda item: item["sampled_duration_seconds"], default=None),
              "unknown_intervals": unknown, "views": views, "events": events,
              "frame_statuses": rows, "limitations": LIMITATIONS.copy()}
    report["box_provenance"] = _provenance_totals(evidence_by_shot or {}, rows, summary.get("source_class_names") or {})
    report["accepted_paired_samples_with_recovery"] = sum(row["accepted"] and row["pair_uses_recovered_box"] is True for row in rows)
    report["conclusions"] = _conclusions(report)
    if summary.get("recover_vehicle_classes") or (summary.get("inference_profile") or {}).get("recover_vehicle_classes"):
        report["limitations"].append("This profile can recover current nonvehicle class boxes through conservative temporal vehicle appearance matches. Original predicted classes and recovery evidence are exported separately. Recovery is an experimental identity hypothesis, not human verified vehicle classification.")
    if report["box_provenance"]["recovered_selected_boxes"] or summary.get("recover_vehicle_classes") or (summary.get("inference_profile") or {}).get("recover_vehicle_classes"):
        report["recovery_note"] = "R marks a current source box with experimental class recovery. Its confidence is the original predicted class score, not car confidence or identity correctness. Original source class IDs/names and recovery provenance are reported separately; recovered boxes are not certified raw vehicle predictions."
        report["limitations"].append(report["recovery_note"])
        report["conclusions"].append(report["recovery_note"])
    return report


def _conclusions(report: dict) -> list[str]:
    count, pairs = report["sample_count"], report["accepted_paired_samples"]
    if not count:
        return ["No sampled frame timeline is available; no run conclusions can be measured."]
    result = [f"Accepted {pairs} of {count} sampled frames ({100 * pairs / count:.1f}%) for paired image separation; {report['assigned_pair_samples']} samples have two different assigned roles."]
    unknown_roles = report["measurement_status_counts"].get("role_unknown", 0)
    if unknown_roles:
        result.append(f"Roles are unknown or not a distinct pair in {unknown_roles} samples. Those gaps do not establish detector failure.")
    gap = report["longest_unknown_interval"]
    if gap:
        result.append(f"Longest pair measurement gap is {gap['sampled_duration_seconds']:.2f}s across {gap['sample_count']} sampled frames in {gap['shot_id']}, replay {gap['start_playback_seconds']:.2f}s to {gap['end_playback_seconds']:.2f}s; first source sample {gap['start_source_seconds']:.2f}s.")
    for view in report["views"]:
        separation = view["separation"]
        if separation:
            result.append(f"{view['label']} (view {view['local_shot_index']}): {view['accepted_paired_samples']}/{view['sample_count']} accepted pairs; image separation median {separation['median']:.3f}, range {separation['minimum']:.3f} to {separation['maximum']:.3f}. These values apply only to this view.")
        else:
            result.append(f"{view['label']} (view {view['local_shot_index']}): no accepted paired image separation samples.")
    evidence = report["observation_status_counts"]
    if evidence.get("no_detector_candidates"):
        result.append(f"Saved selected-box exports contain no vehicle candidate boxes in {evidence['no_detector_candidates']} frames. Visibility and missed physical cars cannot be established from that count alone.")
    if evidence.get("detections_without_track_ids"):
        result.append(f"Selected vehicle candidate boxes exist without any current tracked observation in {evidence['detections_without_track_ids']} frames; this is a detector/tracker output distinction, not verified identity loss.")
    provenance = report["box_provenance"]
    if provenance["detector_exports_available"]:
        result.append(f"Available selected-box exports contain {provenance['selected_box_count']} current boxes: {provenance['native_vehicle_prediction_boxes']} native vehicle-class predictions, {provenance['orientation_vehicle_prediction_boxes']} orientation vehicle-class predictions, {provenance['recovered_selected_boxes']} experimental recovered boxes and {provenance['unspecified_selected_box_provenance']} with unspecified provenance.")
    if provenance["recovered_selected_boxes"] or provenance["recovered_tracked_boxes"]:
        classes = ", ".join(f"{item['source_class_name']} (class {item['source_class_id']})" for item in provenance["recovered_source_classes"])
        result.append(f"Recovered source classes: {classes}. {provenance['recovered_tracked_boxes']} exported tracked boxes use recovery; {report['accepted_paired_samples_with_recovery']} accepted pair samples include at least one recovered box.")
    return result


def report_text(report: dict) -> str:
    """A readable downloadable report containing timestamps and limitations."""
    lines = [report["title"], f"Run: {report['run_id']}", f"Role review: {report['role_review_status']}", "", *report["conclusions"], "", "Review events (source and replay time are separate):"]
    for event in report["events"]:
        lines.append(f"Replay {event['playback_time']:.2f}s | source {event['source_time']:.2f}s | frame {event['frame_index']} | {event['shot_id']} | {event['label']}")
    lines.extend(["", "Measurement limits:", *report["limitations"], ""])
    return "\n".join(lines)


def report_payloads(report: dict) -> dict[str, bytes]:
    return {"analysis.json": (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8"),
            "report.txt": report_text(report).encode("utf-8")}


def load_evidence(root: Path, run_dir: Path, shots: list[dict] | None = None) -> dict[str, dict]:
    """Read optional evidence only inside the project; absent exports stay unknown."""
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    run_dir.relative_to(root)
    locations = [(str(shot.get("shot_id", shot.get("id", index))), run_dir / "shots" / str(shot.get("shot_id", shot.get("id", index)))) for index, shot in enumerate(shots)] if shots else [(run_dir.name, run_dir)]
    result = {}
    for shot_id, directory in locations:
        directory = directory.resolve()
        directory.relative_to(root)
        values = {}
        for name in ("detections", "observations"):
            path = (directory / f"{name}.csv").resolve()
            path.relative_to(root)
            try:
                with path.open(encoding="utf-8-sig", newline="") as handle:
                    reader = csv.DictReader(handle)
                    values[name] = list(reader) if reader.fieldnames and "frame_index" in reader.fieldnames else None
            except (OSError, ValueError, csv.Error):
                values[name] = None
        result[shot_id] = values
    return result


def playback_seek(playback_time: Any) -> dict:
    """Streamlit's native player floors numeric offsets to whole seconds.

    Expose that fallback explicitly and retain the exact event timestamp for
    annotated-frame inspection instead of claiming a fractional video seek.
    """
    target = _number(playback_time)
    if target is None or target < 0:
        raise ValueError("Replay target must be a finite nonnegative time.")
    whole = math.floor(target)
    return {"target_seconds": target, "player_start_seconds": whole,
            "preroll_seconds": round(target - whole, 6), "exact_video_seek": target == whole}
