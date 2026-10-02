"""Sparse, manually labelled detector and identity checks for real tandem clips.

This module deliberately does not report full MOT benchmark metrics. Sparse labels
cannot describe every identity transition, nor prove continuity inside smoke.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def intersection_over_union(first: list[float], second: list[float]) -> float:
    """Return IoU for two xyxy boxes in the same pixel coordinate system."""
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0


def match_boxes(
    ground_truth: list[list[float]], predictions: list[list[float]], min_iou: float = 0.5
) -> list[tuple[int, int, float]]:
    """Greedily match descending IoU, using each box at most once."""
    if not 0 < min_iou <= 1:
        raise ValueError("min_iou must be greater than zero and at most one")
    candidates = [
        (intersection_over_union(truth, prediction), truth_index, prediction_index)
        for truth_index, truth in enumerate(ground_truth)
        for prediction_index, prediction in enumerate(predictions)
    ]
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_truth, used_predictions = set(), set()
    matches = []
    for score, truth_index, prediction_index in candidates:
        if score < min_iou:
            break
        if truth_index in used_truth or prediction_index in used_predictions:
            continue
        matches.append((truth_index, prediction_index, score))
        used_truth.add(truth_index)
        used_predictions.add(prediction_index)
    return matches


def _number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _integer(value: Any, name: str) -> int:
    number = _number(value, name)
    if number != int(number):
        raise ValueError(f"{name} must be an integer")
    return int(number)


def _box(values: Any) -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError("bbox must contain four xyxy coordinates")
    result = [_number(value, "bbox coordinate") for value in values]
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("bbox must have positive width and height")
    return result


def _is_observed(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if str(value).strip().lower() in {"true", "1", "yes"}:
        return True
    if str(value).strip().lower() in {"false", "0", "no", "", "none"}:
        return False
    raise ValueError(f"Unknown observed flag: {value!r}")


def _track_id(value: Any) -> str | None:
    if value is None or str(value).strip().lower() in {"", "none", "nan"}:
        return None
    try:
        numeric = float(value)
        if math.isfinite(numeric) and numeric == int(numeric):
            return str(int(numeric))
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def evaluate_samples(
    annotations: dict[str, Any],
    observations: list[dict[str, Any]],
    frames: list[dict[str, Any]] | None = None,
    *,
    min_iou: float = 0.5,
    max_time_delta: float = 0.1,
) -> dict[str, Any]:
    """Compare sparse source pixel labels with actual processed frames.

    A frame manifest preserves frames with no detections. Without one, labels
    outside observed frames are unaligned and excluded, with an explicit count.
    A miss or wholly hidden label breaks the chain used to count ID switches.
    """
    if annotations.get("coordinate_space") != "source_pixels":
        raise ValueError("Annotations must declare coordinate_space: source_pixels")
    if not 0 < min_iou <= 1 or max_time_delta < 0:
        raise ValueError("Invalid matching threshold or time tolerance")
    samples = annotations.get("samples")
    if not isinstance(samples, list):
        raise ValueError("Annotations must contain a samples list")
    sample_clip_ids = {str(sample.get("clip_id", "")) for sample in samples}
    if "" in sample_clip_ids:
        raise ValueError("Every annotation sample must specify clip_id")
    default_clip_id = next(iter(sample_clip_ids)) if len(sample_clip_ids) == 1 else None

    def clip_id(row: dict[str, Any]) -> str:
        value = str(row.get("clip_id") or default_clip_id or "")
        if not value:
            raise ValueError("CSV clip_id is required when annotations span clips")
        return value

    frame_lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for row in frames if frames is not None else observations:
        key = (clip_id(row), _integer(row["frame_index"], "frame_index"))
        context = {
            "clip_id": key[0],
            "frame_index": key[1],
            "clip_time": _number(row["clip_time"], "clip_time"),
            "shot_index": _integer(row.get("shot_index", 0), "shot_index"),
        }
        if key in frame_lookup and frame_lookup[key] != context:
            raise ValueError(f"Inconsistent frame metadata for {key}")
        frame_lookup[key] = context
    rows_by_frame: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in observations:
        key = (clip_id(row), _integer(row["frame_index"], "frame_index"))
        if not _is_observed(row.get("observed", True)):
            continue
        candidate = {
            "bbox": _box([row[coordinate] for coordinate in ("x1", "y1", "x2", "y2")]),
            "track_id": _track_id(row.get("track_id")),
        }
        rows_by_frame.setdefault(key, []).append(candidate)

    counts = {
        "labelled_frames": len(samples), "aligned_frames": 0, "unaligned_frames": 0,
        "visible_ground_truth_cars": 0, "excluded_hidden_cars": 0,
        "true_positives": 0, "false_positives": 0, "false_negatives": 0,
        "matched_cars_with_track_id": 0, "comparable_identity_transitions": 0,
        "identity_switches": 0,
    }
    details, switch_events = [], []
    previous: dict[str, tuple[int, dict[str, str]]] = {}
    seen_sample_times: set[tuple[str, float]] = set()
    evaluated_frames: set[tuple[str, int]] = set()
    for sample in sorted(samples, key=lambda item: (str(item.get("clip_id", "")), _number(item.get("time_seconds"), "time_seconds"))):
        sample_clip = str(sample["clip_id"])
        time_seconds = _number(sample.get("time_seconds"), "time_seconds")
        if time_seconds < 0:
            raise ValueError("time_seconds cannot be negative")
        if (sample_clip, time_seconds) in seen_sample_times:
            raise ValueError("Duplicate annotation clip and time")
        seen_sample_times.add((sample_clip, time_seconds))
        cars = sample.get("cars")
        if not isinstance(cars, list):
            raise ValueError("Every annotation sample must contain a cars list")
        visible, identities = [], set()
        for car in cars:
            identity = str(car.get("identity", ""))
            if not identity or identity in identities:
                raise ValueError("Ground truth identities must be nonempty and unique per sample")
            identities.add(identity)
            visibility = car.get("visibility", "visible")
            if visibility == "hidden":
                counts["excluded_hidden_cars"] += 1
                continue
            if visibility not in {"visible", "partial"}:
                raise ValueError("visibility must be visible, partial, or hidden")
            visible.append({"identity": identity, "bbox": _box(car.get("bbox"))})

        candidates = [frame for key, frame in frame_lookup.items() if key[0] == sample_clip]
        if "frame_index" in sample:
            frame = frame_lookup.get((sample_clip, _integer(sample["frame_index"], "frame_index")))
        else:
            frame = min(candidates, key=lambda item: (abs(item["clip_time"] - time_seconds), item["frame_index"])) if candidates else None
        if frame is None or abs(frame["clip_time"] - time_seconds) > max_time_delta:
            counts["unaligned_frames"] += 1
            previous.pop(sample_clip, None)
            details.append({"clip_id": sample_clip, "time_seconds": time_seconds, "aligned": False})
            continue
        key = (sample_clip, frame["frame_index"])
        if key in evaluated_frames:
            raise ValueError("Multiple labels align to the same processed frame; choose distinct frames")
        evaluated_frames.add(key)
        if "shot_index" in sample and _integer(sample["shot_index"], "shot_index") != frame["shot_index"]:
            raise ValueError("Annotation shot_index disagrees with processed frame")
        predictions = rows_by_frame.get(key, [])
        matches = match_boxes([car["bbox"] for car in visible], [prediction["bbox"] for prediction in predictions], min_iou)
        counts["aligned_frames"] += 1
        counts["visible_ground_truth_cars"] += len(visible)
        counts["true_positives"] += len(matches)
        counts["false_positives"] += len(predictions) - len(matches)
        counts["false_negatives"] += len(visible) - len(matches)
        current: dict[str, str] = {}
        match_details = []
        for truth_index, prediction_index, score in matches:
            identity = visible[truth_index]["identity"]
            track = predictions[prediction_index]["track_id"]
            match_details.append({"identity": identity, "track_id": track, "iou": score})
            if track is not None:
                counts["matched_cars_with_track_id"] += 1
                current[identity] = track
        last_shot, last_ids = previous.get(sample_clip, (-1, {}))
        if last_shot == frame["shot_index"]:
            for identity, track in current.items():
                if identity not in last_ids:
                    continue
                counts["comparable_identity_transitions"] += 1
                if track != last_ids[identity]:
                    counts["identity_switches"] += 1
                    switch_events.append({"clip_id": sample_clip, "time_seconds": time_seconds, "shot_index": frame["shot_index"], "identity": identity, "previous_track_id": last_ids[identity], "track_id": track})
        previous[sample_clip] = (frame["shot_index"], current)
        details.append({"clip_id": sample_clip, "time_seconds": time_seconds, "aligned": True, "frame_index": frame["frame_index"], "sample_time_error_seconds": abs(frame["clip_time"] - time_seconds), "shot_index": frame["shot_index"], "visible_cars": len(visible), "predictions": len(predictions), "matches": match_details})

    true_positives, false_positives, false_negatives = (counts[name] for name in ("true_positives", "false_positives", "false_negatives"))
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else None
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else None
    f1_denominator = 2 * true_positives + false_positives + false_negatives
    return {
        "schema_version": 1,
        "evaluation_type": "sparse_manual_annotation",
        "matching": {"method": "descending_iou_greedy_one_to_one", "minimum_iou": min_iou, "maximum_time_error_seconds": max_time_delta, "coordinate_space": "source_pixels", "frame_manifest_used": frames is not None},
        "counts": counts,
        "detection": {"precision": precision, "recall": recall, "f1": 2 * true_positives / f1_denominator if f1_denominator else None},
        "tracking": {"visible_ground_truth_coverage": counts["matched_cars_with_track_id"] / counts["visible_ground_truth_cars"] if counts["visible_ground_truth_cars"] else None, "identity_switches": counts["identity_switches"], "comparable_identity_transitions": counts["comparable_identity_transitions"]},
        "switch_events": switch_events,
        "samples": details,
        "limitations": ["Sparse samples cannot measure full MOT metrics or every identity switch.", "No identity continuity is assumed after a labelled miss, hidden car, missing track ID, unaligned sample, or camera cut.", "Hidden cars are excluded from detection denominators. Partial visibility is included.", "Metrics cover only aligned annotated frames; unaligned frames require review."],
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sparse manual labels against DriftLens observations.")
    parser.add_argument("labels", type=Path)
    parser.add_argument("observations", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frames", type=Path, help="Frame manifest; defaults to adjacent frames.csv if available")
    parser.add_argument("--min-iou", type=float, default=0.5)
    parser.add_argument("--max-time-delta", type=float, default=0.1)
    arguments = parser.parse_args()
    with arguments.labels.open("r", encoding="utf-8-sig") as handle:
        annotations = json.load(handle)
    frames_path = arguments.frames or arguments.observations.with_name("frames.csv")
    if arguments.frames and not frames_path.is_file():
        parser.error("The specified frame manifest does not exist")
    frames = read_csv(frames_path) if frames_path.is_file() else None
    report = evaluate_samples(annotations, read_csv(arguments.observations), frames, min_iou=arguments.min_iou, max_time_delta=arguments.max_time_delta)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(arguments.output), "counts": report["counts"], "detection": report["detection"], "tracking": report["tracking"]}, indent=2))


if __name__ == "__main__":
    main()
