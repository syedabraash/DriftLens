"""Separate human review records from the frozen assistant reference labels.

Review is a named person's explicit attestation, never an expert certification.
This module performs no image decoding, model inference or training.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any
import uuid


CLASS_NAMES = {2: "car", 7: "truck"}
SPLITS = {"tune", "train", "val", "test"}
TAG_VALUES = {
    "smoke": {"clear", "light", "heavy", "unknown"},
    "overlap": {"none", "partial", "severe", "unknown"},
}
CONFIRMATIONS = ("image", "boxes", "identities")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _integer(value: Any, name: str) -> int:
    number = _number(value, name)
    if number != int(number) or number < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(number)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value.strip()


def _image_path(value: Any) -> str:
    if value in (None, ""):
        return ""
    value = _text(value, "image").replace("\\", "/")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or ":" in value:
        raise ValueError("image must be a relative path inside the project")
    return path.as_posix()


def validate_cars(cars: Any, width: int, height: int) -> list[dict]:
    """Validate visible source extents and unique identities within this shot."""
    if not isinstance(cars, list):
        raise ValueError("cars must be a list")
    result, identities = [], set()
    for car in cars:
        if not isinstance(car, dict):
            raise ValueError("Every car must be an object")
        identity = _text(car.get("identity"), "identity")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", identity):
            raise ValueError("Use a shot local identity such as car_a; global or tracker IDs are not accepted")
        if identity in identities:
            raise ValueError("Identities must be unique within each frame")
        identities.add(identity)
        class_id = _integer(car.get("class_id", 2), "class_id")
        if class_id not in CLASS_NAMES:
            raise ValueError("class_id must be 2 (car) or 7 (truck)")
        visibility = car.get("visibility", "visible")
        if not isinstance(visibility, str) or visibility not in {"visible", "partial", "hidden"}:
            raise ValueError("visibility must be visible, partial or hidden")
        bbox = car.get("bbox")
        if visibility == "hidden":
            if bbox is not None:
                raise ValueError("Hidden vehicles must not have invented bounding boxes")
        else:
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                raise ValueError("Visible and partial vehicles require four xyxy coordinates")
            bbox = [_number(coordinate, "bbox coordinate") for coordinate in bbox]
            if not (0 <= bbox[0] < bbox[2] <= width and 0 <= bbox[1] < bbox[3] <= height):
                raise ValueError("Bounding boxes must have positive area and stay inside the source image")
        result.append({"identity": identity, "class_id": class_id, "visibility": visibility, "bbox": bbox})
    return result


def _tags(tags: Any) -> dict:
    if not isinstance(tags, dict) or set(tags) != set(TAG_VALUES):
        raise ValueError("tags must contain smoke and overlap")
    for name, choices in TAG_VALUES.items():
        if not isinstance(tags[name], str) or tags[name] not in choices:
            raise ValueError(f"Invalid {name} tag")
    return dict(tags)


def _review_payload(record: dict) -> dict:
    return {key: value for key, value in record.items() if key not in {"review", "provenance"}}


def validate_review_queue(queue: dict) -> dict:
    """Reject malformed labels, stale confirmations and grouped split leakage."""
    if not isinstance(queue, dict) or queue.get("schema_version") != 1 or queue.get("kind") != "annotation_review":
        raise ValueError("Unsupported annotation review schema")
    if queue.get("coordinate_space") != "source_pixels":
        raise ValueError("Review labels must use source_pixels")
    if queue.get("split_grouping") not in {"battle", "video"}:
        raise ValueError("split_grouping must be battle or video")
    origin = queue.get("original_reference")
    if not isinstance(origin, dict):
        raise ValueError("Original reference provenance is required")
    _image_path(_text(origin.get("path"), "original reference path"))
    if not re.fullmatch(r"[a-f0-9]{64}", str(origin.get("sha256", ""))):
        raise ValueError("The frozen reference fingerprint is required")
    _text(origin.get("annotation_method"), "original annotation method")
    records = queue.get("records")
    if not isinstance(records, list):
        raise ValueError("records must be a list")
    seen, times, split_by_group = set(), set(), {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each review record must be an object")
        sample_id = _text(record.get("sample_id"), "sample_id")
        if sample_id in seen:
            raise ValueError("Duplicate sample_id")
        seen.add(sample_id)
        clip = _text(record.get("clip_id"), "clip_id")
        shot = _integer(record.get("shot_index"), "shot_index")
        if record.get("identity_scope") != f"{clip}:shot{shot}":
            raise ValueError("Identity scope must be local to this clip and camera shot")
        seconds = _number(record.get("time_seconds"), "time_seconds")
        if seconds < 0 or (clip, shot, seconds) in times:
            raise ValueError("Review times must be nonnegative and unique within each shot")
        times.add((clip, shot, seconds))
        if record.get("frame_index") is not None:
            _integer(record["frame_index"], "frame_index")
        width = _integer(record.get("width"), "width")
        height = _integer(record.get("height"), "height")
        if not width or not height:
            raise ValueError("Source image dimensions must be positive")
        _image_path(record.get("image"))
        video = _text(record.get("video_id"), "video_id")
        battle = _text(record.get("battle_group"), "battle_group")
        split = record.get("split")
        if not isinstance(split, str) or split not in SPLITS:
            raise ValueError("split must be tune, train, val or test")
        group = (video, battle) if queue["split_grouping"] == "battle" else (video,)
        if group in split_by_group and split_by_group[group] != split:
            raise ValueError("Split leakage: keep the complete selected video or battle group in one split")
        split_by_group[group] = split
        validate_cars(record.get("original_cars"), width, height)
        validate_cars(record.get("cars"), width, height)
        _tags(record.get("tags"))
        review = record.get("review")
        if review is None:
            if record.get("provenance") != "assistant_draft":
                raise ValueError("Pending assistant references cannot claim human review")
            continue
        if not isinstance(review, dict) or record.get("provenance") != "human_self_attested":
            raise ValueError("Reviewed records must preserve explicit human attestation provenance")
        _text(review.get("reviewer"), "reviewer")
        try:
            timestamp = datetime.fromisoformat(review["reviewed_at"].replace("Z", "+00:00"))
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise ValueError("Review timestamp must be an ISO timestamp with timezone") from error
        if timestamp.tzinfo is None:
            raise ValueError("Review timestamp must include a timezone")
        confirmations = review.get("confirmations")
        if not isinstance(confirmations, dict) or any(confirmations.get(name) is not True for name in CONFIRMATIONS):
            raise ValueError("Image, boxes and shot local identities require explicit confirmation")
        if review.get("payload_sha256") != _digest(_review_payload(record)):
            raise ValueError("Reviewed content changed; explicitly review and confirm it again")
    return queue


def build_review_queue(annotations: dict, catalog: dict, *, source_labels_path: str = "data/annotations/labels.json", group_by: str = "battle") -> dict:
    """Copy evaluator references into a separate pending queue without promotion."""
    if annotations.get("coordinate_space") != "source_pixels":
        raise ValueError("Annotations must declare source_pixels")
    width = _integer(annotations.get("width", annotations.get("image_width")), "width")
    height = _integer(annotations.get("height", annotations.get("image_height")), "height")
    source = deepcopy(catalog.get("source") or {})
    video = _text(source.get("video_id") or source.get("path") or source.get("url"), "source video ID")
    clips = {str(clip["id"]): clip for clip in catalog.get("clips", [])}
    records = []
    for sample in annotations.get("samples", []):
        clip_id = str(sample.get("clip_id", ""))
        if clip_id not in clips:
            raise ValueError(f"Reference clip {clip_id!r} is absent from the catalogue")
        clip = clips[clip_id]
        split = sample.get("split", clip.get("split"))
        if sample.get("split") and clip.get("split") and sample["split"] != clip["split"]:
            raise ValueError("Reference split disagrees with catalogue")
        shot = _integer(sample.get("shot_index", 0), "shot_index")
        seconds = _number(sample.get("time_seconds"), "time_seconds")
        cars = validate_cars(sample.get("cars"), width, height)
        records.append({"sample_id": f"{clip_id}_shot{shot}_{seconds:.6f}", "clip_id": clip_id,
                        "shot_index": shot, "identity_scope": f"{clip_id}:shot{shot}",
                        "time_seconds": seconds, "frame_index": sample.get("frame_index"),
                        "image": _image_path(sample.get("frame", sample.get("image", ""))),
                        "width": width, "height": height, "video_id": video,
                        "battle_group": _text(clip.get("run_group"), "battle group"), "split": split,
                        "original_cars": deepcopy(cars), "cars": cars,
                        "tags": {"smoke": "unknown", "overlap": "unknown"},
                        "provenance": "assistant_draft", "review": None})
    return validate_review_queue({"schema_version": 1, "kind": "annotation_review",
                                 "coordinate_space": "source_pixels", "source": source,
                                 "original_reference": {"path": source_labels_path, "sha256": _digest(annotations),
                                                        "annotation_method": annotations.get("annotation_method", "Assistant draft; no human expert review")},
                                 "split_grouping": group_by, "records": records})


def confirm_sample(queue: dict, sample_id: str, cars: list[dict], tags: dict, reviewer: str, confirmations: dict, *, reviewed_at: str | None = None) -> dict:
    """Save a deliberate human review while retaining the original draft."""
    validate_review_queue(queue)
    updated = deepcopy(queue)
    record = next((row for row in updated["records"] if row["sample_id"] == sample_id), None)
    if record is None:
        raise ValueError("Unknown sample_id")
    record["cars"] = validate_cars(cars, record["width"], record["height"])
    record["tags"] = _tags(tags)
    record["provenance"] = "human_self_attested"
    record["review"] = {"reviewer": _text(reviewer, "reviewer"),
                        "reviewed_at": reviewed_at or datetime.now(timezone.utc).isoformat(),
                        "confirmations": dict(confirmations), "payload_sha256": _digest(_review_payload(record))}
    return validate_review_queue(updated)


def save_review_queue(path: Path | str, queue: dict) -> None:
    """Publish a validated queue atomically; never use the frozen label path."""
    validate_review_queue(queue)
    path = Path(path)
    original_path = Path(queue["original_reference"]["path"])
    if path.as_posix().lower().endswith(original_path.as_posix().lower()):
        raise ValueError("The review queue cannot replace the frozen reference file")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(queue, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_review_queue(path: Path | str) -> dict:
    return validate_review_queue(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def import_review_queue(existing: dict, incoming: dict) -> dict:
    """Merge validated reviews only for known frames from the same frozen source."""
    validate_review_queue(existing)
    validate_review_queue(incoming)
    for key in ("schema_version", "kind", "source", "original_reference", "split_grouping"):
        if existing[key] != incoming[key]:
            raise ValueError("Imported review has different source provenance or split policy")
    merged = deepcopy(existing)
    by_id = {record["sample_id"]: record for record in merged["records"]}
    for record in incoming["records"]:
        previous = by_id.get(record["sample_id"])
        if previous is None:
            raise ValueError("Imported review contains an unknown frame")
        for key in set(previous) - {"cars", "tags", "review", "provenance"}:
            if previous[key] != record.get(key):
                raise ValueError("Imported review changed immutable frame or split context")
        if record["review"] is not None:
            if previous["review"] and datetime.fromisoformat(record["review"]["reviewed_at"].replace("Z", "+00:00")) < datetime.fromisoformat(previous["review"]["reviewed_at"].replace("Z", "+00:00")):
                raise ValueError("Imported review would replace a newer review")
            previous.update(deepcopy(record))
    return validate_review_queue(merged)


def export_reviewed_labels(queue: dict, split: str | None = None, *, target_class_ids: tuple[int, ...] = (2,)) -> dict:
    """Export confirmed source labels directly consumable by evaluate_samples."""
    validate_review_queue(queue)
    if split is not None and split not in SPLITS:
        raise ValueError("Unknown split")
    if not target_class_ids or not set(target_class_ids).issubset(CLASS_NAMES):
        raise ValueError("Choose supported target class IDs")
    selected = [record for record in queue["records"] if record["review"] and (split is None or record["split"] == split)]
    if not selected:
        raise ValueError("No human confirmed samples are available for this export")
    samples = []
    for record in selected:
        sample = {"clip_id": record["clip_id"], "time_seconds": record["time_seconds"],
                  "shot_index": record["shot_index"], "split": record["split"],
                  "identity_scope": record["identity_scope"], "video_id": record["video_id"],
                  "width": record["width"], "height": record["height"],
                  "run_group": record["battle_group"], "frame": record["image"],
                  "cars": [deepcopy(car) for car in record["cars"] if car["class_id"] in target_class_ids],
                  "tags": deepcopy(record["tags"]), "review": deepcopy(record["review"]),
                  "provenance": record["provenance"]}
        if record["frame_index"] is not None:
            sample["frame_index"] = record["frame_index"]
        samples.append(sample)
    return {"coordinate_space": "source_pixels", "annotation_method": "Named human self attestation of assistant draft corrections; no expert validation is inferred",
            "source": deepcopy(queue["source"]), "original_reference": deepcopy(queue["original_reference"]),
            "split_grouping": queue["split_grouping"], "target_class_ids": list(target_class_ids),
            "reviewed_frames": len(samples), "pending_frames": sum(record["review"] is None for record in queue["records"]),
            "samples": samples,
            "limitations": "Review attestations are self reported. Sparse samples remain a diagnostic set; reference identities are local to each camera shot."}


def export_detector_labels(queue: dict, split: str) -> dict:
    """Emit visible human confirmed boxes with grouping for later retraining."""
    exported = export_reviewed_labels(queue, split, target_class_ids=(2, 7))
    exported["class_names"] = dict(CLASS_NAMES)
    exported["samples"] = [{**sample, "id": f"{sample['clip_id']}_shot{sample['shot_index']}_{sample['time_seconds']:.6f}",
                            "boxes": [{"class_id": car["class_id"], "bbox": car["bbox"]}
                                      for car in sample["cars"] if car["visibility"] != "hidden"]}
                           for sample in exported["samples"]]
    exported["training_allowed"] = split in {"train", "val", "tune"}
    return exported
