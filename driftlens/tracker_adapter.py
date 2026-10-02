"""Preserve raw detection indices in the pinned Ultralytics tracking backend.

Ultralytics 8.3.203 creates subset-local indices inside init_track after dividing
detections by confidence. This adapter restores indices into the current raw
detector snapshot before the tracker associates cars and returns track results.
"""

from __future__ import annotations

import math
from typing import Any


def map_original_detection_indices(
    boxes: Any,
    confidences: Any,
    snapshot: list[dict],
    classes: Any = None,
) -> list[int]:
    """Map an exact filtered detector subset to unique raw snapshot rows.

    Coordinates and confidence must retain their original precision. Matching
    uses exact equality because both values originate from the same detector
    result, rather than from independently measured or Kalman-filtered boxes.
    """
    if len(boxes) != len(confidences) or classes is not None and len(classes) != len(boxes):
        raise ValueError("Filtered detection boxes and metadata have inconsistent lengths")
    source = []
    for row in snapshot:
        signature = tuple(float(row[key]) for key in ("x1", "y1", "x2", "y2", "confidence"))
        if not all(math.isfinite(value) for value in signature):
            raise ValueError("Raw detector snapshot contains nonfinite values")
        source.append((signature, float(row["class"]) if row.get("class") is not None else None))
    result, used = [], set()
    for index, (box, confidence) in enumerate(zip(boxes, confidences)):
        if len(box) != 4:
            raise ValueError("Filtered detection box must contain four coordinates")
        signature = (*[float(coordinate) for coordinate in box], float(confidence))
        if not all(math.isfinite(value) for value in signature):
            raise ValueError("Filtered detection subset contains nonfinite values")
        class_id = float(classes[index]) if classes is not None else None
        matches = [
            source_index
            for source_index, (source_signature, source_class) in enumerate(source)
            if signature == source_signature
            and (source_class is None or class_id is None or source_class == class_id)
        ]
        if not matches:
            raise ValueError("A filtered detection has no exact match in the current raw snapshot")
        if len(matches) != 1:
            raise ValueError("A filtered detection ambiguously matches multiple raw snapshot rows")
        if matches[0] in used:
            raise ValueError("Filtered detections reuse one raw snapshot row")
        used.add(matches[0])
        result.append(matches[0])
    return result


def _install_adapter(tracker: Any, snapshot: list[dict]) -> None:
    tracker._driftlens_raw_snapshot = snapshot
    if hasattr(tracker, "_driftlens_original_init_track"):
        return
    tracker._driftlens_original_init_track = tracker.init_track

    def init_track_with_original_indices(results: Any, img: Any = None):
        indices = map_original_detection_indices(
            results.xyxy, results.conf, tracker._driftlens_raw_snapshot, results.cls
        )
        tracks = tracker._driftlens_original_init_track(results, img)
        if len(tracks) != len(indices):
            raise ValueError("Pinned tracker created a different number of tracks than detections")
        for track, original_index in zip(tracks, indices):
            track.idx = original_index
        return tracks

    tracker.init_track = init_track_with_original_indices


def bind_adapter(predictor: Any, snapshot: list[dict]) -> None:
    """Bind from raw capture after initialisation and before tracker updates.

    Register raw capture as an on_predict_postprocess_end callback before the
    tracking postprocess callback. Fill snapshot, then call this helper. Normal
    model.track startup has already created tracker instances at that point.
    Repeated binding is safe. The snapshot stays mutable between frames.
    This supports the pinned BYTETracker and BOTSORT instances only.
    """
    if not isinstance(snapshot, list):
        raise TypeError("Raw detector snapshot must be a mutable list")
    trackers = getattr(predictor, "trackers", None)
    if trackers is None:
        raise ValueError("Tracking must be initialised before binding its index adapter")
    for tracker in trackers:
        if type(tracker).__name__ not in {"BYTETracker", "BOTSORT"} or not type(tracker).__module__.startswith("ultralytics.trackers."):
            raise TypeError("The index adapter only supports pinned Ultralytics ByteTrack and BoTSORT")
        _install_adapter(tracker, snapshot)
