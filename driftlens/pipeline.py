"""CPU detection, tracking, honest observations and browser compatible replay."""
from __future__ import annotations

import csv
import hashlib
from importlib.metadata import distribution
import json
import math
import os
import subprocess
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from .review import derive_frame_metrics, write_metrics

ROOT = Path(__file__).resolve().parents[1]
OBSERVATION_FIELDS = ["clip_id", "frame_index", "clip_time", "source_time", "shot_index", "track_id", "x1", "y1", "x2", "y2", "confidence", "observed", "class", "source_class_id", "vehicle_anchor_class", "appearance_similarity", "detection_method", "recovery_anchor_track_id"]
FRAME_FIELDS = ["frame_index", "clip_time", "source_time", "shot_index"]
DETECTOR_MODELS = {"yolov8n.pt", "yolov8s.pt"}
IMAGE_SIZES = {320, 416, 512, 640, 960, 1280}


def validate_inference_profile(model_name: str, confidence_threshold: float, tracker_options: dict | None) -> dict:
    """Keep optional profiles local, explicit and free of implicit downloads."""
    if model_name not in DETECTOR_MODELS:
        raise ValueError("Choose installed pretrained yolov8n.pt or yolov8s.pt weights.")
    if isinstance(confidence_threshold, bool) or not isinstance(confidence_threshold, (int, float)) or not math.isfinite(confidence_threshold) or not 0.01 <= confidence_threshold <= 0.99:
        raise ValueError("Detector confidence must be a finite number between 0.01 and 0.99.")
    if tracker_options is None:
        return {}
    if not isinstance(tracker_options, dict):
        raise ValueError("Tracker options must be a dictionary.")
    numeric = {"track_high_thresh", "track_low_thresh", "new_track_thresh", "match_thresh", "proximity_thresh", "appearance_thresh"}
    flags = {"fuse_score", "with_reid"}
    allowed = numeric | flags | {"track_buffer", "gmc_method", "model"}
    if set(tracker_options) - allowed:
        raise ValueError("Unsupported tracker option: " + ", ".join(sorted(set(tracker_options) - allowed)))
    for key, value in tracker_options.items():
        if key in numeric and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"Tracker {key} must be a finite number between 0 and 1.")
        if key in flags and not isinstance(value, bool):
            raise ValueError(f"Tracker {key} must be a boolean.")
        if key == "track_buffer" and (isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 300):
            raise ValueError("Tracker track_buffer must be an integer between 1 and 300.")
        if key == "gmc_method" and value not in {"none", "sparseOptFlow", "orb"}:
            raise ValueError("Choose none, sparseOptFlow or orb camera motion compensation.")
        if key == "model" and value != "auto":
            raise ValueError("Appearance matching only supports native detector features (model auto).")
    return dict(tracker_options)


def resolve_tracker_settings(tracker: str, options: dict) -> dict:
    """Snapshot the pinned backend defaults plus validated explicit overrides."""
    import yaml
    if tracker not in {"bytetrack", "botsort"}:
        raise ValueError("Supported trackers are bytetrack and botsort.")
    path = distribution("ultralytics").locate_file(f"ultralytics/cfg/trackers/{tracker}.yaml")
    settings = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if set(options) - set(settings):
        raise ValueError("These tracker options do not apply to the selected backend.")
    settings.update(options)
    if settings["track_low_thresh"] > settings["track_high_thresh"]:
        raise ValueError("Tracker low confidence threshold must not exceed its high threshold.")
    return settings


def validate_orientations(orientations, tracker_settings: dict) -> list[int]:
    """Bound orientation augmentation and retain the normal full-frame pass."""
    if orientations is None:
        return [0]
    if not isinstance(orientations, (list, tuple)) or not 1 <= len(orientations) <= 3 or any(isinstance(value, bool) or value not in {0, 90, 180, 270} for value in orientations) or orientations[0] != 0 or len(set(orientations)) != len(orientations):
        raise ValueError("Orientations must start with 0 and contain at most three unique right-angle rotations.")
    if len(orientations) > 1 and tracker_settings.get("with_reid"):
        raise ValueError("Native appearance features cannot be combined across orientation passes.")
    return list(orientations)


def restore_rotated_box(box, width: int, height: int, degrees: int) -> list[float]:
    """Map one actual rotated-image detection back to original source pixels."""
    x1, y1, x2, y2 = map(float, box)
    if degrees == 0:
        return [x1, y1, x2, y2]
    if degrees == 90:
        return [y1, height - x2, y2, height - x1]
    if degrees == 180:
        return [width - x2, height - y2, width - x1, height - y1]
    if degrees == 270:
        return [width - y2, x1, width - y1, x2]
    raise ValueError("Rotation must be 0, 90, 180 or 270 degrees.")


def _suppression_indices(rows: np.ndarray, iou: float, agnostic_nms: bool) -> list[int]:
    """Keep original payload indices through deterministic observed-box NMS."""
    rows = np.asarray(rows, dtype=np.float32).reshape(-1, 6)
    if not len(rows):
        return []
    order = np.argsort(-rows[:, 4], kind="stable")
    kept = []
    while len(order):
        selected = int(order[0])
        kept.append(selected)
        others = order[1:]
        if not len(others):
            break
        first, rest = rows[selected], rows[others]
        intersection = np.maximum(0, np.minimum(first[2], rest[:, 2]) - np.maximum(first[0], rest[:, 0])) * np.maximum(0, np.minimum(first[3], rest[:, 3]) - np.maximum(first[1], rest[:, 1]))
        first_area = (first[2] - first[0]) * (first[3] - first[1])
        rest_area = (rest[:, 2] - rest[:, 0]) * (rest[:, 3] - rest[:, 1])
        overlap = intersection / np.maximum(first_area + rest_area - intersection, 1e-9)
        suppressed = overlap > iou
        if not agnostic_nms:
            suppressed &= rest[:, 5] == first[5]
        order = others[~suppressed]
    return kept


def suppress_duplicate_detections(rows: np.ndarray, iou: float, agnostic_nms: bool) -> np.ndarray:
    """Merge overlapping observed boxes; never emit a predicted position."""
    rows = np.asarray(rows, dtype=np.float32).reshape(-1, 6)
    return rows[_suppression_indices(rows, iou, agnostic_nms)]


def select_observed_detections(candidates: list[dict], iou: float, agnostic_nms: bool) -> list[dict]:
    """Carry chosen provenance by NMS index, using the tracker's float32 values.

    Inverse rotation arithmetic can produce doubles that are rounded when the
    backend receives its tensor. Canonicalize once and copy those same values
    into the raw snapshot; never match independently rounded box signatures.
    """
    keys = ("x1", "y1", "x2", "y2", "confidence", "class")
    array = np.asarray([[row[key] for key in keys] for row in candidates], dtype=np.float32).reshape(-1, 6)
    kept = _suppression_indices(array, iou, agnostic_nms)
    result = [{**candidates[index], **dict(zip(keys, map(float, array[index])))} for index in kept]
    signatures = [tuple(row[key] for key in keys) for row in result]
    if len(signatures) != len(set(signatures)):
        raise RuntimeError("Selected detector boxes have ambiguous duplicate associations.")
    for row in result:
        row["class"] = int(row["class"])
    return result


def _box_geometry(row) -> np.ndarray:
    return np.array([float(row[key]) for key in ("x1", "y1", "x2", "y2")])


def _box_overlap(first, second) -> float:
    a, b = _box_geometry(first), _box_geometry(second)
    intersection = np.prod(np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])))
    return float(intersection / max(np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - intersection, 1e-9))


def _appearance(frame: np.ndarray, row) -> np.ndarray | None:
    """Use chromatic source pixels; grey road and smoke provide no identity cue."""
    box = _box_geometry(row)
    if not np.isfinite(box).all() or np.any(box[2:] - box[:2] < 8):
        return None
    x1, y1, x2, y2 = np.rint(box).astype(int)
    crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
    if not crop.size:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] >= 60) & (hsv[:, :, 2] >= 40)).astype(np.uint8) * 255
    if np.count_nonzero(mask) < max(40, 0.08 * mask.size):
        return None
    histogram = cv2.calcHist([hsv], [0, 1], mask, [24, 16], [0, 180, 0, 256]).reshape(-1)
    return histogram / histogram.sum()


class VehicleClassRecovery:
    """Conservative current-box recovery seeded only by confirmed vehicles.

    A recovered observation can extend the spatial anchor. Its appearance never
    replaces the fingerprint from a genuine class 2/7 confirmation. This cannot
    create a box, bridge a missed sample, or reconnect identities across cuts.
    """
    max_age_seconds = 0.5
    min_appearance_similarity = 0.65

    def __init__(self):
        self.anchors = {}

    def reset(self):
        self.anchors.clear()

    def recover(self, frame, native_rows, candidates, timestamp):
        self.anchors = {key: value for key, value in self.anchors.items() if 0 <= timestamp - value["last_seen"] <= self.max_age_seconds + 1e-8}
        possible = []
        for index, candidate in enumerate(candidates):
            if int(candidate["class"]) in {2, 7} or float(candidate["confidence"]) < 0.15:
                continue
            if any(_box_overlap(candidate, row) > 0.3 for row in native_rows):
                continue
            appearance = _appearance(frame, candidate)
            if appearance is None:
                continue
            size = _box_geometry(candidate)[2:] - _box_geometry(candidate)[:2]
            if np.any(size <= 0):
                continue
            for identity, anchor in self.anchors.items():
                previous_size = _box_geometry(anchor["box"])[2:] - _box_geometry(anchor["box"])[:2]
                area_ratio = float(np.prod(size) / np.prod(previous_size))
                shape_ratio = float((size[0] / size[1]) / (previous_size[0] / previous_size[1]))
                overlap = _box_overlap(candidate, anchor["box"])
                similarity = float(np.minimum(appearance, anchor["appearance"]).sum())
                if 0.65 <= area_ratio <= 1.45 and 0.75 <= shape_ratio <= 1.33 and overlap >= 0.25 and similarity >= self.min_appearance_similarity:
                    possible.append((index, identity, similarity))
        # Ambiguous candidates and anchors are excluded, rather than ranked.
        candidate_counts = {index: sum(item[0] == index for item in possible) for index, _, _ in possible}
        anchor_counts = {identity: sum(item[1] == identity for item in possible) for _, identity, _ in possible}
        recovered = []
        for index, identity, similarity in possible:
            if candidate_counts[index] != 1 or anchor_counts[identity] != 1:
                continue
            candidate = candidates[index]
            recovered.append({**candidate, "source_class_id": int(candidate["class"]), "class": self.anchors[identity]["vehicle_class"], "vehicle_anchor_class": self.anchors[identity]["vehicle_class"], "appearance_similarity": similarity, "detection_method": "temporal_appearance_class_recovery", "recovery_anchor_track_id": identity})
        return recovered

    def observe(self, frame, observations, timestamp):
        native = [row for row in observations if row.get("detection_method") != "temporal_appearance_class_recovery" and int(row["class"]) in {2, 7}]
        for row in observations:
            if row.get("detection_method") == "temporal_appearance_class_recovery":
                identity = int(row["recovery_anchor_track_id"])
                if identity in self.anchors:
                    self.anchors[identity].update(box=dict(row), last_seen=timestamp)
                continue
            if any(other is not row and _box_overlap(row, other) > 0.35 for other in native):
                continue
            identity = int(row["track_id"])
            if identity in self.anchors:
                previous = self.anchors[identity]["box"]
                size, previous_size = _box_geometry(row)[2:] - _box_geometry(row)[:2], _box_geometry(previous)[2:] - _box_geometry(previous)[:2]
                ratio = float(np.prod(size) / max(np.prod(previous_size), 1e-9))
                if not 0.5 <= ratio <= 1.8:
                    # A sudden oversized box can span both overlapping cars.
                    continue
                self.anchors[identity].update(box=dict(row), last_seen=timestamp)
            if float(row["confidence"]) < 0.25 or int(row["class"]) not in {2, 7}:
                continue
            appearance = _appearance(frame, row)
            if appearance is not None:
                self.anchors[identity] = {"box": dict(row), "appearance": appearance, "vehicle_class": int(row["class"]), "last_seen": timestamp}


def video_info(source: Path) -> dict:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError("Could not open the selected video.")
    info = {"source_fps": capture.get(cv2.CAP_PROP_FPS), "total_frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    capture.release()
    if info["source_fps"] <= 0:
        raise ValueError("The source does not provide a valid frame rate.")
    info["duration_seconds"] = info["total_frames"] / info["source_fps"]
    return info


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _camera_cut(previous: np.ndarray | None, current: np.ndarray) -> bool:
    """Conservative scene cut safeguard, not a camera identity estimator."""
    if previous is None:
        return False
    difference = float(np.mean(cv2.absdiff(previous, current))) / 255
    first = cv2.calcHist([previous], [0], None, [32], [0, 256])
    second = cv2.calcHist([current], [0], None, [32], [0, 256])
    correlation = cv2.compareHist(first, second, cv2.HISTCMP_CORREL)
    return difference > 0.20 and correlation < 0.65


def _notify(callback, fraction: float, message: str) -> None:
    if callback is not None:
        callback(float(fraction), message)


def analyze_video(source: Path, output_dir: Path, start_seconds: float, end_seconds: float, tracker: str = "bytetrack", imgsz: int = 416, target_fps: float = 10, progress_callback=None, agnostic_nms: bool = False, model_name: str = "yolov8n.pt", tracker_options: dict | None = None, confidence_threshold: float = 0.15, orientations=None, recover_vehicle_classes: bool = False) -> dict:
    """Record failures from inference, decoding and final exports consistently."""
    try:
        return _analyze_video(source, output_dir, start_seconds, end_seconds, tracker, imgsz, target_fps, progress_callback, agnostic_nms, model_name, tracker_options, confidence_threshold, orientations, recover_vehicle_classes)
    except Exception as error:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.json").write_text(json.dumps({"status": "failed", "clip_id": output_dir.name, "error": str(error)}, indent=2), encoding="utf-8")
        raise


def _analyze_video(source: Path, output_dir: Path, start_seconds: float, end_seconds: float, tracker: str = "bytetrack", imgsz: int = 416, target_fps: float = 10, progress_callback=None, agnostic_nms: bool = False, model_name: str = "yolov8n.pt", tracker_options: dict | None = None, confidence_threshold: float = 0.15, orientations=None, recover_vehicle_classes: bool = False) -> dict:
    """Process a bounded clip; trackers only emit observations, never hidden cars."""
    import torch
    (ROOT / ".settings").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".settings"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "matplotlib_cache"))
    from ultralytics import YOLO

    source, output_dir = Path(source).resolve(), Path(output_dir).resolve()
    info = video_info(source)
    start_seconds, end_seconds = float(start_seconds), float(end_seconds)
    if start_seconds < 0 or end_seconds <= start_seconds or end_seconds > info["duration_seconds"] + 0.05:
        raise ValueError("Choose a valid start and end within the source video.")
    if end_seconds - start_seconds > 60:
        raise ValueError("Process at most 60 seconds at once on this computer.")
    if tracker not in {"bytetrack", "botsort"}:
        raise ValueError("Supported trackers are bytetrack and botsort.")
    if not 2 <= target_fps <= 30:
        raise ValueError("Sample rate must be between 2 and 30 frames per second.")
    if imgsz not in IMAGE_SIZES:
        raise ValueError("Choose an image size of 320, 416, 512, 640, 960 or 1280.")
    tracker_options = validate_inference_profile(model_name, confidence_threshold, tracker_options)
    tracker_settings = resolve_tracker_settings(tracker, tracker_options)
    orientations = validate_orientations(orientations, tracker_settings)
    if not isinstance(recover_vehicle_classes, bool):
        raise ValueError("Vehicle class recovery must be a boolean.")
    if recover_vehicle_classes and tracker_settings.get("with_reid"):
        raise ValueError("Native appearance features cannot be combined with class recovery.")
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    weights = ROOT / "models" / model_name
    if not weights.exists():
        raise FileNotFoundError("Detector weights are missing. Run setup.ps1 first.")
    output_dir.mkdir(parents=True, exist_ok=True)
    tracker_config = f"{tracker}.yaml"
    if tracker_options:
        tracker_config = str(output_dir / "tracker.yaml")
        # JSON is a YAML subset; this retains typed values without custom tags.
        Path(tracker_config).write_text(json.dumps(tracker_settings, indent=2) + "\n", encoding="utf-8")
    clip_id = output_dir.parent.parent.name if output_dir.parent.name == "comparisons" else output_dir.name
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps({"status": "processing", "clip_id": output_dir.name}), encoding="utf-8")
    model = YOLO(str(weights))
    augmentation_model = YOLO(str(weights)) if len(orientations) > 1 else None
    from .tracker_adapter import bind_adapter
    raw_detections = []
    extra_detections = []
    current_candidates = []
    recovery = VehicleClassRecovery() if recover_vehicle_classes else None
    current_frame = None
    current_time = None
    def capture_detections(predictor):
        raw_detections.clear()
        boxes = predictor.results[0].boxes
        original = boxes.data.cpu().numpy() if boxes is not None else np.empty((0, 6), dtype=np.float32)
        current_candidates.clear()
        for entry in original:
            current_candidates.append({**dict(zip(("x1", "y1", "x2", "y2"), map(float, entry[:4]))), "confidence": float(entry[4]), "class": int(entry[5]), "source_class_id": int(entry[5]), "detection_method": "native_detector"})
        for entry in extra_detections:
            current_candidates.append({**dict(zip(("x1", "y1", "x2", "y2"), map(float, entry[:4]))), "confidence": float(entry[4]), "class": int(entry[5]), "source_class_id": int(entry[5]), "detection_method": "orientation_detector"})
        selected = [row for row in current_candidates if row["class"] in {2, 7}]
        if recovery is not None:
            selected.extend(recovery.recover(current_frame, selected, current_candidates, current_time))
        if extra_detections or recovery is not None:
            selected = select_observed_detections(selected, 0.5, agnostic_nms)
            array = np.asarray([[row[key] for key in ("x1", "y1", "x2", "y2", "confidence", "class")] for row in selected], dtype=np.float32).reshape(-1, 6)
            predictor.results[0].update(boxes=torch.as_tensor(array))
        raw_detections.extend(dict(row) for row in selected)
        # The automatic on_predict_start callback has created the trackers.
        # Bind before their postprocessing update, using this frame's raw boxes.
        bind_adapter(predictor, raw_detections)
    # Added before register_tracker so the raw detector output is copied first.
    model.add_callback("on_predict_postprocess_end", capture_detections)
    stride = max(1, round(info["source_fps"] / target_fps))
    sampled_fps = info["source_fps"] / stride
    first_index = round(start_seconds * info["source_fps"])
    last_index = min(info["total_frames"], round(end_seconds * info["source_fps"]))
    capture = cv2.VideoCapture(str(source))
    capture.set(cv2.CAP_PROP_POS_FRAMES, first_index)
    observations, detections, frames, candidates = [], [], [], []
    previous, shot, track_offset = None, 0, 0
    started = time.perf_counter()
    try:
        for source_index in range(first_index, last_index):
            success, frame = capture.read()
            if not success:
                raise RuntimeError(f"Source decoding ended early at frame {source_index}; expected {last_index}.")
            if (source_index - first_index) % stride:
                continue
            small = cv2.cvtColor(cv2.resize(frame, (96, 54)), cv2.COLOR_BGR2GRAY)
            if _camera_cut(previous, small):
                shot += 1
                track_offset = shot * 10000
                if hasattr(model.predictor, "trackers"):
                    for state in model.predictor.trackers:
                        state.reset()
                if recovery is not None:
                    recovery.reset()
            previous = small
            frame_index = len(frames)
            frame_record = {"frame_index": frame_index, "clip_time": round((source_index - first_index) / info["source_fps"], 6), "source_time": round(source_index / info["source_fps"], 6), "shot_index": shot}
            frames.append(frame_record)
            current_frame, current_time = frame, float(frame_record["clip_time"])
            extra_detections.clear()
            for degrees in orientations[1:]:
                rotation = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[degrees]
                augmented = augmentation_model.predict(cv2.rotate(frame, rotation), classes=[2, 7], conf=confidence_threshold, iou=0.5, agnostic_nms=agnostic_nms, imgsz=imgsz, device="cpu", verbose=False)[0]
                if augmented.boxes is not None:
                    for box, confidence, class_id in zip(augmented.boxes.xyxy.cpu().tolist(), augmented.boxes.conf.cpu().tolist(), augmented.boxes.cls.cpu().tolist()):
                        extra_detections.append([*restore_rotated_box(box, info["width"], info["height"], degrees), float(confidence), int(class_id)])
            result = model.track(frame, persist=True, tracker=tracker_config, classes=None if recover_vehicle_classes else [2, 7], conf=confidence_threshold, iou=0.5, agnostic_nms=agnostic_nms, imgsz=imgsz, device="cpu", verbose=False)[0]
            boxes = result.boxes
            for raw in raw_detections:
                detections.append({"clip_id": clip_id, **frame_record, "track_id": "", **raw, "observed": True})
            if recover_vehicle_classes:
                candidates.extend({"clip_id": clip_id, **frame_record, "track_id": "", **row, "observed": True} for row in current_candidates)
            current_observations = []
            if boxes is not None and boxes.id is not None:
                association = {int(state.track_id): int(state.idx) for state in model.predictor.trackers[0].tracked_stracks if state.is_activated}
                for identifier in boxes.id.int().cpu().tolist():
                    raw_index = association.get(int(identifier))
                    if raw_index is None or not 0 <= raw_index < len(raw_detections):
                        raise RuntimeError("Tracker association does not map to a current detection.")
                    observation = {"clip_id": clip_id, **frame_record, "track_id": int(identifier) + track_offset, **raw_detections[raw_index], "observed": True}
                    observations.append(observation)
                    current_observations.append(observation)
            if recovery is not None:
                recovery.observe(frame, current_observations, current_time)
            if frame_index % 10 == 0:
                _notify(progress_callback, (source_index - first_index) / max(1, last_index - first_index), f"Tracking frame {frame_index + 1}")
    except Exception as error:
        summary_path.write_text(json.dumps({"status": "failed", "error": str(error), "clip_id": output_dir.name}), encoding="utf-8")
        raise
    finally:
        capture.release()
    elapsed = time.perf_counter() - started
    if not frames:
        raise ValueError("No frames were decoded in that interval.")
    _write_csv(output_dir / "observations.csv", OBSERVATION_FIELDS, observations)
    _write_csv(output_dir / "detections.csv", OBSERVATION_FIELDS, detections)
    _write_csv(output_dir / "frames.csv", FRAME_FIELDS, frames)
    if recover_vehicle_classes:
        _write_csv(output_dir / "candidates.csv", OBSERVATION_FIELDS, candidates)
    metrics = derive_frame_metrics(observations, frames, None, None)
    write_metrics(output_dir / "frame_metrics.csv", metrics)
    summary = {"clip_id": output_dir.name, "source_path": str(source), "start_seconds": start_seconds, "end_seconds": end_seconds, "duration_seconds": end_seconds - start_seconds, **info, "sampled_fps": sampled_fps, "frame_count": len(frames), "processing_seconds": round(elapsed, 3), "processing_fps": round(len(frames) / elapsed, 3), "tracker": tracker, "imgsz": imgsz, "model": model_name, "confidence_threshold": confidence_threshold, "track_ids": sorted({row["track_id"] for row in observations}), "lead_id": None, "chase_id": None, "observed_frames": len({row["frame_index"] for row in observations}), "paired_frames": 0, "pair_coverage": 0, "shot_count": shot + 1, "status": "complete", "files": {"video": "annotated.mp4", "observations": "observations.csv", "metrics": "frame_metrics.csv", "frames": "frames.csv"}}
    # Preserve clip duration rather than overwrite with the whole source duration.
    summary["duration_seconds"] = end_seconds - start_seconds
    summary["clip_id"] = clip_id
    summary["files"]["detections"] = "detections.csv"
    summary["pipeline_revision"] = 2
    summary["agnostic_nms"] = bool(agnostic_nms)
    summary["detector_classes"] = [2, 7]
    summary["nms_iou"] = 0.5
    summary["orientations"] = orientations
    summary["detector_passes_per_frame"] = len(orientations)
    summary["recover_vehicle_classes"] = recover_vehicle_classes
    summary["recovered_observations"] = sum(row.get("detection_method") == "temporal_appearance_class_recovery" for row in observations)
    summary["recovery_settings"] = {"max_age_seconds": VehicleClassRecovery.max_age_seconds, "min_appearance_similarity": VehicleClassRecovery.min_appearance_similarity, "min_seed_confidence": 0.25, "min_box_iou": 0.25, "area_ratio": [0.65, 1.45], "shape_ratio": [0.75, 1.33], "one_to_one_unambiguous": True} if recover_vehicle_classes else None
    summary["source_class_names"] = getattr(model, "names", {})
    if recover_vehicle_classes:
        summary["files"]["all_class_candidates"] = "candidates.csv"
        summary["recovery_confidence_note"] = "R marks a current observed box recovered by recent confirmed vehicle appearance and geometry. Confidence remains the original detector class score, not a car probability. Source class and recovery provenance are exported."
    summary["tracker_options"] = tracker_options
    summary["tracker_settings"] = tracker_settings
    summary["tracker_settings_sha256"] = hashlib.sha256(json.dumps(tracker_settings, sort_keys=True).encode("utf-8")).hexdigest()
    with weights.open("rb") as handle:
        summary["model_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    if tracker_options:
        summary["files"]["tracker_config"] = "tracker.yaml"
    render_run(output_dir, summary, observations, frames)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _notify(progress_callback, 1.0, "Replay and observations saved")
    return summary


def render_run(run_dir: Path, summary: dict, observations: list[dict], frames: list[dict]) -> None:
    """Render saved observations without rerunning the detector."""
    import imageio_ffmpeg
    from .review import roles_at
    capture = cv2.VideoCapture(str(summary["source_path"]))
    if not capture.isOpened():
        raise ValueError("The original video is needed to render role assignments.")
    width, height = int(summary["width"]), int(summary["height"])
    video_path = run_dir / "annotated.tmp.mp4"
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(summary["sampled_fps"]), "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video_path)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    by_frame = defaultdict(list)
    for row in observations:
        by_frame[int(row["frame_index"])].append(row)
    trajectories = defaultdict(lambda: deque(maxlen=20))
    previous_shot = None
    next_source_index = None
    try:
        for frame_record in frames:
            shot = int(frame_record["shot_index"])
            if shot != previous_shot:
                trajectories.clear()
            previous_shot = shot
            source_index = round(float(frame_record["source_time"]) * summary["source_fps"])
            if next_source_index is None or source_index < next_source_index:
                capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
                next_source_index = source_index
            while next_source_index < source_index:
                capture.grab()
                next_source_index += 1
            success, image = capture.read()
            next_source_index += 1
            if not success:
                raise RuntimeError("Could not decode an observed frame for replay.")
            frame_index = int(frame_record["frame_index"])
            lead_id, chase_id = roles_at(summary, float(frame_record["clip_time"]))
            if "role_intervals" in summary and frame_index > 0:
                previous = roles_at(summary, float(frames[frame_index - 1]["clip_time"]))
                if previous != (lead_id, chase_id):
                    trajectories.clear()
            for row in by_frame[frame_index]:
                identifier = int(row["track_id"])
                lead = lead_id is not None and identifier == int(lead_id)
                chase = chase_id is not None and identifier == int(chase_id)
                role = "LEAD" if lead else "CHASE" if chase else "CAR"
                color = (65, 215, 255) if lead else (230, 175, 40) if chase else (180, 180, 180)
                x1, y1, x2, y2 = [round(float(row[key])) for key in ("x1", "y1", "x2", "y2")]
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                text = f"{role}  ID {identifier}  {float(row['confidence']):.2f}"
                if row.get("detection_method") == "temporal_appearance_class_recovery":
                    text += f"  R class {row['source_class_id']}"
                cv2.putText(image, text, (max(4, x1), max(36, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                # Clear paths after a missed sample; don't bridge smoke gaps.
                history = trajectories[identifier]
                if history and history[-1][0] != frame_index - 1:
                    history.clear()
                history.append((frame_index, ((x1 + x2) // 2, (y1 + y2) // 2)))
                points = np.array([item[1] for item in history], dtype=np.int32)
                if len(points) >= 2:
                    cv2.polylines(image, [points], False, color, 2, cv2.LINE_AA)
            cv2.rectangle(image, (0, 0), (width, 30), (20, 22, 28), -1)
            heading = f"DRIFTLENS  |  {summary['tracker'].upper()}  |  {float(frame_record['clip_time']):.1f}s  |  OBSERVED IMAGE POSITIONS"
            if summary.get("full_run_id"):
                elapsed = float(summary.get("full_run_offset_seconds", 0)) + float(frame_record["clip_time"])
                heading = f"DRIFTLENS  |  {summary['full_run_shot'].upper()}  |  RUN {elapsed:.1f}s  |  {summary['tracker'].upper()}  |  IDs LOCAL TO THIS SHOT"
            cv2.putText(image, heading, (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
            if summary.get("lead_id") is not None or "role_intervals" in summary:
                seen = {int(row["track_id"]) for row in by_frame[frame_index]}
                absent = [role for role, identifier in (("LEAD", lead_id), ("CHASE", chase_id)) if identifier not in seen]
                if absent:
                    cv2.putText(image, "MISSING OR UNASSIGNED: " + ", ".join(absent), (14, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (75, 150, 255), 2, cv2.LINE_AA)
            process.stdin.write(image.tobytes())
        process.stdin.close()
        error = process.stderr.read().decode("utf-8", errors="replace")
        status = process.wait()
        if status:
            raise RuntimeError(f"Replay encoder failed: {error[-1000:]}")
        video_path.replace(run_dir / "annotated.mp4")
    finally:
        capture.release()
        if process.poll() is None:
            process.kill()

