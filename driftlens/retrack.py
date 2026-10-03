"""Rebuild cut boundaries and tracking from saved current detector candidates.

No model is loaded or inference rerun. Detector boxes, confidence, original
classes and source times remain the cached evidence. Roles require new review.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
import uuid

import cv2
import numpy as np

from .camera_cuts import CAMERA_CUT_SETTINGS, CameraCutDetector
from .pipeline import (FRAME_FIELDS, OBSERVATION_FIELDS, ROOT, VehicleClassRecovery,
                       _write_csv, render_run, select_observed_detections, video_info)
from .review import derive_frame_metrics, write_metrics


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def cached_inputs(run_dir: Path) -> tuple[dict, list[dict], dict[int, list[dict]]]:
    """Require complete pre-recovery snapshots before resetting associations."""
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8-sig"))
    if summary.get("status") != "complete":
        raise ValueError("Re-tracking needs a completed run.")
    if summary.get("tracker_settings", {}).get("with_reid"):
        raise ValueError("Cached CSV has no native detector appearance features.")
    frames = _rows(run_dir / "frames.csv")
    if len(frames) != int(summary["frame_count"]) or not frames:
        raise ValueError("Saved frame count does not match the completed run.")
    for index, row in enumerate(frames):
        if int(row["frame_index"]) != index:
            raise ValueError("Cached frame indices must be consecutive.")
        for key in ("clip_time", "source_time"):
            if not math.isfinite(float(row[key])):
                raise ValueError("Cached timestamps must be finite.")
        if index and float(row["source_time"]) <= float(frames[index-1]["source_time"]):
            raise ValueError("Cached source timestamps must increase.")
        clip_time = float(row["clip_time"])
        if not 0 <= clip_time < float(summary["duration_seconds"]) or abs(float(row["source_time"])-float(summary["start_seconds"])-clip_time) > 2e-6:
            raise ValueError("Cached source and clip times are not aligned.")
    grouped = defaultdict(list)
    for row in _rows(run_dir / "candidates.csv"):
        index = int(row["frame_index"])
        if not 0 <= index < len(frames):
            raise ValueError("Candidate references an unsampled frame.")
        if any(abs(float(row[key])-float(frames[index][key])) > 1e-8 for key in ("clip_time", "source_time")):
            raise ValueError("Candidate and sampled source timestamps disagree.")
        if row.get("detection_method") not in {"native_detector", "orientation_detector"}:
            raise ValueError("Re-tracking requires original detector candidates, before class recovery.")
        parsed = {**row, **{key: float(row[key]) for key in ("x1", "y1", "x2", "y2", "confidence")}}
        if not all(math.isfinite(parsed[key]) for key in ("x1", "y1", "x2", "y2", "confidence")):
            raise ValueError("Candidate geometry and confidence must be finite.")
        if parsed["x2"] <= parsed["x1"] or parsed["y2"] <= parsed["y1"] or not 0 <= parsed["confidence"] <= 1:
            raise ValueError("Candidate geometry or confidence is invalid.")
        parsed.update(class_=int(float(row["class"])))
        parsed["class"] = parsed.pop("class_")
        parsed["source_class_id"] = int(float(row.get("source_class_id") or row["class"]))
        if parsed["class"] != parsed["source_class_id"]:
            raise ValueError("Cached candidates contain reassigned detector classes.")
        grouped[index].append(parsed)
    return summary, frames, grouped


def retrack_cached_run(run_dir: Path, output_dir: Path, progress_callback=None) -> dict:
    """Create a new complete sibling result; preserve the original run untouched."""
    run_dir, output_dir = Path(run_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists() or output_dir == run_dir:
        raise ValueError("Choose a new result directory; the original cannot be overwritten.")
    summary, old_frames, grouped = cached_inputs(run_dir)
    source = Path(summary["source_path"]).resolve()
    info = video_info(source)
    if any(int(summary[key]) != int(info[key]) for key in ("width", "height", "total_frames")) or abs(float(summary["source_fps"])-info["source_fps"]) > 1e-8:
        raise ValueError("The source video no longer matches the saved run.")
    source_hash = _digest(source)
    expected = summary.get("uploaded_source", {}).get("sha256")
    if expected and source_hash != expected:
        raise ValueError("The uploaded source hash no longer matches.")
    settings = copy.deepcopy(summary.get("tracker_settings"))
    if not settings or settings.get("tracker_type") != summary.get("tracker"):
        raise ValueError("Saved resolved tracker settings are required.")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".settings"))
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.bot_sort import BOTSORT
    from ultralytics.trackers.byte_tracker import BYTETracker
    from .tracker_adapter import bind_adapter
    tracker_class = BOTSORT if summary["tracker"] == "botsort" else BYTETracker
    tracker = tracker_class(SimpleNamespace(**settings), frame_rate=30)
    raw = []
    bind_adapter(SimpleNamespace(trackers=[tracker]), raw)
    recovery = VehicleClassRecovery() if summary.get("recover_vehicle_classes") else None
    detector = CameraCutDetector()
    capture = cv2.VideoCapture(str(source))
    next_source_index = None
    frames, candidates, detections, observations, cuts = [], [], [], [], []
    shot = 0
    started = time.perf_counter()
    try:
        for index, saved in enumerate(old_frames):
            source_index = round(float(saved["source_time"])*float(summary["source_fps"]))
            if next_source_index is None:
                capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
                next_source_index = source_index
            while next_source_index < source_index:
                if not capture.grab():
                    raise RuntimeError("Source decoding stopped during cached re-tracking.")
                next_source_index += 1
            ok, image = capture.read()
            next_source_index += 1
            if not ok:
                raise RuntimeError("Source decoding stopped during cached re-tracking.")
            timestamp = float(saved["clip_time"])
            if detector.update(image):
                shot += 1
                cuts.append(timestamp)
                tracker.reset()
                if recovery is not None:
                    recovery.reset()
            record = {"frame_index": index, "clip_time": timestamp,
                      "source_time": float(saved["source_time"]), "shot_index": shot}
            frames.append(record)
            current = [{**row, "clip_id": output_dir.name, **record, "track_id": "", "observed": True}
                       for row in grouped[index]]
            candidates.extend(current)
            selected = [row for row in current if row["class"] in {2, 7}]
            if recovery is not None:
                selected.extend(recovery.recover(image, selected, current, timestamp))
            if len(summary.get("orientations", [0])) > 1 or recovery is not None:
                selected = select_observed_detections(selected, float(summary.get("nms_iou", .5)), bool(summary.get("agnostic_nms")))
            raw.clear()
            raw.extend(selected)
            detections.extend({**row, "track_id": ""} for row in raw)
            array = np.asarray([[row[key] for key in ("x1", "y1", "x2", "y2", "confidence", "class")]
                                for row in raw], dtype=np.float32).reshape(-1, 6)
            confirmed = tracker.update(Boxes(array, image.shape[:2]), image)
            current_observations = []
            for item in confirmed:
                raw_index = int(item[-1])
                if not 0 <= raw_index < len(raw):
                    raise RuntimeError("Tracker association is not a current cached detector box.")
                row = {**raw[raw_index], "track_id": int(item[4]) + shot*10000}
                observations.append(row)
                current_observations.append(row)
            if recovery is not None:
                recovery.observe(image, current_observations, timestamp)
            if progress_callback and index % 20 == 0:
                progress_callback(index/len(old_frames), f"Re-tracking saved candidates {index+1}/{len(old_frames)}")
    finally:
        capture.release()
    elapsed = time.perf_counter()-started
    revised = copy.deepcopy(summary)
    for key in ("role_intervals", "role_review_status", "role_assignment_method", "role_continuity", "role_continuation",
                "role_suggestions", "assistant_camera_review"):
        revised.pop(key, None)
    revised.update(clip_id=output_dir.name, lead_id=None, chase_id=None,
                   track_ids=sorted({int(row["track_id"]) for row in observations}),
                   observed_frames=len({int(row["frame_index"]) for row in observations}),
                   paired_frames=0, pair_coverage=0, shot_count=shot+1,
                   detected_camera_cuts_seconds=cuts, camera_cut_settings=dict(CAMERA_CUT_SETTINGS), visibility_revision=4,
                   role_review_status="unassigned_after_cached_retracking",
                   original_inference_processing_seconds=summary.get("original_inference_processing_seconds", summary.get("processing_seconds")),
                   processing_seconds=round(elapsed, 3),
                   processing_fps=round(len(frames)/max(elapsed, 1e-9), 3),
                   processing_mode="cached_detector_candidates_retracking_no_fresh_inference",
                   recovered_observations=sum(row.get("detection_method")=="temporal_appearance_class_recovery" for row in observations),
                   cached_retracking={"input_run_id": summary["clip_id"],
                                      "input_summary_sha256": _digest(run_dir/"summary.json"),
                                      "input_frames_sha256": _digest(run_dir/"frames.csv"),
                                      "input_candidates_sha256": _digest(run_dir/"candidates.csv"),
                                      "source_sha256": source_hash,
                                      "roles_cleared": True,
                                      "note": "Reused original detector boxes and confidence. Recomputed cut boundaries, class recovery and local tracking. No detector inference or model training."})
    revised["files"] = {**revised.get("files", {}), "video": "annotated.mp4", "frames": "frames.csv",
                        "observations": "observations.csv", "detections": "detections.csv",
                        "all_class_candidates": "candidates.csv", "metrics": "frame_metrics.csv",
                        "analysis": "analysis.json", "report": "report.txt", "summary": "summary.json"}
    revised["files"].pop("role_matches", None)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = output_dir.parent/f".cached_retracking_{uuid.uuid4().hex}"
    stage.mkdir()
    try:
        _write_csv(stage/"frames.csv", FRAME_FIELDS, frames)
        _write_csv(stage/"candidates.csv", OBSERVATION_FIELDS, candidates)
        _write_csv(stage/"detections.csv", OBSERVATION_FIELDS, detections)
        _write_csv(stage/"observations.csv", OBSERVATION_FIELDS, observations)
        write_metrics(stage/"frame_metrics.csv", derive_frame_metrics(observations, frames, None, None))
        (stage/"tracker.yaml").write_text(json.dumps(settings, indent=2)+"\n", encoding="utf-8")
        revised["files"]["tracker_config"] = "tracker.yaml"
        render_run(stage, revised, observations, frames)
        from .run_analysis_ui import save_shot_analysis
        revised = save_shot_analysis(stage, revised)
        (stage/"summary.json").write_text(json.dumps(revised, indent=2)+"\n", encoding="utf-8")
        if output_dir.exists():
            raise ValueError("Result directory appeared while re-tracking; refusing overwrite.")
        stage.rename(output_dir)
    finally:
        if stage.exists():
            resolved = stage.resolve()
            resolved.relative_to(output_dir.parent.resolve())
            if resolved.parent != output_dir.parent.resolve() or not resolved.name.startswith(".cached_retracking_"):
                raise ValueError("Cached re-tracking cleanup escaped its staging directory.")
            shutil.rmtree(stage)
    if progress_callback:
        progress_callback(1.0, "New cached re-tracking result saved; roles need review.")
    return revised
