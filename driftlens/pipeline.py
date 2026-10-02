"""CPU detection, tracking, honest observations and browser compatible replay."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from .review import derive_frame_metrics, write_metrics

ROOT = Path(__file__).resolve().parents[1]
OBSERVATION_FIELDS = ["clip_id", "frame_index", "clip_time", "source_time", "shot_index", "track_id", "x1", "y1", "x2", "y2", "confidence", "observed"]
FRAME_FIELDS = ["frame_index", "clip_time", "source_time", "shot_index"]


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


def analyze_video(source: Path, output_dir: Path, start_seconds: float, end_seconds: float, tracker: str = "bytetrack", imgsz: int = 416, target_fps: float = 10, progress_callback=None, agnostic_nms: bool = False) -> dict:
    """Record failures from inference, decoding and final exports consistently."""
    try:
        return _analyze_video(source, output_dir, start_seconds, end_seconds, tracker, imgsz, target_fps, progress_callback, agnostic_nms)
    except Exception as error:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.json").write_text(json.dumps({"status": "failed", "clip_id": output_dir.name, "error": str(error)}, indent=2), encoding="utf-8")
        raise


def _analyze_video(source: Path, output_dir: Path, start_seconds: float, end_seconds: float, tracker: str = "bytetrack", imgsz: int = 416, target_fps: float = 10, progress_callback=None, agnostic_nms: bool = False) -> dict:
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
    if imgsz not in {320, 416, 512, 640}:
        raise ValueError("Choose an image size of 320, 416, 512 or 640.")
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    weights = ROOT / "models" / "yolov8n.pt"
    if not weights.exists():
        raise FileNotFoundError("Detector weights are missing. Run setup.ps1 first.")
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_id = output_dir.parent.parent.name if output_dir.parent.name == "comparisons" else output_dir.name
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps({"status": "processing", "clip_id": output_dir.name}), encoding="utf-8")
    model = YOLO(str(weights))
    from .tracker_adapter import bind_adapter
    raw_detections = []
    def capture_detections(predictor):
        raw_detections.clear()
        boxes = predictor.results[0].boxes
        if boxes is not None:
            for box, confidence in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist()):
                raw_detections.append({**{key: float(value) for key, value in zip(("x1", "y1", "x2", "y2"), box)}, "confidence": float(confidence)})
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
    observations, detections, frames = [], [], []
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
            previous = small
            frame_index = len(frames)
            frame_record = {"frame_index": frame_index, "clip_time": round((source_index - first_index) / info["source_fps"], 6), "source_time": round(source_index / info["source_fps"], 6), "shot_index": shot}
            frames.append(frame_record)
            result = model.track(frame, persist=True, tracker=f"{tracker}.yaml", classes=[2, 7], conf=0.15, iou=0.5, agnostic_nms=agnostic_nms, imgsz=imgsz, device="cpu", verbose=False)[0]
            boxes = result.boxes
            for raw in raw_detections:
                detections.append({"clip_id": clip_id, **frame_record, "track_id": "", **raw, "observed": True})
            if boxes is not None and boxes.id is not None:
                association = {int(state.track_id): int(state.idx) for state in model.predictor.trackers[0].tracked_stracks if state.is_activated}
                for identifier in boxes.id.int().cpu().tolist():
                    raw_index = association.get(int(identifier))
                    if raw_index is None or not 0 <= raw_index < len(raw_detections):
                        raise RuntimeError("Tracker association does not map to a current detection.")
                    observations.append({"clip_id": clip_id, **frame_record, "track_id": int(identifier) + track_offset, **raw_detections[raw_index], "observed": True})
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
    metrics = derive_frame_metrics(observations, frames, None, None)
    write_metrics(output_dir / "frame_metrics.csv", metrics)
    summary = {"clip_id": output_dir.name, "source_path": str(source), "start_seconds": start_seconds, "end_seconds": end_seconds, "duration_seconds": end_seconds - start_seconds, **info, "sampled_fps": sampled_fps, "frame_count": len(frames), "processing_seconds": round(elapsed, 3), "processing_fps": round(len(frames) / elapsed, 3), "tracker": tracker, "imgsz": imgsz, "model": "yolov8n.pt", "confidence_threshold": 0.15, "track_ids": sorted({row["track_id"] for row in observations}), "lead_id": None, "chase_id": None, "observed_frames": len({row["frame_index"] for row in observations}), "paired_frames": 0, "pair_coverage": 0, "shot_count": shot + 1, "status": "complete", "files": {"video": "annotated.mp4", "observations": "observations.csv", "metrics": "frame_metrics.csv", "frames": "frames.csv"}}
    # Preserve clip duration rather than overwrite with the whole source duration.
    summary["duration_seconds"] = end_seconds - start_seconds
    summary["clip_id"] = clip_id
    summary["files"]["detections"] = "detections.csv"
    summary["pipeline_revision"] = 2
    summary["agnostic_nms"] = bool(agnostic_nms)
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

