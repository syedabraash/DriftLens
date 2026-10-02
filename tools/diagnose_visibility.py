"""Audit saved raw boxes, confirmed observations and matched original frames.

This is a demonstration diagnostic, not a held-out tracking benchmark. Source
images and detector derivatives stay in the private local output directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def group_frames(rows):
    result = defaultdict(list)
    for row in rows:
        result[int(row["frame_index"])].append(row)
    return result


def frame_at(capture, index):
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Cannot decode original source frame {index}")
    return frame


def replay_saved(frames, detections, source, source_fps, overrides):
    """Use identical saved boxes; this isolates tracking from detector changes."""
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.bot_sort import BOTSORT
    from ultralytics.utils import YAML
    from ultralytics.utils.checks import check_yaml
    from driftlens.tracker_adapter import bind_adapter
    settings = YAML.load(check_yaml("botsort.yaml"))
    settings.update(overrides)
    if settings["with_reid"]:
        raise ValueError("Saved CSV has no native appearance features")
    tracker = BOTSORT(SimpleNamespace(**settings), frame_rate=30)
    raw = []
    bind_adapter(SimpleNamespace(trackers=[tracker]), raw)
    capture = cv2.VideoCapture(str(source))
    result = defaultdict(list)
    previous_shot = None
    next_source_index = None
    started = time.perf_counter()
    try:
        for record in frames:
            frame_index = int(record["frame_index"])
            shot = int(record["shot_index"])
            if previous_shot is not None and shot != previous_shot:
                tracker.reset()
            previous_shot = shot
            raw.clear()
            raw.extend({**{key: float(row[key]) for key in ("x1", "y1", "x2", "y2", "confidence")}, "class": float(row.get("class") or 2)} for row in detections[frame_index])
            array = np.asarray([[row[key] for key in ("x1", "y1", "x2", "y2", "confidence", "class")] for row in raw], dtype=np.float32).reshape(-1, 6)
            source_index = round(float(record["source_time"]) * source_fps)
            if next_source_index is None or source_index < next_source_index:
                capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
                next_source_index = source_index
            while next_source_index < source_index:
                capture.grab()
                next_source_index += 1
            ok, frame = capture.read()
            next_source_index += 1
            if not ok:
                raise RuntimeError(f"Cannot decode original source frame {source_index}")
            confirmed = tracker.update(Boxes(array, frame.shape[:2]), frame)
            for row in confirmed:
                raw_index = int(row[-1])
                if not 0 <= raw_index < len(raw):
                    raise RuntimeError("Confirmed association is not a current raw box")
                result[frame_index].append({**raw[raw_index], "track_id": int(row[4])})
    finally:
        capture.release()
    return result, settings, time.perf_counter() - started


def annotate(frame, boxes, title):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font_path = Path(r"C:\Windows\Fonts\arial.ttf")
    font = ImageFont.truetype(str(font_path), 24) if font_path.exists() else ImageFont.load_default()
    for row in boxes:
        box = [float(row[key]) for key in ("x1", "y1", "x2", "y2")]
        draw.rectangle(box, outline="#ffee44", width=3)
        label = f"ID {row.get('track_id', '')} class {row.get('class', '?')} {float(row['confidence']):.2f}"
        draw.text((box[0], max(40, box[1] - 28)), label, fill="#ffff44", font=font)
    draw.rectangle((0, 0, image.width, 40), fill="#111822")
    draw.text((8, 8), title, fill="white", font=font)
    return image


def write_sheets(frames, source, fps, profiles, indices, output):
    output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    paths = []
    width = 480
    height = 270
    labels = list(profiles)
    try:
        for start in range(0, len(indices), 4):
            selected = indices[start:start + 4]
            sheet = Image.new("RGB", ((len(labels) + 1) * width, len(selected) * (height + 28)), "#111822")
            for position, frame_index in enumerate(selected):
                record = frames[frame_index]
                frame = frame_at(capture, round(float(record["source_time"]) * fps))
                original = annotate(frame, [], f"ORIGINAL {float(record['clip_time']):.1f}s / source {record['source_time']}")
                y = position * (height + 28)
                sheet.paste(original.resize((width, height)), (0, y))
                draw = ImageDraw.Draw(sheet)
                draw.text((8, y + height + 7), f"Source frame {round(float(record['source_time']) * fps)}", fill="white")
                for column, name in enumerate(labels, 1):
                    boxes = profiles[name][frame_index]
                    panel = annotate(frame, boxes, name)
                    sheet.paste(panel.resize((width, height)), (column * width, y))
                    draw.text((column * width + 8, y + height + 7), f"Current boxes: {len(boxes)}", fill="white")
            path = output / f"end_frames_{start // 4 + 1:02d}.jpg"
            sheet.save(path, quality=95)
            paths.append(str(path))
    finally:
        capture.release()
    return paths


def detector_probe(frames, source, fps, indices, weights, variants, roi=None, all_classes=False):
    from ultralytics import YOLO
    import torch
    torch.set_num_threads(2)
    model = YOLO(str(weights))
    capture = cv2.VideoCapture(str(source))
    profiles = {}
    timing = {}
    try:
        originals = {index: frame_at(capture, round(float(frames[index]["source_time"]) * fps)) for index in indices}
        for size, degrees in variants:
            name = f"{size}_rot{degrees}"
            profiles[name] = defaultdict(list)
            started = time.perf_counter()
            for index, original in originals.items():
                x_offset, y_offset = 0, 0
                if roi:
                    x_offset, y_offset, x_end, y_end = roi
                    original = original[y_offset:y_end, x_offset:x_end]
                rotation = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(degrees)
                frame = cv2.rotate(original, rotation) if degrees else original
                result = model.predict(frame, classes=None if all_classes else [2, 7], conf=0.15, iou=0.5, agnostic_nms=True, imgsz=size, device="cpu", verbose=False)[0]
                for box, confidence, class_id in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
                    if degrees == 90:
                        # Clockwise forward transform: x'=H-y, y'=x.
                        box = [box[1], original.shape[0] - box[2], box[3], original.shape[0] - box[0]]
                    elif degrees == 180:
                        box = [original.shape[1] - box[2], original.shape[0] - box[3], original.shape[1] - box[0], original.shape[0] - box[1]]
                    elif degrees == 270:
                        box = [original.shape[1] - box[3], box[0], original.shape[1] - box[1], box[2]]
                    box = [box[0] + x_offset, box[1] + y_offset, box[2] + x_offset, box[3] + y_offset]
                    profiles[name][index].append({**dict(zip(("x1", "y1", "x2", "y2"), map(float, box))), "confidence": float(confidence), "class": int(class_id)})
            timing[name] = time.perf_counter() - started
    finally:
        capture.release()
    return profiles, timing, model.names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", default="full_run01")
    parser.add_argument("--shot", default="shot04")
    parser.add_argument("--indices", default="69,74,76,79,80,82,84,86,87,88,90,91")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--skip-tracker-replay", action="store_true")
    parser.add_argument("--probe-variants", default="640:0,960:0,640:90", help="Comma separated size:clockwise rotation pairs")
    parser.add_argument("--weights", type=Path, help="Existing local pretrained weights; defaults to the pinned baseline")
    parser.add_argument("--roi", help="Optional source pixel crop x1,y1,x2,y2 for a diagnostic only")
    parser.add_argument("--all-classes", action="store_true", help="Diagnostic only: show mistaken COCO classes without relabelling them")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".settings")
    os.environ["MPLCONFIGDIR"] = str(args.output_dir / "matplotlib_cache")
    cv2.setNumThreads(2)
    directory = args.project_dir / "outputs/full_runs" / args.run_id / "shots" / args.shot
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    frames = read_csv(directory / "frames.csv")
    detections = group_frames(read_csv(directory / "detections.csv"))
    observations = group_frames(read_csv(directory / "observations.csv"))
    source = Path(summary["source_path"])
    fps = float(summary["source_fps"])
    indices = [int(value) for value in args.indices.split(",")]
    profiles = {"saved_raw": detections, "saved_confirmed": observations}
    experiments, settings = {}, {}
    configurations = (("default_replay", {}), ("gmc_none", {"gmc_method": "none"}), ("threshold_015", {"track_high_thresh": 0.15, "new_track_thresh": 0.15}), ("none_threshold_015", {"gmc_method": "none", "track_high_thresh": 0.15, "new_track_thresh": 0.15}), ("none_no_fuse", {"gmc_method": "none", "fuse_score": False}), ("none_no_fuse_015", {"gmc_method": "none", "fuse_score": False, "track_high_thresh": 0.15, "new_track_thresh": 0.15}))
    for name, overrides in (() if args.skip_tracker_replay else configurations):
        rows, profile, elapsed = replay_saved(frames, detections, source, fps, overrides)
        experiments[name] = rows
        settings[name] = {"settings": profile, "seconds": elapsed, "observed_frames": len([value for value in rows.values() if value]), "observations": sum(map(len, rows.values())), "ids": sorted({row['track_id'] for value in rows.values() for row in value})}
        print(name, settings[name], flush=True)
    # Keep source, raw and confirmed panels readable rather than many columns.
    sheets = write_sheets(frames, source, fps, profiles, indices, args.output_dir)
    probe_timings = None
    probe_names = None
    if args.probe:
        variants = [tuple(map(int, value.split(":"))) for value in args.probe_variants.split(",")]
        if any(size < 320 or size % 32 or degrees not in {0, 90, 180, 270} for size, degrees in variants):
            parser.error("Probe size must be a multiple of32 at least320 and rotation must be0,90,180,270")
        weights = args.weights or args.project_dir / "models/yolov8n.pt"
        roi = tuple(map(int, args.roi.split(","))) if args.roi else None
        if roi and (len(roi) != 4 or not 0 <= roi[0] < roi[2] <= summary["width"] or not 0 <= roi[1] < roi[3] <= summary["height"]):
            parser.error("ROI must stay within the original source frame")
        probes, probe_timings, probe_names = detector_probe(frames, source, fps, indices, weights, variants, roi, args.all_classes)
        sheets.extend(write_sheets(frames, source, fps, probes, indices, args.output_dir / "probe"))
        profiles.update(probes)
    profiles.update(experiments)
    audit = [{**record, **{name + "_count": len(rows[int(record["frame_index"])]) for name, rows in profiles.items()}} for record in frames]
    result = {"run_id": args.run_id, "shot": args.shot, "source": str(source), "source_fps": fps, "baseline_settings": {key: summary.get(key) for key in ("tracker", "imgsz", "confidence_threshold", "agnostic_nms", "sampled_fps", "model", "model_sha256")}, "selected_frame_indices": indices, "probe_weights": str(args.weights or args.project_dir / "models/yolov8n.pt"), "probe_variants": args.probe_variants if args.probe else None, "probe_roi": args.roi, "probe_all_classes": args.all_classes, "probe_class_names": probe_names, "per_frame_counts": audit, "tracking_experiments": settings, "detector_probe_seconds": probe_timings, "boxes": {name: {str(index): rows[index] for index in indices} for name, rows in profiles.items()}, "contact_sheets": sheets, "limitations": ["Matched source samples from the existing demonstration, not independent evaluation.", "Saved raw CSV does not preserve detector class; saved-box replay supplies class 2, and classes do not participate in default IoU association.", "Counts include parked and false candidates; visually inspect source livery before role assignments.", "Probe results are raw boxes, not validated identities or physical measurements."]}
    (args.output_dir / "visibility_diagnostic.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "per_frame_counts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)
    print(args.output_dir / "visibility_diagnostic.json", flush=True)


if __name__ == "__main__":
    main()
