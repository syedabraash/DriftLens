"""Verify the independent project before delivery."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import math
import sys
import unicodedata

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_rows(directory, name):
    with (directory / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value):
    result = float(value)
    require(math.isfinite(result), f"Nonfinite exported number: {value!r}")
    return result


def optional_number(value):
    return None if value is None or str(value).strip().lower() in {"", "none", "null"} else number(value)


def optional_id(value):
    result = optional_number(value)
    if result is not None:
        require(result == int(result), f"Fractional exported track ID: {value!r}")
    return None if result is None else int(result)


def flag(value):
    text = str(value).strip().lower()
    require(text in {"true", "false", "1", "0", "yes", "no"}, f"Unknown boolean flag: {value!r}")
    return text in {"true", "1", "yes"}


def close(actual, expected, context, tolerance=0.000002):
    require(abs(number(actual) - number(expected)) <= tolerance, f"Incorrect {context}: {actual!r} versus {expected!r}")


def check_observations(raw, observations, context, require_unique=False):
    keys = ("frame_index", "shot_index", "x1", "y1", "x2", "y2", "confidence")
    available = Counter(tuple(row[key] for key in keys) for row in raw)
    used = Counter(tuple(row[key] for key in keys) for row in observations)
    require(all(count <= available[key] for key, count in used.items()), f"Duplicate or invented observed detection: {context}")
    if require_unique:
        require(all(count == 1 for count in used.values()), f"Multiple tracks reuse a current raw detection: {context}")
    ids = [(row["frame_index"], row["shot_index"], row["track_id"]) for row in observations]
    require(len(ids) == len(set(ids)), f"Repeated observed track ID: {context}")
    require(all(flag(row["observed"]) for row in observations), f"Unobserved prediction exported as observation: {context}")


def expected_roles(summary, clip_time):
    if "role_intervals" in summary:
        for entry in summary["role_intervals"]:
            if number(entry["start_clip_seconds"]) <= clip_time < number(entry["end_clip_seconds"]):
                return optional_id(entry.get("lead_id")), optional_id(entry.get("chase_id"))
        return None, None
    return optional_id(summary.get("lead_id")), optional_id(summary.get("chase_id"))


def check_actual_measurement(row, lead, chase, context):
    require(flag(row["lead_observed"]) == (lead is not None), f"Lead observation is invented or missing: {context}")
    require(flag(row["chase_observed"]) == (chase is not None), f"Chase observation is invented or missing: {context}")
    require(flag(row["pair_observed"]) == (lead is not None and chase is not None), f"Incorrect pair presence: {context}")
    proxy = optional_number(row["separation_proxy"])
    if lead is None or chase is None:
        require(proxy is None, f"Separation invented across unknown roles or missing detections: {context}")
    else:
        centres = [((number(car["x1"]) + number(car["x2"])) / 2, (number(car["y1"]) + number(car["y2"])) / 2) for car in (lead, chase)]
        mean_width = sum(number(car["x2"]) - number(car["x1"]) for car in (lead, chase)) / 2
        require(mean_width > 0 and proxy is not None, f"Invalid pair measurement: {context}")
        close(proxy, math.dist(*centres) / mean_width, f"observed separation in {context}")
    for key, car in (("lead_confidence", lead), ("chase_confidence", chase)):
        actual = optional_number(row[key])
        if car is None:
            require(actual is None, f"Confidence exists without a role observation: {context}")
        else:
            require(actual is not None, f"Missing observed confidence: {context}")
            close(actual, car["confidence"], f"{key} in {context}")


def verify_full_run(catalog):
    import cv2
    from driftlens.full_run import validate_run_manifest
    from driftlens.review import validate_role_intervals

    validate_run_manifest(catalog)
    source = (root / catalog["source"]["path"]).resolve()
    source.relative_to(root)
    source_capture = cv2.VideoCapture(str(source))
    require(source_capture.isOpened(), "Full run source video is unavailable")
    try:
        source_fps = source_capture.get(cv2.CAP_PROP_FPS)
        source_duration = source_capture.get(cv2.CAP_PROP_FRAME_COUNT) / source_fps
        dimensions = (int(source_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(source_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        source_capture.release()
    validate_run_manifest(catalog, source_duration)
    close(source_fps, 30, "source frame rate")
    require(dimensions == (1280, 720), "Unexpected source dimensions")
    require(catalog["end_frame_exclusive"] - catalog["start_frame"] == 801, "Full run must cover all 801 source frames")
    close(catalog["source_end_seconds"] - catalog["source_start_seconds"], 26.7, "complete source duration")
    require(catalog["frame_count"] == 801, "Manifest source frame count disagrees")
    require(len(catalog["shots"]) == 4, "The inspected full run must contain four camera shots")
    expected_source_frame = catalog["start_frame"]
    for shot in catalog["shots"]:
        require(shot["start_frame"] == expected_source_frame, "Source frame coverage contains a gap or overlap")
        require(shot["end_frame_exclusive"] > shot["start_frame"], "Empty source camera shot")
        close(shot["start_seconds"] * source_fps, shot["start_frame"], "shot start frame")
        close(shot["end_seconds"] * source_fps, shot["end_frame_exclusive"], "shot end frame")
        expected_source_frame = shot["end_frame_exclusive"]
    require(expected_source_frame == catalog["end_frame_exclusive"], "The source finish is not covered")

    run = root / "outputs/full_runs" / catalog["id"]
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    timeline = load_rows(run, "timeline.csv")
    shot_records = json.loads((run / "shots.json").read_text(encoding="utf-8"))
    require(summary["status"] == "complete", "Complete run publication did not finish")
    require(Path(summary["source_path"]).resolve() == source, "Complete run names a different source")
    require(summary["tracker"] == "botsort" and summary["imgsz"] == 640 and summary.get("agnostic_nms") is True, "Full run profile must be 640 pixel BoTSORT with agnostic NMS")
    require(summary["frame_count"] == len(timeline) == 268, "Full run must export exactly 268 actual sampled rows")
    require(summary["shot_count"] == len(shot_records) == 4, "Full run camera metadata is incomplete")
    close(summary["source_duration_seconds"], 26.7, "summary source duration")
    close(summary["duration_seconds"], 26.8, "summary playback duration")
    close(summary["sampled_fps"], 10, "summary sample rate")

    signature = {"size_bytes": source.stat().st_size, "modified_ns": source.stat().st_mtime_ns}
    with (root / "models/yolov8n.pt").open("rb") as handle:
        model_digest = hashlib.file_digest(handle, "sha256").hexdigest()
    require(summary.get("model_sha256") == model_digest == catalog.get("model_sha256"), "Full run model hash disagrees with current reviewed weights")
    global_index, offset, manual_cuts, internal_cuts, role_cuts = 0, 0.0, 0, 0, 0
    observed_count, unassigned_count, missing_count, paired_count = 0, 0, 0, 0
    records = []
    for shot_position, (shot, shot_record) in enumerate(zip(catalog["shots"], shot_records)):
        child = run / "shots" / shot["id"]
        child_summary = json.loads((child / "summary.json").read_text(encoding="utf-8"))
        frames = load_rows(child, "frames.csv")
        raw = load_rows(child, "detections.csv")
        observations = load_rows(child, "observations.csv")
        metrics = load_rows(child, "frame_metrics.csv")
        require(child_summary["status"] == "complete" and child_summary.get("pipeline_revision") == 2, f"Outdated or incomplete child: {shot['id']}")
        require(Path(child_summary["source_path"]).resolve() == source and child_summary.get("source_signature") == signature, f"Source cache is stale: {shot['id']}")
        require(child_summary["tracker"] == "botsort" and child_summary["imgsz"] == 640 and child_summary.get("agnostic_nms") is True, f"Wrong child profile: {shot['id']}")
        require(child_summary.get("model_sha256") == model_digest, f"Stale model cache: {shot['id']}")
        close(child_summary["sampled_fps"], 10, "child sample rate")
        close(child_summary["start_seconds"], shot["start_seconds"], "child source start")
        close(child_summary["end_seconds"], shot["end_seconds"], "child source end")
        source_indices = list(range(shot["start_frame"], shot["end_frame_exclusive"], 3))
        require(len(frames) == len(metrics) == child_summary["frame_count"] == len(source_indices), f"Child sampled rows are missing or fabricated: {shot['id']}")
        check_observations(raw, observations, child, require_unique=True)
        if "role_intervals" in child_summary:
            validate_role_intervals(child_summary["role_intervals"], child_summary["duration_seconds"], {int(identifier) for identifier in child_summary["track_ids"]})
        by_frame = {(int(row["frame_index"]), int(row["shot_index"]), int(row["track_id"])): row for row in observations}
        valid_frames = {(int(frame["frame_index"]), int(frame["shot_index"])) for frame in frames}
        require(all((int(row["frame_index"]), int(row["shot_index"])) in valid_frames for row in raw + observations), f"Detection refers to an unprocessed frame: {shot['id']}")
        previous_local_shot, previous_roles, child_pairs = None, None, 0
        for local_index, (frame, metric, source_index) in enumerate(zip(frames, metrics, source_indices)):
            context = f"{shot['id']} frame {local_index}"
            require(int(frame["frame_index"]) == int(metric["frame_index"]) == local_index, f"Wrong local frame ID: {context}")
            local_shot = int(frame["shot_index"])
            require(int(metric["shot_index"]) == local_shot, f"Metric changes camera identity: {context}")
            close(frame["source_time"], source_index / source_fps, f"actual source timestamp {context}")
            close(metric["source_time"], frame["source_time"], f"metric source timestamp {context}")
            close(frame["clip_time"], local_index / 10, f"local sampled clock {context}")
            close(metric["clip_time"], frame["clip_time"], f"metric sampled clock {context}")
            roles = expected_roles(child_summary, number(frame["clip_time"]))
            require(tuple(optional_id(metric[key]) for key in ("lead_track_id", "chase_track_id")) == roles, f"Metric ignores reviewed role intervals: {context}")
            lead = by_frame.get((local_index, local_shot, roles[0])) if roles[0] is not None else None
            chase = by_frame.get((local_index, local_shot, roles[1])) if roles[1] is not None else None
            check_actual_measurement(metric, lead, chase, context)

            exported = timeline[global_index]
            require(int(exported["frame_index"]) == global_index and int(exported["local_frame_index"]) == local_index and int(exported["local_shot_index"]) == local_shot and exported["shot_id"] == shot["id"], f"Timeline frame identity is incorrect: {context}")
            close(exported["source_time"], frame["source_time"], f"timeline source timestamp {context}")
            close(exported["run_time"], number(frame["source_time"]) - catalog["source_start_seconds"], f"source elapsed clock {context}")
            close(exported["playback_time"], global_index / 10, f"encoded playback clock {context}")
            require(tuple(optional_id(exported[key]) for key in ("lead_track_id", "chase_track_id")) == roles, f"Timeline changes reviewed roles: {context}")
            check_actual_measurement(exported, lead, chase, context)
            manual_cut = shot_position > 0 and local_index == 0
            internal_cut = previous_local_shot is not None and local_shot != previous_local_shot
            require(flag(exported["camera_cut"]) == (manual_cut or internal_cut), f"Incorrect camera boundary: {context}")
            assignment_cut = previous_roles is not None and roles != previous_roles
            require(flag(exported["role_assignment_cut"]) == assignment_cut, f"Incorrect role boundary: {context}")
            manual_cuts += int(manual_cut)
            internal_cuts += int(internal_cut)
            role_cuts += int(assignment_cut)
            child_pairs += int(lead is not None and chase is not None)
            unassigned_count += int(None in roles)
            missing_count += int(None not in roles and (lead is None or chase is None))
            previous_local_shot, previous_roles = local_shot, roles
            global_index += 1
        require(child_summary["paired_frames"] == child_pairs, f"Child paired count is incorrect: {shot['id']}")
        close(child_summary["pair_coverage"], child_pairs / len(frames), f"child coverage {shot['id']}")
        require(shot_record["shot_id"] == shot["id"] and shot_record["frame_count"] == len(frames) and shot_record["paired_frames"] == child_pairs, "Combined shot inventory disagrees with actual child frames")
        close(shot_record["run_start_seconds"], offset, "shot playback start")
        offset += len(frames) / 10
        close(shot_record["run_end_seconds"], offset, "shot playback end")
        paired_count += child_pairs
        observed_count += len(observations)
        records.append({"shot_id": shot["id"], "frames": len(frames), "observations": len(observations), "paired_frames": child_pairs, "role_intervals": len(child_summary.get("role_intervals", [])), "current_detection_association": "passed", "timestamp_and_role_mapping": "passed"})
    require(global_index == 268 and manual_cuts == 3, "Full run sampled coverage or manual camera cut count is wrong")
    require(summary["paired_frames"] == paired_count, "Complete run paired count is incorrect")
    close(summary["pair_coverage"], paired_count / len(timeline), "complete run pair coverage")
    close(offset, 26.8, "assembled playback extent")

    capture = cv2.VideoCapture(str(run / "annotated.mp4"))
    require(capture.isOpened(), "Combined MP4 cannot be opened")
    decoded = 0
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        close(fps, 10, "combined MP4 frame rate")
        require(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 268, "Combined MP4 metadata frame count is incorrect")
        while True:
            success, image = capture.read()
            if not success:
                break
            require(image.shape[:2] == (720, 1280), "A combined replay frame has wrong dimensions")
            decoded += 1
    finally:
        capture.release()
    require(decoded == 268, f"Combined MP4 decoded {decoded} frames instead of 268")
    close(decoded / fps, 26.8, "decoded MP4 duration")
    overlapping_test_clips = [clip["id"] for clip in catalog_sparse["clips"] if clip["split"] == "test" and clip["start_seconds"] < catalog["source_end_seconds"] and clip["end_seconds"] > catalog["source_start_seconds"]]
    return {"run_id": catalog["id"], "status": "passed", "source_frames_covered": 801, "source_duration_seconds": 26.7, "sampled_frames": 268, "child_results_checked": 4, "observations_checked": observed_count, "manual_camera_cuts": manual_cuts, "internal_camera_cuts": internal_cuts, "role_assignment_cuts": role_cuts, "paired_frames": paired_count, "pair_coverage": paired_count / 268, "unknown_role_frames": unassigned_count, "missing_assigned_observation_frames": missing_count, "decoded_mp4": {"frames": decoded, "fps": fps, "duration_seconds": decoded / fps, "width": 1280, "height": 720}, "source_cache_signature_check": "passed", "children": records, "evidence_scope": {"additional_independent_test": False, "overlapping_sparse_test_clips": overlapping_test_clips, "accuracy_evaluation_for_640_agnostic_profile": "not established by the separate 416 pixel sparse diagnostic", "role_reference_method": "Assistant visual review; no human expert validation asserted"}}


excluded = {".venv", ".git", "pip_cache"}
readmes = [p for p in root.rglob("README*") if p.is_file() and not any(part in excluded for part in p.relative_to(root).parts)]
for path in readmes:
    bad = [(i, char) for i, char in enumerate(path.read_text(encoding="utf-8-sig")) if char == "-" or char == "\u2212" or unicodedata.category(char) == "Pd"]
    if bad:
        raise SystemExit(f"Dash character in {path}: {bad[:5]}")
verification = {"project": str(root), "readmes_checked": len(readmes), "dash_check": "passed", "runs_checked": 0, "observations_checked": 0}
catalog_sparse = json.loads((root / "data/clip_catalog.json").read_text(encoding="utf-8-sig"))
for clip in catalog_sparse["clips"]:
    runs = [root / "outputs/runs" / clip["id"]]
    if clip["split"] == "test":
        runs.append(runs[0] / "comparisons/botsort")
    for run in runs:
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "complete" and summary.get("pipeline_revision") == 2, str(run)
        frames, raw, observations = (load_rows(run, name) for name in ("frames.csv", "detections.csv", "observations.csv"))
        assert len(frames) == summary["frame_count"], str(run)
        keys = ("frame_index", "shot_index", "x1", "y1", "x2", "y2", "confidence")
        available = Counter(tuple(row[key] for key in keys) for row in raw)
        used = Counter(tuple(row[key] for key in keys) for row in observations)
        assert all(count <= available[key] for key, count in used.items()), f"Duplicate or invented observed detection: {run}"
        ids = [(row["frame_index"], row["shot_index"], row["track_id"]) for row in observations]
        assert len(ids) == len(set(ids)), f"Repeated observed track ID: {run}"
        assert all(row["observed"] == "True" for row in observations), str(run)
        assert (run / "annotated.mp4").stat().st_size > 10000, str(run)
        assert summary["shot_count"] == 1, f"Unexpected camera cut: {run}"
        assert abs(summary["duration_seconds"] - (clip["end_seconds"] - clip["start_seconds"])) < 0.001, str(run)
        verification["runs_checked"] += 1
        verification["observations_checked"] += len(observations)
verification["current_detection_association_check"] = "passed"
full_catalog = json.loads((root / "data/full_run_catalog.json").read_text(encoding="utf-8-sig"))
verification["full_runs"] = [verify_full_run(full_catalog)]
verification["full_runs_checked"] = 1
verification["full_run_child_results_checked"] = 4
verification["status"] = "passed"
(root / "outputs/verification_report.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
print(json.dumps(verification))
