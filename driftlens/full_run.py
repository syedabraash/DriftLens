"""Complete tandem run review with explicit camera boundaries and local IDs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import uuid

ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = 0.001
TIMELINE_FIELDS = ["frame_index", "run_time", "playback_time", "source_time", "shot_id", "local_frame_index", "local_shot_index", "camera_cut", "role_assignment_cut", "lead_track_id", "chase_track_id", "lead_observed", "chase_observed", "pair_observed", "separation_proxy", "lead_confidence", "chase_confidence", "measurement_status"]


def _number(value, name: str) -> float:
    try:
        result = float(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _flag(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _optional(value):
    return None if value is None or str(value).strip() in {"", "None", "null", "nan"} else _number(value, "metric")


def validate_run_manifest(manifest: dict, source_duration: float | None = None) -> dict:
    """Require a complete contiguous source interval, without silent omissions."""
    safe_id = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,79}\Z")
    if not safe_id.fullmatch(str(manifest.get("id", ""))):
        raise ValueError("Run ID must be a safe folder name")
    if not str(manifest.get("title", "")).strip():
        raise ValueError("A complete run needs a title")
    if manifest.get("coverage") != "initiation_to_finish":
        raise ValueError("Coverage must explicitly declare initiation_to_finish")
    start = _number(manifest.get("source_start_seconds"), "run start")
    end = _number(manifest.get("source_end_seconds"), "run end")
    if start < 0 or not 0 < end - start <= 180:
        raise ValueError("A complete run must span more than zero and at most 180 seconds")
    if source_duration is not None and end > _number(source_duration, "source duration") + TOLERANCE:
        raise ValueError("The complete run exceeds the source video")
    shots = manifest.get("shots")
    if not isinstance(shots, list) or not shots:
        raise ValueError("A complete run needs ordered camera shots")
    seen, expected = set(), start
    for shot in shots:
        identifier = str(shot.get("id", ""))
        if not safe_id.fullmatch(identifier) or identifier in seen:
            raise ValueError("Shot IDs must be safe and unique")
        seen.add(identifier)
        shot_start = _number(shot.get("start_seconds"), "shot start")
        shot_end = _number(shot.get("end_seconds"), "shot end")
        if abs(shot_start - expected) > TOLERANCE:
            raise ValueError("Camera shots must be contiguous with no gaps or overlaps")
        if not 0 < shot_end - shot_start <= 60:
            raise ValueError("Each camera shot must span more than zero and at most 60 seconds")
        expected = shot_end
    if abs(expected - end) > TOLERANCE:
        raise ValueError("Camera shots must cover the complete run through its end")
    return manifest


def build_timeline(shots: list[dict], rows_by_shot: dict[str, list[dict]], source_start: float) -> list[dict]:
    """Keep actual frames, missing measurements and shot scoped observations.

    run_time is source elapsed time. playback_time accounts for fractional
    source shot boundaries rounded to the sampled replay's frame duration.
    """
    timeline, playback_offset, last_source = [], 0.0, None
    for shot_position, shot in enumerate(shots):
        identifier = shot["id"]
        rows = rows_by_shot.get(identifier)
        if not rows:
            raise ValueError(f"No sampled frames for camera shot {identifier}")
        previous_local_shot, previous_local_frame, previous_roles = None, None, None
        sample_period = 1 / _number(shot.get("sampled_fps", 10), "sampled fps")
        local_playback_end = 0.0
        for position, row in enumerate(rows):
            source_time = _number(row["source_time"], "source time")
            if source_time < float(shot["start_seconds"]) - TOLERANCE or source_time >= float(shot["end_seconds"]) + TOLERANCE:
                raise ValueError("A sampled frame lies outside its camera shot")
            if last_source is not None and source_time <= last_source:
                raise ValueError("Source timestamps must be unique and strictly ordered")
            local_frame = int(row["frame_index"])
            if previous_local_frame is not None and local_frame <= previous_local_frame:
                raise ValueError("Local frame indices must be strictly ordered")
            local_shot = int(row.get("shot_index", 0))
            camera_cut = (shot_position > 0 and position == 0) or (previous_local_shot is not None and local_shot != previous_local_shot)
            lead, chase = _flag(row.get("lead_observed")), _flag(row.get("chase_observed"))
            active_roles = tuple(int(_optional(row.get(key))) if _optional(row.get(key)) is not None else None for key in ("lead_track_id", "chase_track_id")) if "role_intervals" in shot else (shot.get("lead_id"), shot.get("chase_id"))
            roles_unassigned = ("lead_id" in shot and active_roles[0] is None) or ("chase_id" in shot and active_roles[1] is None)
            if roles_unassigned:
                lead = lead and active_roles[0] is not None
                chase = chase and active_roles[1] is not None
            proxy = _optional(row.get("separation_proxy")) if lead and chase and _flag(row.get("pair_observed")) else None
            if proxy is not None and proxy < 0:
                raise ValueError("Image separation cannot be negative")
            pair = proxy is not None
            status = "roles_unassigned" if roles_unassigned else "pair_observed" if pair else "both_missing" if not lead and not chase else "lead_missing" if not lead else "chase_missing" if not chase else "measurement_unavailable"
            clip_time = _number(row.get("clip_time", source_time - float(shot["start_seconds"])), "clip time")
            timeline.append({"frame_index": len(timeline), "run_time": round(source_time - float(source_start), 6), "playback_time": round(playback_offset + clip_time, 6), "source_time": source_time, "shot_id": identifier, "local_frame_index": local_frame, "local_shot_index": local_shot, "camera_cut": camera_cut, "role_assignment_cut": previous_roles is not None and active_roles != previous_roles, "lead_track_id": active_roles[0], "chase_track_id": active_roles[1], "lead_observed": lead, "chase_observed": chase, "pair_observed": pair, "separation_proxy": proxy, "lead_confidence": _optional(row.get("lead_confidence")) if lead else None, "chase_confidence": _optional(row.get("chase_confidence")) if chase else None, "measurement_status": status})
            local_playback_end = clip_time + sample_period
            previous_local_frame, previous_local_shot, last_source = local_frame, local_shot, source_time
            previous_roles = active_roles
        playback_offset += local_playback_end
    return timeline


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _save(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def assemble_full_run(manifest: dict, run_dir: Path) -> dict:
    """Publish one complete replay and timestamp map from reviewed shot results."""
    import imageio_ffmpeg
    validate_run_manifest(manifest)
    run_dir = Path(run_dir).resolve()
    run_dir.relative_to(ROOT)
    entries, metrics, summaries = [], {}, []
    expected_source = (ROOT / manifest["source"]["path"]).resolve()
    expected_source.relative_to(ROOT)
    for shot in manifest["shots"]:
        child = run_dir / "shots" / shot["id"]
        summary = json.loads((child / "summary.json").read_text(encoding="utf-8"))
        if summary.get("status") != "complete" or summary.get("pipeline_revision") != 2:
            raise ValueError(f"Camera shot is incomplete or outdated: {shot['id']}")
        if Path(summary.get("source_path", "")).resolve() != expected_source:
            raise ValueError("Stored camera shot uses a different source video")
        if abs(summary["start_seconds"] - shot["start_seconds"]) > TOLERANCE or abs(summary["end_seconds"] - shot["end_seconds"]) > TOLERANCE:
            raise ValueError("Stored shot range disagrees with the complete run manifest")
        profile_keys = ("tracker", "model", "model_sha256", "imgsz", "sampled_fps", "agnostic_nms", "confidence_threshold", "pipeline_revision", "source_signature", "inference_profile", "tracker_options", "orientations", "recover_vehicle_classes", "visibility_revision", "camera_cut_settings", "recovery_settings")
        if summaries and any(summary.get(key) != summaries[0].get(key) for key in profile_keys):
            raise ValueError("Camera shots use inconsistent processing profiles or source signatures")
        summaries.append(summary)
        entries.append({**shot, "lead_id": summary.get("lead_id"), "chase_id": summary.get("chase_id"), "sampled_fps": summary["sampled_fps"], **({"role_intervals": summary["role_intervals"]} if "role_intervals" in summary else {})})
        metrics[shot["id"]] = _read_csv(child / "frame_metrics.csv")
        if len(metrics[shot["id"]]) != summary["frame_count"]:
            raise ValueError("Camera shot metrics do not match the stored frame count")
    timeline = build_timeline(entries, metrics, manifest["source_start_seconds"])
    shot_records, playback_offset = [], 0.0
    for entry, summary in zip(entries, summaries):
        duration = summary["frame_count"] / summary["sampled_fps"]
        role_status = summary.get("role_review_status", "assistant_visual_review") if summary.get("role_assignment_method") and (summary.get("lead_id") is not None or summary.get("role_intervals")) else "unassigned"
        shot_records.append({"shot_id": entry["id"], "label": entry["label"], "start_seconds": entry["start_seconds"], "end_seconds": entry["end_seconds"], "run_start_seconds": round(playback_offset, 6), "run_end_seconds": round(playback_offset + duration, 6), "source_elapsed_start_seconds": entry["start_seconds"] - manifest["source_start_seconds"], "lead_id": summary.get("lead_id"), "chase_id": summary.get("chase_id"), "role_status": role_status, "frame_count": summary["frame_count"], "paired_frames": summary["paired_frames"], "pair_coverage": summary["pair_coverage"], "detected_internal_shots": summary["shot_count"], "lead_description": entry.get("lead_description"), "chase_description": entry.get("chase_description"), "role_reference": entry.get("role_reference")})
        playback_offset += duration
        if "role_intervals" in summary:
            shot_records[-1]["role_intervals"] = summary["role_intervals"]
    result = {"schema_version": 1, "status": "complete", "run_id": manifest["id"], "title": manifest["title"], "source_path": str(ROOT / manifest["source"]["path"]), "source_start_seconds": manifest["source_start_seconds"], "source_end_seconds": manifest["source_end_seconds"], "source_duration_seconds": manifest["source_end_seconds"] - manifest["source_start_seconds"], "duration_seconds": round(playback_offset, 6), "frame_count": len(timeline), "shot_count": len(entries), "tracker": summaries[0]["tracker"], "imgsz": summaries[0]["imgsz"], "sampled_fps": summaries[0]["sampled_fps"], "paired_frames": sum(row["pair_observed"] for row in timeline), "pair_coverage": sum(row["pair_observed"] for row in timeline) / len(timeline), "roles_reviewed_shots": sum(row["role_status"] != "unassigned" for row in shot_records), "total_processing_seconds": round(sum(summary["processing_seconds"] for summary in summaries), 3), "role_review_status": "assistant_visual_review_no_human_expert", "identity_scope": "Track IDs are local to each shot. Roles are reviewed independently from source livery and travel order; no automatic cross camera identity association is claimed.", "coverage": manifest["coverage"], "completeness_basis": manifest.get("completeness_basis"), "limitations": manifest.get("limitations", []), "sampling_note": "Individual source shots end between sampled frames. The exported replay includes each sampled frame for its frame duration; playback_time maps these timestamps, while run_time preserves exact source elapsed time.", "files": {"video": "annotated.mp4", "metrics": "timeline.csv", "shots": "shots.json", "summary": "summary.json"}, "shots": shot_records}
    result["agnostic_nms"] = summaries[0].get("agnostic_nms", False)
    result["model"] = summaries[0].get("model")
    result["model_sha256"] = summaries[0].get("model_sha256")
    result["source_class_names"] = summaries[0].get("source_class_names", {})
    result["recovered_observations"] = sum(summary.get("recovered_observations", 0) for summary in summaries)
    result["inference_profile"] = summaries[0].get("inference_profile", {})
    result["manifest_path"] = manifest.get("manifest_path", "data/full_run_catalog.json")
    if any(row["role_status"] == "user_assignment_unverified" for row in shot_records):
        result["role_review_status"] = "mixed_assistant_and_user_assignments_no_expert_validation"
    elif not result["roles_reviewed_shots"]:
        result["role_review_status"] = "unassigned"
    stage = (run_dir / f".publish_{uuid.uuid4().hex}").resolve()
    stage.relative_to(run_dir)
    stage.mkdir()
    try:
        playlist = stage / "concat.txt"
        paths = [run_dir / "shots" / entry["id"] / "annotated.mp4" for entry in entries]
        playlist.write_text("".join("file '" + path.as_posix().replace("'", "'\\''") + "'\n" for path in paths), encoding="utf-8")
        command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(playlist), "-an", "-c", "copy", "-movflags", "+faststart", str(stage / "annotated.mp4")]
        encoded = subprocess.run(command, capture_output=True, check=False)
        if encoded.returncode:
            raise RuntimeError("Combined replay encoding failed: " + encoded.stderr.decode(errors="replace")[-1000:])
        with (stage / "timeline.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=TIMELINE_FIELDS)
            writer.writeheader()
            writer.writerows(timeline)
        _save(stage / "shots.json", shot_records)
        from .run_analysis import build_run_analysis, report_text
        evidence = {}
        for entry in entries:
            directory = run_dir / "shots" / entry["id"]
            evidence[entry["id"]] = {name: _read_csv(directory / f"{name}.csv") if (directory / f"{name}.csv").is_file() else None
                                      for name in ("detections", "observations", "frames")}
        analysis = build_run_analysis(result, timeline, shot_records, evidence)
        _save(stage / "analysis.json", analysis)
        (stage / "report.txt").write_text(report_text(analysis), encoding="utf-8")
        result["files"].update(analysis="analysis.json", report="report.txt")
        _save(stage / "summary.json", result)
        names = ("annotated.mp4", "timeline.csv", "shots.json", "analysis.json", "report.txt", "summary.json")
        prior = stage / "prior"
        prior.mkdir()
        existing = set()
        for name in names:
            if (run_dir / name).is_file():
                shutil.copyfile(run_dir / name, prior / name)
                existing.add(name)
        published = []
        try:
            for name in names:
                (stage / name).replace(run_dir / name)
                published.append(name)
        except Exception:
            for name in published:
                destination = run_dir / name
                if name in existing:
                    shutil.copyfile(prior / name, destination)
                else:
                    destination.relative_to(ROOT)
                    destination.unlink()
            raise
    finally:
        resolved = stage.resolve()
        resolved.relative_to(run_dir)
        if resolved.parent != run_dir or not resolved.name.startswith(".publish_"):
            raise ValueError("Combined replay cleanup escaped the run directory")
        shutil.rmtree(resolved)
    return result


def process_full_run(manifest_path: Path, tracker: str = "botsort", force: bool = False, imgsz: int = 640) -> dict:
    from .pipeline import analyze_video, render_run, video_info
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8-sig"))
    manifest["manifest_path"] = Path(manifest_path).resolve().relative_to(ROOT).as_posix()
    profile = manifest.get("inference_profile") or {}
    model_name = profile.get("model_name", "yolov8n.pt")
    source = (ROOT / manifest["source"]["path"]).resolve()
    source.relative_to(ROOT)
    validate_run_manifest(manifest, video_info(source)["duration_seconds"])
    run_dir = ROOT / "outputs/full_runs" / manifest["id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    _save(run_dir / "summary.json", {"run_id": manifest["id"], "status": "processing"})
    playback_offset = 0.0
    def digest(path):
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()
    try:
        source_signature = {"size_bytes": source.stat().st_size, "modified_ns": source.stat().st_mtime_ns}
        model_digest = digest(ROOT / "models" / model_name)
        has_review_map = any(shot.get("role_intervals") for shot in manifest["shots"])
        review_source_matches = not has_review_map or (digest(source) == manifest.get("source_sha256") and model_digest == manifest.get("model_sha256") and profile == manifest.get("role_review_profile", {}))
        for position, shot in enumerate(manifest["shots"]):
            destination = run_dir / "shots" / shot["id"]
            saved = destination / "summary.json"
            summary = json.loads(saved.read_text(encoding="utf-8")) if saved.is_file() else {}
            same_settings = summary.get("status") == "complete" and summary.get("pipeline_revision") == 2 and summary.get("tracker") == tracker and summary.get("imgsz") == imgsz and summary.get("sampled_fps") == 10 and summary.get("agnostic_nms") is True and Path(summary.get("source_path", "")).resolve() == source and summary.get("source_signature") == source_signature and abs(summary.get("start_seconds", -1) - shot["start_seconds"]) <= TOLERANCE and abs(summary.get("end_seconds", -1) - shot["end_seconds"]) <= TOLERANCE
            print(f"Full run shot {position+1}/{len(manifest['shots'])}: {shot['label']}", flush=True)
            same_settings = same_settings and summary.get("model_sha256") == model_digest
            same_settings = same_settings and summary.get("inference_profile", {}) == profile
            if force or not same_settings:
                profile_arguments = {key: profile[key] for key in ("model_name", "tracker_options", "confidence_threshold", "orientations", "recover_vehicle_classes") if key in profile}
                summary = analyze_video(source, destination, shot["start_seconds"], shot["end_seconds"], tracker, imgsz, 10, lambda progress, message: print(f"{progress:.0%} {message}", flush=True), agnostic_nms=True, **profile_arguments)
            summary.update(full_run_id=manifest["id"], full_run_shot=shot["id"], full_run_offset_seconds=playback_offset, source_signature=source_signature, model_sha256=model_digest, inference_profile=profile)
            if shot.get("role_intervals") and review_source_matches and tracker == "botsort" and imgsz == 640 and summary.get("visibility_revision", 2) == 2 and summary.get("role_review_status") != "user_assignment_unverified":
                summary = apply_reviewed_intervals(destination, summary, shot["role_intervals"])
            elif summary.get("role_review_status") == "assistant_visual_review" and not review_source_matches:
                from .review import derive_frame_metrics, write_metrics
                summary.pop("role_intervals", None)
                summary.update(lead_id=None, chase_id=None, role_review_status="unassigned", role_assignment_method=None, paired_frames=0, pair_coverage=0)
                write_metrics(destination / "frame_metrics.csv", derive_frame_metrics(_read_csv(destination / "observations.csv"), _read_csv(destination / "frames.csv"), None, None))
            render_run(destination, summary, _read_csv(destination / "observations.csv"), _read_csv(destination / "frames.csv"))
            _save(saved, summary)
            playback_offset += summary["frame_count"] / summary["sampled_fps"]
            print(json.dumps({"shot": shot["id"], "frames": summary["frame_count"], "ids": summary["track_ids"], "internal_shots": summary["shot_count"]}), flush=True)
        return assemble_full_run(manifest, run_dir)
    except Exception as error:
        _save(run_dir / "summary.json", {"run_id": manifest["id"], "status": "failed", "error": str(error)})
        raise


def apply_reviewed_intervals(run_dir: Path, summary: dict, intervals: list[dict]) -> dict:
    """Use explicit visual review decisions; never infer identity across gaps."""
    from .review import derive_frame_metrics, validate_role_intervals, write_metrics
    validate_role_intervals(intervals, summary["duration_seconds"], {int(i) for i in summary["track_ids"]})
    frames = _read_csv(run_dir / "frames.csv")
    observations = _read_csv(run_dir / "observations.csv")
    metrics = derive_frame_metrics(observations, frames, None, None, intervals)
    first_pair = next((i for i in intervals if i.get("lead_id") is not None and i.get("chase_id") is not None), {})
    summary.update(lead_id=first_pair.get("lead_id"), chase_id=first_pair.get("chase_id"), role_intervals=intervals, role_assignment_method="AI assistant visually reviewed car livery against source footage at interval boundaries; unknown intervals and identity ambiguities are excluded. No human expert validation.", role_review_status="assistant_visual_review", paired_frames=sum(row["pair_observed"] for row in metrics), pair_coverage=sum(row["pair_observed"] for row in metrics) / len(metrics))
    write_metrics(run_dir / "frame_metrics.csv", metrics)
    return summary


def update_shot_roles(manifest_path: Path, run_dir: Path, shot_id: str,
                      lead_id: int | None = None, chase_id: int | None = None,
                      role_intervals: list[dict] | None = None) -> dict:
    """Review one camera shot and rebuild the complete exports, with rollback."""
    from .review import assign_roles
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8-sig"))
    manifest["manifest_path"] = Path(manifest_path).resolve().relative_to(ROOT).as_posix()
    validate_run_manifest(manifest)
    run_dir = Path(run_dir).resolve()
    expected = (ROOT / "outputs/full_runs" / manifest["id"]).resolve()
    if run_dir != expected or shot_id not in {shot["id"] for shot in manifest["shots"]}:
        raise ValueError("Role review must refer to a known shot in this complete run")
    child = run_dir / "shots" / shot_id
    backup = (run_dir / f".role_backup_{uuid.uuid4().hex}").resolve()
    backup.relative_to(run_dir)
    backup.mkdir()
    saved_files = []
    try:
        for label, directory, names in (("shot", child, ("summary.json", "frame_metrics.csv", "annotated.mp4", "analysis.json", "report.txt")), ("run", run_dir, ("summary.json", "timeline.csv", "shots.json", "annotated.mp4", "analysis.json", "report.txt"))):
            for name in names:
                source = directory / name
                saved = None
                if source.is_file():
                    saved = backup / f"{label}_{name}"
                    shutil.copyfile(source, saved)
                saved_files.append((source, saved))
        if role_intervals is not None:
            from .pipeline import render_run
            current = json.loads((child / "summary.json").read_text(encoding="utf-8-sig"))
            revised = apply_reviewed_intervals(child, current, role_intervals)
            render_run(child, revised, _read_csv(child / "observations.csv"), _read_csv(child / "frames.csv"))
        else:
            if lead_id is None or chase_id is None:
                raise ValueError("Choose both whole shot roles or supply an interval mapping.")
            revised = assign_roles(child, int(lead_id), int(chase_id))
        revised["role_assignment_method"] = "User assigned local track IDs or bounded role intervals through the review interface; no independent human expert validation is asserted."
        revised["role_review_status"] = "user_assignment_unverified"
        _save(child / "summary.json", revised)
        from .run_analysis_ui import save_shot_analysis
        save_shot_analysis(child, revised)
        return assemble_full_run(manifest, run_dir)
    except Exception:
        for original, saved in saved_files:
            if saved is not None:
                shutil.copyfile(saved, original)
            elif original.is_file():
                original.relative_to(run_dir)
                original.unlink()
        raise
    finally:
        resolved = backup.resolve()
        resolved.relative_to(run_dir)
        if resolved.parent != run_dir or not resolved.name.startswith(".role_backup_"):
            raise ValueError("Role backup cleanup escaped the run directory")
        shutil.rmtree(resolved)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--tracker", choices=["bytetrack", "botsort"], default="botsort")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--size", type=int, choices=[416, 640], default=640)
    parser.add_argument("--assemble-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    result = assemble_full_run(manifest, ROOT / "outputs/full_runs" / manifest["id"]) if args.assemble_only else process_full_run(args.manifest, args.tracker, args.force, args.size)
    print(json.dumps({key: result[key] for key in ("run_id", "duration_seconds", "source_duration_seconds", "shot_count", "frame_count", "pair_coverage")}, indent=2))


if __name__ == "__main__":
    main()
