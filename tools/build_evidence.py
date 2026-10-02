"""Build reproducible diagnostics from saved observations and independent references."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from driftlens.evaluation import evaluate_samples, read_csv


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def compact(report: dict, summaries: list[dict]) -> dict:
    frames = sum(row["frame_count"] for row in summaries)
    seconds = sum(row["processing_seconds"] for row in summaries)
    return {
        **report["detection"],
        "visible_reference_coverage": report["tracking"]["visible_ground_truth_coverage"],
        "sparse_identity_switches": report["tracking"]["identity_switches"],
        "comparable_identity_transitions": report["tracking"]["comparable_identity_transitions"],
        **report["counts"],
        "processing_fps": round(frames / seconds, 3) if seconds else None,
        "processed_frames": frames,
        "processing_seconds": round(seconds, 3),
    }


def role_votes(report: dict) -> dict:
    votes = {"car_a": Counter(), "car_b": Counter()}
    for sample in report["samples"]:
        for match in sample.get("matches", []):
            if match["track_id"] is not None:
                votes.setdefault(match["identity"], Counter())[int(match["track_id"])] += 1
    return {identity: dict(counter.most_common()) for identity, counter in votes.items()}


def detector_compact(report: dict, summaries: list[dict]) -> dict:
    result = compact(report, summaries)
    for name in ("visible_reference_coverage", "sparse_identity_switches", "comparable_identity_transitions", "matched_cars_with_track_id", "identity_switches"):
        result.pop(name, None)
    return result


def build() -> dict:
    labels_path = ROOT / "data/annotations/labels.json"
    annotations = json.loads(labels_path.read_text(encoding="utf-8-sig"))
    catalog = json.loads((ROOT / "data/clip_catalog.json").read_text(encoding="utf-8-sig"))
    test_clips = [clip for clip in catalog["clips"] if clip["split"] == "test"]
    test_ids = {clip["id"] for clip in test_clips}
    test_labels = {**annotations, "samples": [sample for sample in annotations["samples"] if sample["clip_id"] in test_ids]}
    aggregated = {tracker: {"observations": [], "detections": [], "frames": [], "summaries": []} for tracker in ("bytetrack", "botsort")}
    clips, proposals = [], {}
    for clip in catalog["clips"]:
        clip_id = clip["id"]
        samples = [sample for sample in annotations["samples"] if sample["clip_id"] == clip_id]
        record = {"clip_id": clip_id, "label": clip["label"], "condition": clip["condition"], "split": clip["split"], "trackers": {}}
        for tracker in ("bytetrack", "botsort"):
            run = ROOT / "outputs/runs" / clip_id
            if tracker == "botsort":
                run = run / "comparisons/botsort"
            if not (run / "summary.json").is_file():
                continue
            summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
            if summary.get("status") != "complete":
                raise RuntimeError(f"Incomplete run: {run}")
            if summary.get("pipeline_revision") != 2:
                raise RuntimeError(f"Regenerate outdated tracking association: {run}")
            frames = [{**row, "clip_id": clip_id} for row in read_csv(run / "frames.csv")]
            observations, detections = read_csv(run / "observations.csv"), read_csv(run / "detections.csv")
            if samples:
                subset = {**annotations, "samples": samples}
                tracked = evaluate_samples(subset, observations, frames)
                raw = evaluate_samples(subset, detections, frames)
                for result in (tracked, raw):
                    result["evaluation_type"] = "sparse_assistant_visual_reference_diagnostic"
                    result["annotation_method"] = annotations["annotation_method"]
                    result["limitations"].append(annotations["limitations"])
                save(run / "evaluation.json", tracked)
                save(run / "raw_detection_evaluation.json", raw)
                record["trackers"][tracker] = compact(tracked, [summary])
                record["trackers"][tracker]["assigned_pair_frame_coverage"] = summary["pair_coverage"]
                proposals[f"{clip_id}/{tracker}"] = role_votes(tracked)
            if clip_id in test_ids:
                aggregate = aggregated[tracker]
                aggregate["observations"].extend(observations)
                aggregate["detections"].extend(detections)
                aggregate["frames"].extend(frames)
                aggregate["summaries"].append(summary)
        if clip_id in test_ids:
            clips.append(record)
    reports, raw_reports = {}, {}
    for tracker, aggregate in aggregated.items():
        reports[tracker] = compact(evaluate_samples(test_labels, aggregate["observations"], aggregate["frames"]), aggregate["summaries"])
        raw_reports[tracker] = detector_compact(evaluate_samples(test_labels, aggregate["detections"], aggregate["frames"]), aggregate["summaries"])
    report = {
        "schema_version": 1,
        "pipeline_revision": 2,
        "annotation_method": annotations["annotation_method"],
        "summary": {
            "evaluation_type": "Small sparse visual reference diagnostic, without human expert review",
            "test_clips": len(test_clips),
            "test_battles": len({clip["run_group"] for clip in test_clips}),
            "labelled_test_frames": len(test_labels["samples"]),
            "reference_car_boxes": sum(len(sample["cars"]) for sample in test_labels["samples"]),
            "minimum_iou": 0.5,
            "maximum_time_error_seconds": 0.1,
            "model": "YOLOv8n pretrained COCO, 416 pixel inference, confidence 0.15",
            "sampling": "10 sampled frames per second from 30 fps source",
            "raw_detector": raw_reports["bytetrack"],
            "confirmed_tracks": reports,
            "timing_scope": "Decode, detector inference and tracking only; excludes model loading and replay encoding. Observed offline CPU speed, not real time.",
        },
        "clips": clips,
        "raw_detector_by_execution": raw_reports,
        "split_basis": "Entire battles separated between six tuning clips and six test clips; same single event and broadcast. Settings were fixed before running test clips. No model training.",
        "limitations": [annotations["limitations"], "Reference boxes are approximate assistant drafted visible extents, independently of detector predictions. They have no human expert validation.", "Sparse switches only compare matched identities on adjacent labelled samples in the same shot. Misses break continuity; switches during gaps may be uncounted.", "Screen separation is perspective dependent and does not measure metres, speed, drift angle or judging quality.", "Track coverage means matched reference boxes with confirmed IDs; it does not prove dense identity correctness.", "Local footage reuse and redistribution rights have not been verified. Source and video outputs stay private."],
    }
    save(ROOT / "outputs/evaluation_report.json", report)
    save(ROOT / "outputs/reference_role_votes.json", proposals)
    manifest = {
        "source": {"path": catalog["source"]["path"], "size_bytes": (ROOT / catalog["source"]["path"]).stat().st_size, "metadata": catalog["source"]},
        "annotations_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        "weights_sha256": hashlib.sha256((ROOT / "models/yolov8n.pt").read_bytes()).hexdigest(),
        "settings": {"imgsz": 416, "target_fps": 10, "confidence": 0.15, "trackers": ["bytetrack", "botsort"], "device": "cpu", "torch_threads": 2},
    }
    previous_manifest = ROOT / "outputs/reproducibility_manifest.json"
    previous = json.loads(previous_manifest.read_text(encoding="utf-8")) if previous_manifest.is_file() else {}
    source_path = ROOT / catalog["source"]["path"]
    previous_source = previous.get("source", {})
    if previous_source.get("size_bytes") == source_path.stat().st_size and previous_source.get("modified_ns") == source_path.stat().st_mtime_ns and previous_source.get("sha256"):
        source_hash = previous_source["sha256"]
    else:
        digest = hashlib.sha256()
        with source_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        source_hash = digest.hexdigest()
    manifest["source"].update(sha256=source_hash, modified_ns=source_path.stat().st_mtime_ns)
    save(ROOT / "outputs/reproducibility_manifest.json", manifest)
    return report


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    result = build()
    print(json.dumps(result["summary"], indent=2))
