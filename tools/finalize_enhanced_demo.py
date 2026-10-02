"""Rebuild saved demo roles and a matched descriptive visibility report.

The explicit maps below are assistant visual decisions from source/replay
inspection on 3 October 2026. They are not an automatic role classifier.
"""
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from driftlens.full_run import process_full_run
from driftlens.run_analysis_ui import save_shot_analysis


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def interval(start, end, lead, chase, reason):
    return dict(start_clip_seconds=start, end_clip_seconds=end, lead_id=lead, chase_id=chase, review_basis=reason)


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def describe(run_id):
    folder = ROOT / "outputs/full_runs" / run_id
    summary, analysis = load(folder / "summary.json"), load(folder / "analysis.json") if (folder / "analysis.json").exists() else {}
    shots = []
    for shot in summary["shots"]:
        child = folder / "shots" / shot["shot_id"]
        detections, observations = rows(child / "detections.csv"), rows(child / "observations.csv")
        frames = rows(child / "frames.csv")
        seen = {int(row["frame_index"]) for row in observations}
        candidate_frames = {int(row["frame_index"]) for row in detections}
        shots.append(dict(shot_id=shot["shot_id"], samples=len(frames), accepted_pair_samples=shot["paired_frames"],
                          tracked_frames=len(seen), no_selected_candidate_frames=len(frames)-len(candidate_frames),
                          candidate_frames_without_tracks=len(candidate_frames-seen),
                          recovered_observations=sum(row.get("detection_method") == "temporal_appearance_class_recovery" for row in observations)))
    return dict(run_id=run_id, source_start=summary["source_start_seconds"], source_end=summary["source_end_seconds"],
                sampled_frames=summary["frame_count"], accepted_pair_samples=summary["paired_frames"],
                pair_coverage=summary["pair_coverage"], processing_seconds=summary["total_processing_seconds"], shots=shots,
                recovered_selected_boxes=analysis.get("box_provenance", {}).get("recovered_selected_boxes"),
                accepted_pair_samples_with_recovery=analysis.get("accepted_paired_samples_with_recovery"))


def main():
    manifest_path = ROOT / "data/full_run_enhanced_catalog.json"
    manifest = load(manifest_path)
    maps = [
        [interval(0, 5.8, 1, 2, "Separate green pink lead1 and orange blue chase2 confirmed from aligned source livery and travel order."),
         interval(5.8, 7.9, None, None, "Overlap produces merged participant boxes; ID27 also spans both cars before7.9s. Exclude ambiguous geometry."),
         interval(7.9, manifest["shots"][0]["end_seconds"]-manifest["shots"][0]["start_seconds"], 1, 27, "Source and replay show distinct green lead1 and orange blue chase27 at7.9,8.0,8.1s.")],
        [interval(0, 2.7, 2, 1, "Green pink lead2 and orange blue chase1 are distinct; IDs are reviewed independently in this camera."),
         interval(2.7, 3.1, None, None, "Close overlap and an oversized single box; exclude pair measurements."),
         interval(3.1, 4.2, 2, 1, "Separate source-confirmed lead2 and chase1 boxes after overlap."),
         interval(4.2, 4.9, None, None, "Late boxes span both cars or smoke; a high observation count does not establish correct geometry.")],
        [interval(0, .5, None, None, "ID1 is an off-track bin; early IDs2/4 include both small cars. Roles remain unknown."),
         interval(.5, 2.5, 4, 7, "Distinct green pink lead4 and orange blue chase7 confirmed by livery."),
         interval(2.5, 2.8, None, None, "Merged box or absent observation during overlap."),
         interval(2.8, 2.9, 4, 7, "Focused2.8s source and replay show separate participant boxes."),
         interval(2.9, 3.8, None, None, "Merged or absent boxes during close overlap; recovery labels do not make merged geometry acceptable."),
         interval(3.8, 4.3, 4, 7, "Source-confirmed green lead4 and orange chase7 separate again; any missing observation remains missing."),
         interval(4.3, manifest["shots"][2]["end_seconds"]-manifest["shots"][2]["start_seconds"], None, None, "Both current tracks disappear before the next camera cut.")],
        [interval(0, 2.7, 1, 2, "Distinct source-confirmed green pink lead1 and orange blue chase2; individual missing samples are not filled."),
         interval(2.7, 2.9, None, 2, "Orange chase2 has a distinct box under branches; lead lacks an accepted observation."),
         interval(2.9, 3, 1, 2, "Focused2.9s inspection confirms two distinct participant boxes under branches."),
         interval(3, 4.2, None, None, "Both cars are still visually present through the hairpin but lack accepted current observations."),
         interval(4.2, 4.9, 1, None, "Distinct green lead1 box; visible orange chase remains untracked."),
         interval(4.9, 9.2, 1, 26, "Separate current boxes on green pink lead1 and orange blue chase26 through every remaining finish sample.")]
    ]
    excluded = [[10,15,18,19], [], [1,2], [5,14,21,44,45]]
    for shot, decisions, rejected in zip(manifest["shots"], maps, excluded):
        shot.update(role_intervals=decisions, excluded_local_ids=rejected,
                    role_failure_notes="Merged, parked and ambiguous candidates are excluded by explicit reviewed intervals; missing current observations remain missing.")
    manifest["role_review_method"] = "AI assistant review of aligned source and numbered replay, using distinctive livery and travel order. Dense inspection at uncertain interval boundaries; shot04 inspected at all92 sampled timestamps. No human expert validation and no automatic association across cuts."
    save(manifest_path, manifest)
    summary = load(ROOT / "outputs/full_runs/full_run02/summary.json") if "--report-only" in sys.argv else process_full_run(manifest_path)
    for shot in summary["shots"]:
        child = ROOT / "outputs/full_runs/full_run02/shots" / shot["shot_id"]
        save_shot_analysis(child, load(child / "summary.json"))
    baseline, enhanced = describe("full_run01"), describe("full_run02")
    assert baseline["sampled_frames"] == enhanced["sampled_frames"] == 268
    assert baseline["source_start"] == enhanced["source_start"] and baseline["source_end"] == enhanced["source_end"]
    late = []
    for run_id in ("full_run01", "full_run02"):
        timeline = rows(ROOT / "outputs/full_runs" / run_id / "timeline.csv")
        samples = [row for row in timeline if row["shot_id"] == "shot04" and int(row["local_frame_index"]) >= 74]
        late.append(dict(run_id=run_id, samples=len(samples), accepted_pairs=sum(row["pair_observed"].lower() == "true" for row in samples)))
    report = dict(schema_version=1, scope="Matched reused 26.7 second demonstration; not independent detector accuracy or expert validation.",
                  baseline=baseline, enhanced=enhanced, finish_window=late, inference_profile=manifest["inference_profile"],
                  conclusion=f"Reviewed pair availability on this reused demo increased from {baseline['pair_coverage']:.1%} to {enhanced['pair_coverage']:.1%}. Both cars have accepted current observations in all 18 final samples from 7.4 seconds onward in shot04, compared with none previously. Remaining visible misses and ambiguous overlaps are retained as gaps. This does not establish accuracy on other videos.",
                  model_weights_changed=False, new_training_performed=False, human_reviewed_frames=0,
                  remaining_visible_misses=["shot04 both cars3.0to4.1s", "shot04 visible orange car4.2to4.8s", "shot03 last two samples", "ambiguous merged boxes in shots01to03"],
                  negative_experiments=["YOLOv8s640 missed more participant boxes in the late probe than existing nano weights.", "Higher resolution960/1280 did not reliably recover the green finish car.", "Native appearance ReID was not promoted because multi-pass and recovery provenance needs separate validation."],
                  provenance_note="R boxes are current detector outputs matched conservatively to a recent same-view vehicle anchor. Original class and its confidence remain exported; confidence is not car or identity probability. No hidden position is synthesized.",
                  limitation="Profile and role intervals were chosen on this reused demo. Processing timings include uncontrolled machine load. Upload operation is tested separately from arbitrary-footage accuracy. A stronger trained model needs reviewed labels from diverse events and an untouched event-level test.")
    save(ROOT / "outputs/visibility_report.json", report)
    profiles = dict(active_profile="enhanced", default_full_run_id="full_run02", profiles={
        "enhanced": dict(label="Enhanced observed car recovery", description="Experimental orientation and recent-appearance recovery. Slower CPU analysis; recovered boxes are marked R and retain original class confidence.", tracker="botsort", imgsz=640, agnostic_nms=True, **manifest["inference_profile"]),
        "baseline": dict(label="Original 416 pixel diagnostic", description="Original ByteTrack diagnostic profile for comparison, without experimental recovery.", tracker="bytetrack", imgsz=416, agnostic_nms=False, model_name="yolov8n.pt", confidence_threshold=.15, orientations=[0], recover_vehicle_classes=False)
    })
    save(ROOT / "data/inference_profiles.json", profiles)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
