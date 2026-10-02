"""Check the browser-created upload and a real transactional role rebuild.

The private test excerpt's green lead and orange chase were visually reviewed.
This tests saved-file behavior, not new-event accuracy or expert labels.
"""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from driftlens.clip_role_review import update_clip_role_intervals
from driftlens.review import read_rows


def main():
    qa = json.loads((ROOT / "outputs/browser_improvements_qa.json").read_text())
    result = ROOT / "outputs/runs" / qa["upload_analysis"]["result"]
    saved = json.loads((result / "summary.json").read_text())
    source = Path(saved["source_path"]).resolve()
    source.relative_to(ROOT / "data/uploads")
    with source.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == saved["uploaded_source"]["sha256"]
    assert saved["status"] == "complete" and saved["frame_count"] == 20
    assert saved["analysis_profile_id"] == "enhanced" and saved["paired_frames"] == 0
    assert saved["uploaded_source"]["source_type"] == "user_upload"
    actual_observations = (result / "observations.csv").read_bytes()
    with tempfile.TemporaryDirectory(dir=ROOT / "outputs/upload_validation", prefix="role_rebuild_") as temporary:
        copied = Path(temporary)
        for original in result.iterdir():
            if original.is_file():
                shutil.copyfile(original, copied / original.name)
        revised = update_clip_role_intervals(copied, [{"start_clip_seconds": 0, "end_clip_seconds": 2,
                                                      "lead_id": 2, "chase_id": 1,
                                                      "review_basis": "Verification copy only: this new replay has green lead2 and orange chase1, independently reviewed at its endpoints; no expert validation."}])
        analysis = json.loads((copied / "analysis.json").read_text())
        metrics = read_rows(copied / "frame_metrics.csv")
        pairs = sum(row["pair_observed"].lower() == "true" for row in metrics)
        assert pairs == revised["paired_frames"] == analysis["accepted_paired_samples"] == 20
        assert revised["uploaded_source"] == saved["uploaded_source"]
        assert (copied / "observations.csv").read_bytes() == actual_observations
        assert (copied / "annotated.mp4").stat().st_size > 10000
        assert (copied / "report.txt").stat().st_size > 100
        assert revised["role_review_status"] == "user_assignment_unverified"
        cleared = update_clip_role_intervals(copied, [])
        assert cleared["paired_frames"] == 0
        assert json.loads((copied / "analysis.json").read_text())["accepted_paired_samples"] == 0
    assert json.loads((result / "summary.json").read_text())["paired_frames"] == 0
    evidence = dict(status="passed", actual_private_upload_sha256="passed", browser_result=qa["upload_analysis"]["result"],
                    actual_role_rebuild="passed on a verification copy", accepted_pairs_after_review=20,
                    explicit_unknown_reset="passed", original_observations_unchanged=True, uploaded_source_metadata_preserved=True,
                    original_browser_result_roles="unassigned", scope="Workflow checks on a reused source excerpt, not independent accuracy or human-reviewed training labels.")
    (ROOT / "outputs/upload_workflow_qa.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence))


if __name__ == "__main__":
    main()
