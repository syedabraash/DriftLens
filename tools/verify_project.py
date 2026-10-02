"""Verify the independent project before delivery."""
from pathlib import Path
from collections import Counter
import csv
import json
import unicodedata

root = Path(__file__).resolve().parents[1]
excluded = {".venv", ".git", "pip_cache"}
readmes = [p for p in root.rglob("README*") if p.is_file() and not any(part in excluded for part in p.relative_to(root).parts)]
for path in readmes:
    bad = [(i, char) for i, char in enumerate(path.read_text(encoding="utf-8-sig")) if char == "-" or char == "\u2212" or unicodedata.category(char) == "Pd"]
    if bad:
        raise SystemExit(f"Dash character in {path}: {bad[:5]}")
verification = {"project": str(root), "readmes_checked": len(readmes), "dash_check": "passed", "runs_checked": 0, "observations_checked": 0}
catalog = json.loads((root / "data/clip_catalog.json").read_text(encoding="utf-8-sig"))
for clip in catalog["clips"]:
    runs = [root / "outputs/runs" / clip["id"]]
    if clip["split"] == "test":
        runs.append(runs[0] / "comparisons/botsort")
    for run in runs:
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "complete" and summary.get("pipeline_revision") == 2, str(run)
        def rows(name):
            with (run / name).open(encoding="utf-8-sig", newline="") as handle:
                return list(csv.DictReader(handle))
        frames, raw, observations = rows("frames.csv"), rows("detections.csv"), rows("observations.csv")
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
(root / "outputs/verification_report.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
print(json.dumps(verification))
