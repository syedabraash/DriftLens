"""Apply reviewed visual reference roles to reliable demonstration results.

These presets are assistant reviewed starting points. Track IDs remain local
to each shot; ambiguous or swapped results intentionally stay unassigned.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from driftlens.review import assign_roles

PRESETS = [("clip01", "bytetrack", 2, 1), ("clip07", "botsort", 2, 1), ("clip08", "botsort", 1, 2), ("clip11", "botsort", 2, 1)]

if __name__ == "__main__":
    for clip, tracker, lead, chase in PRESETS:
        run = ROOT / "outputs/runs" / clip
        if tracker == "botsort":
            run = run / "comparisons/botsort"
        summary = assign_roles(run, lead, chase)
        summary["role_assignment_method"] = "AI assistant visual review of source references, independently of model predictions; no human expert validation. This role assignment is a review starting point."
        summary["role_reference"] = {"lead_identity": "car_a", "chase_identity": "car_b", "labels": "data/annotations/labels.json"}
        (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"clip": clip, "tracker": tracker, "lead": lead, "chase": chase, "paired_frames": summary["paired_frames"], "pair_coverage": summary["pair_coverage"]}), flush=True)
