"""Role interval updates regenerate evidence and recover from export failures."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from driftlens import full_run, pipeline, review


class IntervalUpdateTests(unittest.TestCase):
    def fixture(self, root):
        manifest = {"id": "review_run", "title": "Run", "source_start_seconds": 10,
                    "source_end_seconds": 12, "coverage": "initiation_to_finish",
                    "source": {"path": "source.mp4"},
                    "shots": [{"id": "shot01", "label": "Finish", "start_seconds": 10, "end_seconds": 12}]}
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest))
        run = root / "outputs/full_runs/review_run"
        child = run / "shots/shot01"
        child.mkdir(parents=True)
        frames = [{"frame_index": i, "clip_time": float(i), "source_time": 10 + i, "shot_index": 0} for i in range(2)]
        boxes = [{"clip_id": "shot01", **frame, "track_id": track, "x1": track * 40, "y1": 10,
                  "x2": track * 40 + 20, "y2": 30, "confidence": .9, "observed": True}
                 for frame in frames for track in (1, 2)]
        for name, fields, rows in (("frames.csv", pipeline.FRAME_FIELDS, frames),
                                  ("observations.csv", pipeline.OBSERVATION_FIELDS, boxes),
                                  ("detections.csv", pipeline.OBSERVATION_FIELDS, boxes)):
            with (child / name).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        summary = {"status": "complete", "clip_id": "shot01", "track_ids": [1, 2], "duration_seconds": 2,
                   "frame_count": 2, "sampled_fps": 1, "shot_count": 1, "lead_id": 1, "chase_id": 2,
                   "start_seconds": 10, "end_seconds": 12, "processing_seconds": 1, "processing_fps": 2}
        (child / "summary.json").write_text(json.dumps(summary))
        review.write_metrics(child / "frame_metrics.csv", review.derive_frame_metrics(boxes, frames, 1, 2))
        (child / "annotated.mp4").write_bytes(b"original shot")
        for name in ("summary.json", "timeline.csv", "shots.json", "annotated.mp4"):
            (run / name).write_bytes(b"original combined")
        return path, run, child

    def render(self, directory, *_):
        (directory / "annotated.mp4").write_bytes(b"revised shot")

    def test_bounded_decision_leaves_unreviewed_time_unknown(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, run, child = self.fixture(root)
            mapping = [{"start_clip_seconds": 0, "end_clip_seconds": .5, "lead_id": 1, "chase_id": 2}]
            with patch.object(full_run, "ROOT", root), patch.object(pipeline, "render_run", side_effect=self.render), patch.object(full_run, "assemble_full_run", return_value={}):
                full_run.update_shot_roles(manifest, run, "shot01", role_intervals=mapping)
            saved = json.loads((child / "summary.json").read_text())
            self.assertEqual(saved["role_review_status"], "user_assignment_unverified")
            self.assertEqual(saved["role_intervals"], mapping)
            rows = full_run._read_csv(child / "frame_metrics.csv")
            self.assertEqual([row["pair_observed"] for row in rows], ["True", "False"])
            self.assertEqual(rows[1]["separation_proxy"], "")

    def test_failed_combined_export_restores_all_originals_and_removes_new_reports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, run, child = self.fixture(root)
            originals = {path: path.read_bytes() for path in run.rglob("*") if path.is_file()}
            def fail(*_):
                (run / "report.txt").write_text("partial report")
                (run / "analysis.json").write_text("{}")
                raise RuntimeError("export failed")
            with patch.object(full_run, "ROOT", root), patch.object(pipeline, "render_run", side_effect=self.render), patch.object(full_run, "assemble_full_run", side_effect=fail):
                with self.assertRaisesRegex(RuntimeError, "export failed"):
                    full_run.update_shot_roles(manifest, run, "shot01", role_intervals=[])
            for path, contents in originals.items():
                self.assertEqual(path.read_bytes(), contents)
            self.assertFalse((run / "report.txt").exists())
            self.assertFalse((run / "analysis.json").exists())
            self.assertFalse((child / "report.txt").exists())
            self.assertFalse((child / "analysis.json").exists())

    def test_unobserved_track_is_rejected_without_changing_exports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, run, child = self.fixture(root)
            before = (child / "summary.json").read_bytes()
            with patch.object(full_run, "ROOT", root):
                with self.assertRaises(ValueError):
                    full_run.update_shot_roles(manifest, run, "shot01", role_intervals=[
                        {"start_clip_seconds": 0, "end_clip_seconds": 1, "lead_id": 1, "chase_id": 99}])
            self.assertEqual((child / "summary.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
