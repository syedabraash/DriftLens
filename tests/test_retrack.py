"""Saved candidate validation and cut/reset/current-box export invariants."""
from pathlib import Path
import csv
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from driftlens.pipeline import FRAME_FIELDS, OBSERVATION_FIELDS, _write_csv, resolve_tracker_settings
from driftlens.retrack import cached_inputs, retrack_cached_run


class CachedRetrackingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "original"
        self.run.mkdir()
        self.source = self.root / "source.avi"
        rng = np.random.default_rng(11)
        first = np.full((108, 192), 110, dtype=np.uint8)
        for _ in range(75):
            point = tuple(map(int, rng.integers((3, 3), (182, 98))))
            cv2.rectangle(first, point, (point[0]+8, point[1]+8), int(rng.choice((30, 190))), -1)
        second = np.flip(first, (0, 1)).copy()
        writer = cv2.VideoWriter(str(self.source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (192, 108))
        if not writer.isOpened():
            self.skipTest("OpenCV MJPG writer unavailable")
        for image in (first, first, second, second):
            writer.write(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
        writer.release()
        settings = resolve_tracker_settings("botsort", {"gmc_method": "none", "fuse_score": False,
                                                         "track_high_thresh": .15, "new_track_thresh": .15,
                                                         "track_low_thresh": .1})
        self.summary = {"status": "complete", "clip_id": "original", "source_path": str(self.source),
                        "source_fps": 10., "total_frames": 4, "width": 192, "height": 108,
                        "start_seconds": 0., "end_seconds": .4, "duration_seconds": .4,
                        "sampled_fps": 10., "frame_count": 4, "tracker": "botsort", "imgsz": 640,
                        "tracker_settings": settings, "orientations": [0], "recover_vehicle_classes": True,
                        "agnostic_nms": True, "processing_seconds": 8.5, "confidence_threshold": .15,
                        "uploaded_source": {"sha256": hashlib.sha256(self.source.read_bytes()).hexdigest()},
                        "lead_id": 1, "chase_id": 2, "role_intervals": [{"start_clip_seconds": 0., "end_clip_seconds": .4, "lead_id": 1, "chase_id": 2}],
                        "files": {}}
        self.frames = [{"frame_index": i, "clip_time": i/10, "source_time": i/10, "shot_index": 0} for i in range(4)]
        self.candidates = [{"clip_id": "original", **self.frames[i], "track_id": "", "x1": 20., "y1": 20.,
                            "x2": 80., "y2": 55., "confidence": float(np.float32(.8)), "observed": True,
                            "class": 2, "source_class_id": 2, "detection_method": "native_detector"} for i in (0, 2, 3)]
        self.save()

    def save(self):
        (self.run/"summary.json").write_text(json.dumps(self.summary), encoding="utf-8")
        _write_csv(self.run/"frames.csv", FRAME_FIELDS, self.frames)
        _write_csv(self.run/"candidates.csv", OBSERVATION_FIELDS, self.candidates)

    def test_rejects_post_recovery_candidates_instead_of_treating_them_as_inference(self):
        self.candidates[0]["detection_method"] = "temporal_appearance_class_recovery"
        self.save()
        with self.assertRaisesRegex(ValueError, "before class recovery"):
            cached_inputs(self.run)

    def test_rejects_disagreeing_or_misaligned_source_timestamps(self):
        self.candidates[0]["source_time"] = .1
        self.save()
        with self.assertRaisesRegex(ValueError, "timestamps disagree"):
            cached_inputs(self.run)
        self.candidates[0]["source_time"] = 0.
        self.frames[2]["clip_time"] = .1
        self.save()
        with self.assertRaisesRegex(ValueError, "not aligned"):
            cached_inputs(self.run)

    def test_new_result_resets_ids_clears_roles_and_keeps_missing_samples_empty(self):
        self.summary["role_continuation"] = {"active": True}
        self.summary["files"]["role_matches"] = "role_matches.json"
        self.save()
        before = {p.name: p.read_bytes() for p in self.run.iterdir()}
        def save_analysis(stage, summary):
            for name in ("analysis.json", "report.txt"):
                (stage/name).write_text("test report", encoding="utf-8")
            return summary
        output = self.root/"rebuilt"
        with patch("driftlens.retrack.render_run"), patch("driftlens.run_analysis_ui.save_shot_analysis", side_effect=save_analysis):
            summary = retrack_cached_run(self.run, output)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.run.iterdir()})
        self.assertEqual(summary["detected_camera_cuts_seconds"], [.2])
        self.assertEqual(summary["shot_count"], 2)
        self.assertIsNone(summary["lead_id"])
        self.assertIsNone(summary["chase_id"])
        self.assertNotIn("role_intervals", summary)
        self.assertNotIn("role_continuation", summary)
        self.assertNotIn("role_matches", summary["files"])
        self.assertEqual(summary["visibility_revision"], 4)
        self.assertEqual(summary["original_inference_processing_seconds"], 8.5)
        self.assertIn("no_fresh_inference", summary["processing_mode"])
        with (output/"observations.csv").open(newline="") as handle:
            observations = list(csv.DictReader(handle))
        self.assertEqual([int(row["frame_index"]) for row in observations], [0, 2, 3])
        self.assertEqual([int(row["track_id"])//10000 for row in observations], [0, 1, 1])
        for row in observations:
            self.assertEqual(tuple(float(row[k]) for k in ("x1", "y1", "x2", "y2", "confidence")),
                             (20., 20., 80., 55., float(np.float32(.8))))
        # A second cached rebuild must retain the detector's original cost,
        # rather than relabel the first cached tracking cost as YOLO inference.
        summary["processing_seconds"] = .25
        (output/"summary.json").write_text(json.dumps(summary), encoding="utf-8")
        first_cache_bytes = {p.name: p.read_bytes() for p in output.iterdir()}
        with patch("driftlens.retrack.render_run"), patch("driftlens.run_analysis_ui.save_shot_analysis", side_effect=save_analysis):
            second = retrack_cached_run(output, self.root/"rebuilt_again")
        self.assertEqual(second["original_inference_processing_seconds"], 8.5)
        self.assertNotEqual(second["original_inference_processing_seconds"], .25)
        self.assertEqual(second["cached_retracking"]["input_run_id"], "rebuilt")
        self.assertEqual(first_cache_bytes, {p.name: p.read_bytes() for p in output.iterdir()})
        self.assertFalse(any(p.name.startswith(".cached_retracking_") for p in self.root.iterdir()))

    def test_existing_result_and_changed_uploaded_source_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "original cannot be overwritten"):
            retrack_cached_run(self.run, self.run)
        self.summary["uploaded_source"]["sha256"] = "0"*64
        self.save()
        with self.assertRaisesRegex(ValueError, "source hash"):
            retrack_cached_run(self.run, self.root/"rebuilt")
        self.assertFalse((self.root/"rebuilt").exists())


if __name__ == "__main__":
    unittest.main()
