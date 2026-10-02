"""Regression checks for preserving review files and reporting failed analysis."""

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from driftlens import pipeline, review


PROJECT = Path(__file__).resolve().parents[1]
VIDEO_INFO = {"source_fps": 10, "total_frames": 10, "width": 64, "height": 64, "duration_seconds": 1.0}


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class EmptyDetector:
    """Inference substitute that yields actual frames without car detections."""

    def __init__(self, *args, **kwargs):
        self.callback = None
        self.predictor = SimpleNamespace(trackers=[])

    def add_callback(self, event, callback):
        self.callback = callback

    def track(self, frame, **kwargs):
        result = SimpleNamespace(boxes=None)
        self.predictor.results = [result]
        if self.callback:
            self.callback(self.predictor)
        return [result]


class FailureSafetyTests(unittest.TestCase):
    def test_failed_role_render_preserves_existing_exports(self):
        with tempfile.TemporaryDirectory(dir=PROJECT, prefix=".failure_test_") as temporary:
            run = Path(temporary)
            summary = {"status": "complete", "track_ids": [1, 2], "shot_count": 1, "lead_id": None, "chase_id": None, "source_path": "unused.mp4"}
            frame = {"frame_index": 0, "clip_time": 0, "source_time": 30, "shot_index": 0}
            observations = [
                {"clip_id": "test", **frame, "track_id": identity, "x1": x, "y1": 10, "x2": x + 20, "y2": 30, "confidence": 0.9, "observed": True}
                for identity, x in ((1, 10), (2, 50))
            ]
            (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            (run / "annotated.mp4").write_bytes(b"original replay sentinel")
            review.write_metrics(run / "frame_metrics.csv", review.derive_frame_metrics(observations, [frame], None, None))
            write_csv(run / "frames.csv", pipeline.FRAME_FIELDS, [frame])
            write_csv(run / "observations.csv", pipeline.OBSERVATION_FIELDS, observations)
            original = {name: (run / name).read_bytes() for name in ("summary.json", "frame_metrics.csv", "annotated.mp4")}

            def fail_after_partial_encoding(destination, *args):
                (destination / "annotated.tmp.mp4").write_bytes(b"incomplete encoded replay")
                raise RuntimeError("The original source became unreadable during rendering")

            with patch.object(pipeline, "render_run", side_effect=fail_after_partial_encoding):
                with self.assertRaisesRegex(RuntimeError, "unreadable"):
                    review.assign_roles(run, 1, 2)
            for name, contents in original.items():
                self.assertEqual((run / name).read_bytes(), contents, name)
            self.assertEqual(list(run.glob(".role_update_*")), [])

    def test_encoder_failure_marks_full_analysis_failed(self):
        with tempfile.TemporaryDirectory(dir=PROJECT, prefix=".failure_test_") as temporary:
            root = self.prepare_root(temporary)
            output = root / "result"
            capture = MagicMock()
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            capture.read.side_effect = [(True, image), (True, image)]
            with patch.dict(os.environ, self.environment(root)), patch.object(pipeline, "ROOT", root), patch.object(pipeline, "video_info", return_value=VIDEO_INFO), patch.object(pipeline.cv2, "VideoCapture", return_value=capture), patch.dict("sys.modules", self.inference_modules(EmptyDetector)), patch.object(pipeline, "render_run", side_effect=RuntimeError("Replay encoder failed")):
                with self.assertRaisesRegex(RuntimeError, "encoder"):
                    pipeline.analyze_video(root / "source.mp4", output, 0, 0.2)
            state = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "failed")
            self.assertIn("encoder", state["error"])
            self.assertTrue((output / "frames.csv").is_file())
            capture.release.assert_called_once()

    def test_early_source_decode_is_failure_instead_of_complete_interval(self):
        with tempfile.TemporaryDirectory(dir=PROJECT, prefix=".failure_test_") as temporary:
            root = self.prepare_root(temporary)
            output = root / "result"
            capture = MagicMock()
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            capture.read.side_effect = [(True, image), (True, image), (False, None)]
            with patch.dict(os.environ, self.environment(root)), patch.object(pipeline, "ROOT", root), patch.object(pipeline, "video_info", return_value=VIDEO_INFO), patch.object(pipeline.cv2, "VideoCapture", return_value=capture), patch.dict("sys.modules", self.inference_modules(EmptyDetector)), patch.object(pipeline, "render_run") as render:
                with self.assertRaisesRegex(RuntimeError, "ended early"):
                    pipeline.analyze_video(root / "source.mp4", output, 0, 0.3)
            state = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "failed")
            self.assertIn("frame 2", state["error"])
            render.assert_not_called()
            capture.release.assert_called_once()

    def test_model_startup_failure_does_not_leave_processing_status(self):
        with tempfile.TemporaryDirectory(dir=PROJECT, prefix=".failure_test_") as temporary:
            root = self.prepare_root(temporary)
            output = root / "result"
            model_failure = MagicMock(side_effect=RuntimeError("Detector weights could not be loaded"))
            with patch.dict(os.environ, self.environment(root)), patch.object(pipeline, "ROOT", root), patch.object(pipeline, "video_info", return_value=VIDEO_INFO), patch.dict("sys.modules", self.inference_modules(model_failure)):
                with self.assertRaisesRegex(RuntimeError, "weights"):
                    pipeline.analyze_video(root / "source.mp4", output, 0, 0.2)
            state = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "failed")
            self.assertIn("weights", state["error"])

    @staticmethod
    def inference_modules(detector):
        return {"ultralytics": SimpleNamespace(YOLO=detector), "torch": SimpleNamespace(set_num_threads=lambda count: None)}

    @staticmethod
    def environment(root):
        return {"YOLO_CONFIG_DIR": str(root / ".settings"), "MPLCONFIGDIR": str(root / "matplotlib_cache")}

    @staticmethod
    def prepare_root(temporary):
        root = Path(temporary)
        (root / "models").mkdir()
        # The mocked detector never opens this placeholder weight file.
        (root / "models" / "yolov8n.pt").write_bytes(b"mock weights")
        return root


if __name__ == "__main__":
    unittest.main()
