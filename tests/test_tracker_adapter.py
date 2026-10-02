"""Check identity association through the actual pinned tracking backends."""

import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from driftlens.tracker_adapter import bind_adapter, map_original_detection_indices


def snapshot(rows):
    return [dict(zip(("x1", "y1", "x2", "y2", "confidence", "class"), map(float, row))) for row in rows]


class TrackerAdapterTests(unittest.TestCase):
    def test_subset_indices_refer_to_original_detection_order(self):
        raw = snapshot([[0, 0, 20, 20, 0.2, 2], [100, 0, 120, 20, 0.9, 2]])
        self.assertEqual(map_original_detection_indices([[100, 0, 120, 20]], [0.9], raw, [2]), [1])
        self.assertEqual(map_original_detection_indices([[0, 0, 20, 20]], [0.2], raw, [2]), [0])
        self.assertEqual(map_original_detection_indices([], [], raw), [])

    def test_missing_ambiguous_or_reused_source_indices_fail(self):
        row = [0, 0, 20, 20, 0.2, 2]
        with self.assertRaisesRegex(ValueError, "no exact match"):
            map_original_detection_indices([[1, 0, 20, 20]], [0.2], snapshot([row]))
        with self.assertRaisesRegex(ValueError, "ambiguously"):
            map_original_detection_indices([[0, 0, 20, 20]], [0.2], snapshot([row, row]))
        with self.assertRaisesRegex(ValueError, "reuse"):
            map_original_detection_indices([[0, 0, 20, 20], [0, 0, 20, 20]], [0.2, 0.2], snapshot([row]))

    def test_class_disambiguates_identical_box_and_confidence(self):
        raw = snapshot([[0, 0, 20, 20, 0.2, 2], [0, 0, 20, 20, 0.2, 7]])
        self.assertEqual(map_original_detection_indices([[0, 0, 20, 20]], [0.2], raw, [7]), [1])

    def test_actual_trackers_preserve_low_and_high_confidence_associations(self):
        # The real trackers exercise their high and low confidence update stages.
        # No model weights, video input, or detector inference is involved.
        settings = Path(__file__).resolve().parents[1] / ".settings"
        settings.mkdir(exist_ok=True)
        original_config = os.environ.get("YOLO_CONFIG_DIR")
        os.environ["YOLO_CONFIG_DIR"] = str(settings)
        try:
            from ultralytics.engine.results import Boxes
            from ultralytics.trackers.byte_tracker import BYTETracker
            from ultralytics.trackers.bot_sort import BOTSORT
        finally:
            if original_config is None:
                os.environ.pop("YOLO_CONFIG_DIR", None)
            else:
                os.environ["YOLO_CONFIG_DIR"] = original_config
        args = SimpleNamespace(track_high_thresh=0.5, track_low_thresh=0.1, new_track_thresh=0.5, track_buffer=30, match_thresh=0.8, fuse_score=False, with_reid=False, proximity_thresh=0.5, appearance_thresh=0.8, gmc_method="none", model="auto")
        for tracker_class in (BYTETracker, BOTSORT):
            with self.subTest(tracker=tracker_class.__name__):
                tracker = tracker_class(args, frame_rate=10)
                raw = []
                predictor = SimpleNamespace(trackers=[tracker])
                bind_adapter(predictor, raw)
                wrapped_method = tracker.init_track
                bind_adapter(predictor, raw)
                self.assertIs(tracker.init_track, wrapped_method)

                initial = np.array([[0, 0, 20, 20, 0.9, 2], [100, 0, 120, 20, 0.8, 2]], dtype=np.float32)
                raw.extend(snapshot(initial))
                initial_tracks = tracker.update(Boxes(initial, orig_shape=(200, 200)))
                identity_a = int(initial_tracks[0, 4])
                identity_b = int(initial_tracks[1, 4])

                # B is now the high confidence prefix; A is the low confidence
                # suffix. An unadapted tracker returns index zero for both cars.
                next_detections = np.array([[100, 0, 120, 20, 0.8, 2], [0, 0, 20, 20, 0.2, 2]], dtype=np.float32)
                raw.clear()
                raw.extend(snapshot(next_detections))
                current_tracks = tracker.update(Boxes(next_detections, orig_shape=(200, 200)))
                associations = {int(row[4]): int(row[-1]) for row in current_tracks}
                self.assertEqual(associations[identity_a], 1)
                self.assertEqual(associations[identity_b], 0)
                self.assertEqual(len(set(associations.values())), 2)


if __name__ == "__main__":
    unittest.main()
