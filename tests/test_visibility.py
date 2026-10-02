"""Regression checks for observed-only visibility profiles and augmentation."""
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from driftlens.pipeline import (restore_rotated_box, resolve_tracker_settings,
                               suppress_duplicate_detections,
                               validate_inference_profile, validate_orientations,
                               VehicleClassRecovery, select_observed_detections)
from driftlens.tracker_adapter import bind_adapter


class VisibilityTests(unittest.TestCase):
    @staticmethod
    def recovery_fixture(box=(50, 40, 90, 120), other_color=False):
        frame = np.full((180, 240, 3), 40, dtype=np.uint8)
        x1, y1, x2, y2 = box
        frame[y1:y2, x1:x2] = (255, 90, 0) if other_color else (20, 230, 20)
        frame[y1:y1 + 25, x1:x2] = (0, 90, 255) if other_color else (180, 20, 210)
        row = dict(zip(("x1", "y1", "x2", "y2"), box))
        row.update(confidence=0.8, **{"class": 2}, track_id=3, detection_method="native_detector")
        return frame, row

    def test_current_misclassified_box_recovers_with_original_confidence_and_provenance(self):
        recovery = VehicleClassRecovery()
        frame, seed = self.recovery_fixture()
        recovery.observe(frame, [seed], 0)
        feature = recovery.anchors[3]["appearance"].copy()
        current, candidate = self.recovery_fixture((55, 42, 95, 122))
        candidate.update({"class": 39, "confidence": 0.48})
        candidate.pop("track_id")
        recovered = recovery.recover(current, [], [candidate], 0.1)
        self.assertEqual(len(recovered), 1)
        self.assertEqual([recovered[0][key] for key in ("x1", "y1", "x2", "y2")], [55, 42, 95, 122])
        self.assertEqual(recovered[0]["confidence"], 0.48)
        self.assertEqual(recovered[0]["source_class_id"], 39)
        self.assertEqual(recovered[0]["class"], 2)
        self.assertEqual(recovered[0]["vehicle_anchor_class"], 2)
        recovered[0]["track_id"] = 3
        recovery.observe(current, recovered, 0.1)
        np.testing.assert_array_equal(recovery.anchors[3]["appearance"], feature)
        self.assertEqual(recovery.anchors[3]["last_seen"], 0.1)

    def test_recovery_rejects_wrong_appearance_stale_seed_and_wrong_geometry(self):
        for failure in ("appearance", "stale", "geometry", "no_seed"):
            with self.subTest(failure=failure):
                recovery = VehicleClassRecovery()
                frame, seed = self.recovery_fixture()
                if failure != "no_seed":
                    recovery.observe(frame, [seed], 0)
                box = (20, 40, 140, 120) if failure == "geometry" else (55, 42, 95, 122)
                current, candidate = self.recovery_fixture(box, other_color=failure == "appearance")
                candidate["class"] = 39
                self.assertEqual(recovery.recover(current, [], [candidate], 0.6 if failure == "stale" else 0.1), [])

    def test_recovery_requires_unambiguous_one_to_one_current_candidate(self):
        recovery = VehicleClassRecovery()
        frame, seed = self.recovery_fixture()
        recovery.observe(frame, [seed], 0)
        current, candidate = self.recovery_fixture((55, 42, 95, 122))
        candidate["class"] = 39
        competing = {**candidate, "class": 67}
        self.assertEqual(recovery.recover(current, [], [candidate, competing], 0.1), [])
        self.assertEqual(recovery.recover(current, [candidate], [candidate], 0.1), [])
        self.assertEqual(recovery.recover(current, [], [], 0.1), [])
        # Two still-recent seeds for one candidate are an identity ambiguity.
        recovery.anchors[4] = dict(recovery.anchors[3])
        self.assertEqual(recovery.recover(current, [], [candidate], 0.1), [])
        recovery.reset()
        self.assertEqual(recovery.recover(current, [], [candidate], 0.1), [])

    def test_profiles_cannot_download_or_select_unapproved_weights(self):
        for model in ("../yolov8n.pt", "https://example.com/model.pt", "driftlens_finetuned.pt"):
            with self.subTest(model=model), self.assertRaises(ValueError):
                validate_inference_profile(model, 0.15, {})
        for value in (float("nan"), float("inf"), -0.1, 0, 1, True):
            with self.subTest(confidence=value), self.assertRaises(ValueError):
                validate_inference_profile("yolov8n.pt", value, {})
        self.assertEqual(validate_inference_profile("yolov8n.pt", 0.15, None), {})

    def test_tracker_options_remain_typed_and_backend_specific(self):
        for options in ({"new_track_thresh": float("nan")}, {"with_reid": "true"}, {"track_buffer": -1}, {"model": "another-model.pt"}, {"tracker_type": "custom"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                validate_inference_profile("yolov8n.pt", 0.15, options)
        with self.assertRaises(ValueError):
            resolve_tracker_settings("bytetrack", {"with_reid": True})
        with self.assertRaises(ValueError):
            resolve_tracker_settings("botsort", {"track_low_thresh": 0.4})
        self.assertEqual(resolve_tracker_settings("botsort", {})["new_track_thresh"], 0.25)

    def test_rotated_boxes_restore_original_non_square_source_coordinates(self):
        original = [10, 20, 40, 60]
        rotated = {0: original, 90: [40, 10, 80, 40], 180: [160, 40, 190, 80], 270: [20, 160, 60, 190]}
        for degrees, box in rotated.items():
            self.assertEqual(restore_rotated_box(box, 200, 100, degrees), original)

    def test_orientation_passes_are_bounded_and_native_features_do_not_mix(self):
        self.assertEqual(validate_orientations(None, {}), [0])
        self.assertEqual(validate_orientations([0, 90, 270], {}), [0, 90, 270])
        for values in ([90], [0, 90, 180, 270], [0, 0], [0, 45], [False]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_orientations(values, {})
        with self.assertRaises(ValueError):
            validate_orientations([0, 90], {"with_reid": True})

    def test_augmentation_nms_deduplicates_classes_without_merging_separate_cars(self):
        boxes = np.array([[10, 20, 40, 60, 0.8, 2], [10, 20, 40, 60, 0.7, 7], [70, 20, 100, 60, 0.6, 2]], dtype=np.float32)
        merged = suppress_duplicate_detections(boxes, 0.5, True)
        np.testing.assert_array_equal(merged, boxes[[0, 2]])
        np.testing.assert_array_equal(suppress_duplicate_detections(boxes, 0.5, False), boxes)
        self.assertEqual(suppress_duplicate_detections(np.empty((0, 6)), 0.5, True).shape, (0, 6))

    def test_nms_preserves_exact_chosen_provenance_after_float32_rounding(self):
        raw = {"x1": 0.123456789, "y1": 3.987654321, "x2": 40.987654321, "y2": 30.123456789, "confidence": 0.223456789, "class": 2, "source_class_id": 39, "detection_method": "temporal_appearance_class_recovery", "appearance_similarity": 0.91}
        duplicate = {**raw, "source_class_id": 2, "detection_method": "orientation_detector"}
        chosen = select_observed_detections([raw, duplicate], 0.5, True)
        self.assertEqual(len(chosen), 1)
        self.assertEqual(chosen[0]["source_class_id"], 39)
        self.assertEqual(chosen[0]["appearance_similarity"], 0.91)
        self.assertEqual(chosen[0]["detection_method"], "temporal_appearance_class_recovery")
        # This exercises the exact adapter equality used by the real backend.
        from driftlens.tracker_adapter import map_original_detection_indices
        array = np.array([[chosen[0][key] for key in ("x1", "y1", "x2", "y2", "confidence", "class")]], dtype=np.float32)
        self.assertEqual(map_original_detection_indices(array[:, :4], array[:, 4], chosen, array[:, 5]), [0])
        with self.assertRaisesRegex(ValueError, "ambiguously"):
            map_original_detection_indices(array[:, :4], array[:, 4], chosen + chosen, array[:, 5])

    def test_real_backend_can_confirm_weak_observed_box_and_withholds_missing_frame(self):
        os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[1] / ".settings"))
        from ultralytics.engine.results import Boxes
        from ultralytics.trackers.bot_sort import BOTSORT
        outputs = []
        for overrides in ({"gmc_method": "none", "fuse_score": False}, {"gmc_method": "none", "fuse_score": False, "track_high_thresh": 0.15, "new_track_thresh": 0.15}):
            tracker = BOTSORT(SimpleNamespace(**resolve_tracker_settings("botsort", overrides)), frame_rate=30)
            raw = []
            bind_adapter(SimpleNamespace(trackers=[tracker]), raw)
            array = np.asarray([[20, 20, 60, 50, 0.18, 2]], dtype=np.float32)
            for _ in range(2):
                raw.clear()
                raw.extend(dict(zip(("x1", "y1", "x2", "y2", "confidence", "class"), map(float, row))) for row in array)
                current = tracker.update(Boxes(array, orig_shape=(100, 100)))
            outputs.append(len(current))
            if len(current):
                self.assertEqual(int(current[0, -1]), 0)
                self.assertEqual(tracker.tracked_stracks[0].idx, 0)
            raw.clear()
            missing = tracker.update(Boxes(np.empty((0, 6), dtype=np.float32), orig_shape=(100, 100)))
            self.assertEqual(len(missing), 0, "A lost Kalman position is never an observation")
        self.assertEqual(outputs, [0, 1])


if __name__ == "__main__":
    unittest.main()
