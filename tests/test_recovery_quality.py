"""Conservative recovery cue regressions; native boxes remain observable."""
import unittest

import cv2
import numpy as np

from driftlens.pipeline import VehicleClassRecovery, select_observed_detections


class RecoveryQualityTests(unittest.TestCase):
    @staticmethod
    def fixture(hues=(60, 150), current=False):
        frame = np.full((180, 240, 3), 80, dtype=np.uint8)
        box = (55, 42, 95, 122) if current else (50, 40, 90, 120)
        x1, y1, x2, y2 = box
        crop = np.full((80, 40, 3), (0, 0, 240), dtype=np.uint8)
        crop[:40, :, :] = (hues[0], 220, 230)
        crop[40:70, :, :] = (hues[1], 220, 230)
        frame[y1:y2, x1:x2] = cv2.cvtColor(crop, cv2.COLOR_HSV2BGR)
        row = dict(zip(("x1", "y1", "x2", "y2"), box))
        row.update(confidence=0.8, **{"class": 2}, track_id=3,
                   detection_method="native_detector")
        return frame, row

    def test_red_and_white_curb_does_not_seed_recovery_but_native_box_remains(self):
        recovery = VehicleClassRecovery()
        frame, native = self.fixture((0, 0))
        recovery.observe(frame, [native], 0)
        self.assertEqual(recovery.anchors, {})
        current, candidate = self.fixture((0, 0), current=True)
        candidate.update({"class": 5, "confidence": 0.48})
        candidate.pop("track_id")
        self.assertEqual(recovery.recover(current, [], [candidate], 0.1), [])
        chosen = select_observed_detections([native], 0.5, True)
        self.assertEqual(len(chosen), 1)
        self.assertEqual(chosen[0]["class"], 2)
        self.assertEqual(chosen[0]["detection_method"], "native_detector")

    def test_red_hue_wrap_is_one_color_family(self):
        recovery = VehicleClassRecovery()
        frame, seed = self.fixture((1, 179))
        recovery.observe(frame, [seed], 0)
        self.assertEqual(recovery.anchors, {})
        current, candidate = self.fixture((179, 1), current=True)
        candidate.update({"class": 0, "confidence": 0.48})
        self.assertEqual(recovery.recover(current, [], [candidate], 0.1), [])

    def test_tiny_color_noise_does_not_turn_curb_into_identity_cue(self):
        recovery = VehicleClassRecovery()
        frame, seed = self.fixture((0, 0))
        frame[50:54, 60:64] = (255, 0, 0)
        recovery.observe(frame, [seed], 0)
        self.assertEqual(recovery.anchors, {})

    def test_multicolor_current_box_preserves_each_original_class_score_and_provenance(self):
        for predicted_class in (0, 5, 39, 67):
            with self.subTest(predicted_class=predicted_class):
                recovery = VehicleClassRecovery()
                frame, seed = self.fixture()
                recovery.observe(frame, [seed], 0)
                original_fingerprint = recovery.anchors[3]["appearance"].copy()
                current, candidate = self.fixture(current=True)
                candidate.update({"class": predicted_class, "confidence": 0.48})
                candidate.pop("track_id")
                recovered = recovery.recover(current, [], [candidate], 0.1)
                self.assertEqual(len(recovered), 1)
                observed = recovered[0]
                self.assertEqual(observed["source_class_id"], predicted_class)
                self.assertEqual(observed["confidence"], 0.48)
                self.assertEqual(observed["class"], 2)
                self.assertEqual(observed["recovery_anchor_track_id"], 3)
                self.assertEqual(observed["detection_method"], "temporal_appearance_class_recovery")
                self.assertEqual([observed[key] for key in ("x1", "y1", "x2", "y2")],
                                 [candidate[key] for key in ("x1", "y1", "x2", "y2")])
                observed["track_id"] = 3
                recovery.observe(current, [observed], 0.1)
                np.testing.assert_array_equal(recovery.anchors[3]["appearance"], original_fingerprint)


if __name__ == "__main__":
    unittest.main()
