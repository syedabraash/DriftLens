"""Cut discontinuity tests including same-histogram scenes and camera motion."""
import unittest

import cv2
import numpy as np

from driftlens.camera_cuts import CameraCutDetector, compare_samples


def scene(seed):
    rng = np.random.default_rng(seed)
    image = rng.integers(30, 200, (108, 192), dtype=np.uint8)
    return cv2.GaussianBlur(image, (3, 3), 0)


def textured_scene(seed):
    rng = np.random.default_rng(seed)
    image = np.full((108, 192), 110, dtype=np.uint8)
    for _ in range(75):
        point = tuple(map(int, rng.integers((3, 3), (182, 98))))
        color = int(rng.choice((30, 190)))
        cv2.rectangle(image, point, (point[0]+8, point[1]+8), color, -1)
    return image


class CameraCutTests(unittest.TestCase):
    def test_initial_frame_and_reset_do_not_make_a_cut(self):
        detector = CameraCutDetector()
        first = textured_scene(1)
        self.assertFalse(detector.update(first))
        self.assertFalse(detector.update(first))
        detector.reset()
        self.assertIsNone(detector.last_metrics)
        self.assertFalse(detector.update(textured_scene(3)))

    def test_unrelated_views_with_same_histogram_are_a_cut(self):
        # Reordering source pixels preserves the histogram exactly; the old
        # global histogram gate cannot distinguish these two camera structures.
        first = textured_scene(12)
        second = np.flip(first, (0, 1)).copy()
        np.testing.assert_array_equal(np.bincount(first.ravel(), minlength=256), np.bincount(second.ravel(), minlength=256))
        detector = CameraCutDetector()
        self.assertFalse(detector.update(first))
        self.assertTrue(detector.update(second))
        self.assertFalse(detector.update(second))

    def test_pan_and_zoom_keep_motion_continuity_despite_large_pixel_change(self):
        image = textured_scene(51)
        transform = cv2.getRotationMatrix2D((96, 54), 2, 1.04)
        transform[:, 2] += (7, -2)
        current = cv2.warpAffine(image, transform, (192, 108), borderMode=cv2.BORDER_REFLECT)
        metrics = compare_samples(image, current)
        self.assertGreater(metrics["structure_residual"], .10)
        self.assertGreater(metrics["motion_inlier_fraction"], .10)
        self.assertFalse(metrics["is_cut"])

    def test_exposure_change_does_not_make_a_cut(self):
        detector = CameraCutDetector()
        image = scene(8)
        detector.update(image)
        self.assertFalse(detector.update((image.astype(np.int16)+35).astype(np.uint8)))
        self.assertLess(detector.last_metrics["structure_residual"], 1e-6)

    def test_small_overlapping_foreground_change_is_not_a_camera_cut(self):
        image = textured_scene(22)
        overlap = image.copy()
        # A local occluding or merged foreground object does not change the view.
        cv2.rectangle(overlap, (65, 50), (113, 81), 240, -1)
        detector = CameraCutDetector()
        detector.update(image)
        self.assertFalse(detector.update(overlap))

    def test_large_foreground_occlusion_retains_background_motion_support(self):
        image = textured_scene(63)
        occluded = image.copy()
        cv2.rectangle(occluded, (40, 35), (158, 108), 240, -1)
        metrics = compare_samples(image, occluded)
        self.assertGreater(metrics["structure_residual"], .10)
        self.assertGreater(metrics["motion_inlier_fraction"], .10)
        self.assertFalse(metrics["is_cut"])

    def test_flat_exposure_change_and_source_shape_validation(self):
        detector = CameraCutDetector()
        detector.update(np.full((720, 1280, 3), 80, dtype=np.uint8))
        self.assertFalse(detector.update(np.full((720, 1280, 3), 160, dtype=np.uint8)))
        with self.assertRaises(ValueError):
            detector.update(np.empty((0, 0), dtype=np.uint8))

    def test_stationary_player_controls_do_not_hide_a_content_cut(self):
        first = textured_scene(101)
        second = np.flip(first, (0, 1)).copy()
        # Dense perfectly static control icons occupy the bottom screen strip.
        # They provide real optical-flow matches, but say nothing about whether
        # the racing camera changed above them.
        controls = textured_scene(6)[86:]
        first[86:] = controls
        second[86:] = controls
        detector = CameraCutDetector()
        detector.update(first)
        self.assertTrue(detector.update(second))

    def test_screen_recorded_pan_still_keeps_interior_motion_support(self):
        first = textured_scene(51)
        transform = cv2.getRotationMatrix2D((96, 54), 2, 1.04)
        transform[:, 2] += (7, -2)
        current = cv2.warpAffine(first, transform, (192, 108), borderMode=cv2.BORDER_REFLECT)
        controls = textured_scene(6)[86:]
        first[86:] = controls
        current[86:] = controls
        self.assertFalse(compare_samples(first, current)["is_cut"])


if __name__ == "__main__":
    unittest.main()
