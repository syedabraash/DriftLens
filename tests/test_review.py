import unittest

from driftlens.review import derive_frame_metrics


def frame(index, time, shot=0):
    return {"frame_index": index, "clip_time": time, "source_time": 30 + time, "shot_index": shot}


def observation(index, time, track, box, observed=True, shot=0):
    return {**frame(index, time, shot), "track_id": track, "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3], "confidence": 0.9, "observed": observed}


class ReviewMetricTests(unittest.TestCase):
    def test_missing_car_produces_unknown_separation(self):
        observations = [observation(0, 0, 1, [0, 0, 20, 20]), observation(0, 0, 2, [50, 0, 70, 20]), observation(1, 1, 1, [10, 0, 30, 20])]
        result = derive_frame_metrics(observations, [frame(0, 0), frame(1, 1), frame(2, 2)], 1, 2)
        self.assertEqual(len(result), 3)
        self.assertTrue(result[0]["pair_observed"])
        self.assertIsNotNone(result[0]["separation_proxy"])
        self.assertGreaterEqual(result[0]["separation_proxy"], 0)
        self.assertFalse(result[1]["pair_observed"])
        self.assertIsNone(result[1]["separation_proxy"])
        self.assertIsNone(result[2]["separation_proxy"])

    def test_unobserved_prediction_cannot_fill_the_gap(self):
        observations = [observation(0, 0, 1, [0, 0, 20, 20]), observation(0, 0, 2, [50, 0, 70, 20], observed=False)]
        result = derive_frame_metrics(observations, [frame(0, 0)], 1, 2)
        self.assertTrue(result[0]["lead_observed"])
        self.assertFalse(result[0]["chase_observed"])
        self.assertIsNone(result[0]["separation_proxy"])

    def test_roles_must_be_distinct(self):
        with self.assertRaises(ValueError):
            derive_frame_metrics([], [frame(0, 0)], 1, 1)

    def test_unassigned_roles_have_no_separation(self):
        observations = [observation(0, 0, 1, [0, 0, 20, 20]), observation(0, 0, 2, [50, 0, 70, 20])]
        result = derive_frame_metrics(observations, [frame(0, 0)], None, None)
        self.assertIsNone(result[0]["separation_proxy"])
        self.assertFalse(result[0]["pair_observed"])

    def test_frame_manifest_preserves_timing_and_camera_cuts(self):
        result = derive_frame_metrics([], [frame(0, 0), frame(10, 1, shot=1)], 1, 2)
        self.assertEqual(result[1]["frame_index"], 10)
        self.assertEqual(result[1]["source_time"], 31)
        self.assertEqual(result[1]["shot_index"], 1)
        self.assertIsNone(result[1]["separation_proxy"])


if __name__ == "__main__":
    unittest.main()
