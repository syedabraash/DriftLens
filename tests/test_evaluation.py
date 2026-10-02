import unittest

from driftlens.evaluation import evaluate_samples, intersection_over_union, match_boxes


def frame(index, time, shot=0, clip="test"):
    return {"clip_id": clip, "frame_index": index, "clip_time": time, "source_time": 30 + time, "shot_index": shot}


def observation(index, time, track=1, bbox=(10, 10, 30, 30), shot=0, clip="test", observed=True):
    return {**frame(index, time, shot, clip), "track_id": track, "x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3], "observed": observed, "confidence": 0.8}


def sample(time, cars=None, clip="test", **extra):
    return {"clip_id": clip, "time_seconds": time, "cars": [{"identity": "car_a", "bbox": [10, 10, 30, 30], "visibility": "visible"}] if cars is None else cars, **extra}


def labels(*samples):
    return {"coordinate_space": "source_pixels", "samples": list(samples)}


class EvaluationTests(unittest.TestCase):
    def test_one_prediction_cannot_match_two_cars(self):
        boxes = [[0, 0, 20, 20], [0, 0, 20, 20]]
        self.assertEqual(len(match_boxes(boxes, [[0, 0, 20, 20]])), 1)
        self.assertEqual(intersection_over_union([0, 0, 20, 20], [30, 30, 40, 40]), 0)

    def test_duplicate_detection_is_a_false_positive(self):
        result = evaluate_samples(labels(sample(0)), [observation(0, 0), observation(0, 0, 2)])
        self.assertEqual(result["counts"]["true_positives"], 1)
        self.assertEqual(result["counts"]["false_positives"], 1)
        self.assertEqual(result["detection"]["precision"], 0.5)
        self.assertEqual(result["detection"]["recall"], 1)

    def test_empty_processed_frame_is_a_miss(self):
        result = evaluate_samples(labels(sample(0)), [], [frame(0, 0)])
        self.assertEqual(result["counts"]["false_negatives"], 1)
        self.assertEqual(result["detection"]["recall"], 0)
        self.assertIsNone(result["detection"]["precision"])
        self.assertEqual(result["tracking"]["visible_ground_truth_coverage"], 0)

    def test_empty_scene_and_empty_annotation_have_undefined_rates(self):
        result = evaluate_samples(labels(sample(0, [])), [], [frame(0, 0)])
        self.assertIsNone(result["detection"]["precision"])
        self.assertIsNone(result["detection"]["recall"])
        self.assertIsNone(result["detection"]["f1"])
        empty = evaluate_samples(labels(), [])
        self.assertEqual(empty["counts"]["labelled_frames"], 0)

    def test_identity_switch_only_between_comparable_labels(self):
        result = evaluate_samples(labels(sample(0), sample(1), sample(2)), [observation(0, 0, 1), observation(1, 1, 1), observation(2, 2, 2)])
        self.assertEqual(result["tracking"]["identity_switches"], 1)
        self.assertEqual(result["tracking"]["comparable_identity_transitions"], 2)
        self.assertEqual(result["switch_events"][0]["identity"], "car_a")

    def test_camera_cut_breaks_identity_comparison(self):
        result = evaluate_samples(labels(sample(0), sample(1)), [observation(0, 0, 1), observation(1, 1, 2, shot=1)])
        self.assertEqual(result["tracking"]["identity_switches"], 0)
        self.assertEqual(result["tracking"]["comparable_identity_transitions"], 0)

    def test_missing_observation_breaks_identity_comparison(self):
        result = evaluate_samples(labels(sample(0), sample(1), sample(2)), [observation(0, 0, 1), observation(2, 2, 2)], [frame(0, 0), frame(1, 1), frame(2, 2)])
        self.assertEqual(result["counts"]["false_negatives"], 1)
        self.assertEqual(result["tracking"]["identity_switches"], 0)
        self.assertEqual(result["tracking"]["comparable_identity_transitions"], 0)

    def test_hidden_label_breaks_identity_comparison_and_excludes_denominator(self):
        hidden = [{"identity": "car_a", "visibility": "hidden"}]
        result = evaluate_samples(labels(sample(0), sample(1, hidden), sample(2)), [observation(0, 0, 1), observation(2, 2, 2)], [frame(0, 0), frame(1, 1), frame(2, 2)])
        self.assertEqual(result["counts"]["visible_ground_truth_cars"], 2)
        self.assertEqual(result["counts"]["excluded_hidden_cars"], 1)
        self.assertEqual(result["tracking"]["identity_switches"], 0)

    def test_unobserved_boxes_are_not_fabricated_detections(self):
        result = evaluate_samples(labels(sample(0)), [observation(0, 0, observed="false")], [frame(0, 0)])
        self.assertEqual(result["counts"]["true_positives"], 0)
        self.assertEqual(result["counts"]["false_negatives"], 1)

    def test_detection_without_track_id_is_not_tracking_coverage(self):
        result = evaluate_samples(labels(sample(0)), [observation(0, 0, track="")])
        self.assertEqual(result["counts"]["true_positives"], 1)
        self.assertEqual(result["tracking"]["visible_ground_truth_coverage"], 0)

    def test_unaligned_sample_is_visible_in_report_and_breaks_comparison(self):
        result = evaluate_samples(labels(sample(0), sample(0.5), sample(1)), [observation(0, 0, 1), observation(1, 1, 2)])
        self.assertEqual(result["counts"]["unaligned_frames"], 1)
        self.assertEqual(result["counts"]["aligned_frames"], 2)
        self.assertEqual(result["tracking"]["identity_switches"], 0)

    def test_coordinate_space_and_bad_boxes_are_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_samples({"coordinate_space": "preview_pixels", "samples": [sample(0)]}, [])
        with self.assertRaises(ValueError):
            evaluate_samples(labels(sample(0, [{"identity": "car_a", "bbox": [2, 2, 1, 1]}])), [observation(0, 0)])
        with self.assertRaises(ValueError):
            evaluate_samples(labels(sample(0)), [observation(0, 0, bbox=(float("nan"), 10, 30, 30))])

    def test_distinct_clips_never_share_identity_state(self):
        result = evaluate_samples(labels(sample(0, clip="one"), sample(0, clip="two")), [observation(0, 0, 1, clip="one"), observation(0, 0, 2, clip="two")])
        self.assertEqual(result["tracking"]["comparable_identity_transitions"], 0)

    def test_labels_cannot_count_one_frame_twice(self):
        with self.assertRaises(ValueError):
            evaluate_samples(labels(sample(0), sample(0.05)), [observation(0, 0)])


if __name__ == "__main__":
    unittest.main()
