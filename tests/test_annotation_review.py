"""Annotation review must not fabricate provenance or leak source groups."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from driftlens.annotation_review import (build_review_queue, confirm_sample, export_detector_labels,
                                        export_reviewed_labels, import_review_queue,
                                        load_review_queue, save_review_queue, validate_review_queue)
from driftlens.evaluation import evaluate_samples


def fixture():
    labels = {"coordinate_space": "source_pixels", "width": 100, "height": 60,
              "annotation_method": "Approximate assistant draft; no human expert review",
              "samples": [{"clip_id": "clip1", "time_seconds": .3, "frame_index": 3,
                           "frame": "outputs/inspection/frame.jpg",
                           "cars": [{"identity": "car_a", "bbox": [10, 10, 40, 30], "visibility": "visible"}]}]}
    catalog = {"source": {"path": "source.mp4", "reuse_status": "unverified"},
               "clips": [{"id": "clip1", "split": "test", "run_group": "battle1"}]}
    return build_review_queue(labels, catalog), labels, catalog


def confirmed(queue=None, **kwargs):
    queue = queue or fixture()[0]
    record = queue["records"][0]
    return confirm_sample(queue, record["sample_id"], kwargs.pop("cars", record["cars"]),
                          kwargs.pop("tags", {"smoke": "light", "overlap": "partial"}),
                          kwargs.pop("reviewer", "Test reviewer"),
                          kwargs.pop("confirmations", {"image": True, "boxes": True, "identities": True}),
                          reviewed_at="2026-10-03T09:00:00+00:00", **kwargs)


class AnnotationReviewTests(unittest.TestCase):
    def test_assistant_drafts_are_pending_and_do_not_export(self):
        queue, labels, _ = fixture()
        self.assertEqual(queue["records"][0]["provenance"], "assistant_draft")
        self.assertIsNone(queue["records"][0]["review"])
        with self.assertRaisesRegex(ValueError, "No human confirmed"):
            export_reviewed_labels(queue)
        queue["records"][0]["cars"][0]["bbox"][0] = 12
        self.assertEqual(labels["samples"][0]["cars"][0]["bbox"][0], 10)
        self.assertEqual(queue["records"][0]["original_cars"][0]["bbox"][0], 10)

    def test_explicit_confirmation_and_reviewer_are_required(self):
        for reviewer, confirmations in [("", {"image": True, "boxes": True, "identities": True}),
                                         ("Reviewer", {"image": True, "boxes": True, "identities": False}),
                                         ("Reviewer", {"image": 1, "boxes": True, "identities": True})]:
            with self.assertRaises(ValueError):
                confirmed(reviewer=reviewer, confirmations=confirmations)

    def test_invalid_extent_class_identity_and_visibility_are_rejected(self):
        car = fixture()[0]["records"][0]["cars"][0]
        for changes in [{"bbox": [-1, 1, 10, 10]}, {"bbox": [1, 1, 101, 10]},
                        {"bbox": [1, 1, 1, 10]}, {"bbox": [1, 1, float("nan"), 10]},
                        {"class_id": 0}, {"class_id": True}, {"identity": "run:car_a"},
                        {"identity": "123"}, {"visibility": "hidden"}, {"visibility": "unknown"}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                confirmed(cars=[{**car, **changes}])
        with self.assertRaisesRegex(ValueError, "unique"):
            confirmed(cars=[car, car])

    def test_hidden_car_has_no_box_and_is_excluded_by_real_evaluator(self):
        queue = confirmed(cars=[{"identity": "car_a", "class_id": 2, "visibility": "hidden", "bbox": None}])
        labels = export_reviewed_labels(queue)
        report = evaluate_samples(labels, [], [{"clip_id": "clip1", "frame_index": 3, "clip_time": .3, "shot_index": 0}])
        self.assertEqual(report["counts"]["excluded_hidden_cars"], 1)
        self.assertEqual(report["counts"]["visible_ground_truth_cars"], 0)
        self.assertIsNone(report["detection"]["recall"])

    def test_human_export_is_evaluator_compatible(self):
        queue = confirmed()
        labels = export_reviewed_labels(queue, "test")
        observation = {"clip_id": "clip1", "frame_index": 3, "clip_time": .3, "shot_index": 0,
                       "track_id": 8, "x1": 10, "y1": 10, "x2": 40, "y2": 30}
        report = evaluate_samples(labels, [observation], [observation])
        self.assertEqual(report["counts"]["true_positives"], 1)
        self.assertEqual(labels["samples"][0]["review"]["reviewer"], "Test reviewer")
        self.assertEqual(labels["samples"][0]["identity_scope"], "clip1:shot0")

    def test_changes_invalidate_review_attestation(self):
        queue = confirmed()
        queue["records"][0]["cars"][0]["bbox"][0] = 11
        with self.assertRaisesRegex(ValueError, "changed"):
            validate_review_queue(queue)

    def test_review_timestamp_needs_timezone_and_import_respects_actual_time(self):
        queue = fixture()[0]
        record = queue["records"][0]
        args = (queue, record["sample_id"], record["cars"], record["tags"], "Test reviewer",
                {"image": True, "boxes": True, "identities": True})
        for timestamp in ("2026-10-03T09:00:00", "yesterday", 3):
            with self.subTest(timestamp=timestamp), self.assertRaisesRegex(ValueError, "timestamp"):
                confirm_sample(*args, reviewed_at=timestamp)
        original = confirm_sample(*args, reviewed_at="2026-10-03T09:00:00+00:00")
        older = confirm_sample(*args, reviewed_at="2026-10-03T10:00:00+05:00")
        with self.assertRaisesRegex(ValueError, "newer review"):
            import_review_queue(original, older)

    def test_invalid_identity_scope_or_repeated_frame_is_rejected(self):
        queue = fixture()[0]
        queue["records"][0]["identity_scope"] = "physical_car_across_cameras"
        with self.assertRaisesRegex(ValueError, "local"):
            validate_review_queue(queue)
        queue = fixture()[0]
        duplicate = deepcopy(queue["records"][0])
        duplicate["sample_id"] = "another_id_for_same_frame"
        queue["records"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_review_queue(queue)

    def test_battle_and_video_split_leakage_are_blocked(self):
        _, labels, catalog = fixture()
        second = deepcopy(labels["samples"][0])
        second["clip_id"] = "clip2"
        labels["samples"].append(second)
        catalog["clips"].append({"id": "clip2", "split": "train", "run_group": "battle1"})
        with self.assertRaisesRegex(ValueError, "leakage"):
            build_review_queue(labels, catalog)
        catalog["clips"][1]["run_group"] = "battle2"
        build_review_queue(labels, catalog)
        with self.assertRaisesRegex(ValueError, "leakage"):
            build_review_queue(labels, catalog, group_by="video")

    def test_import_preserves_known_source_frame_context(self):
        queue = fixture()[0]
        imported = import_review_queue(queue, confirmed(queue))
        self.assertEqual(imported["records"][0]["provenance"], "human_self_attested")
        self.assertIsNone(queue["records"][0]["review"])
        wrong = deepcopy(queue)
        wrong["records"][0]["battle_group"] = "another_battle"
        with self.assertRaisesRegex(ValueError, "immutable"):
            import_review_queue(queue, wrong)
        wrong = deepcopy(queue)
        wrong["original_reference"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "provenance"):
            import_review_queue(queue, wrong)

    def test_persistence_and_frozen_file_protection(self):
        queue = confirmed()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "queue.json"
            save_review_queue(path, queue)
            self.assertEqual(load_review_queue(path), queue)
            with self.assertRaisesRegex(ValueError, "frozen"):
                save_review_queue(Path(folder) / "data/annotations/labels.json", queue)
            self.assertFalse(list(Path(folder).glob("*.tmp")))

    def test_detector_export_retains_classes_and_excludes_hidden_boxes(self):
        queue = confirmed(cars=[{"identity": "car_a", "class_id": 2, "bbox": None, "visibility": "hidden"},
                                {"identity": "truck_a", "class_id": 7, "bbox": [5, 5, 20, 15], "visibility": "partial"}])
        result = export_detector_labels(queue, "test")
        self.assertFalse(result["training_allowed"])
        self.assertEqual(result["samples"][0]["boxes"], [{"class_id": 7, "bbox": [5., 5., 20., 15.]}])
        self.assertEqual(export_reviewed_labels(queue)["samples"][0]["cars"][0]["identity"], "car_a")

    def test_invalid_tag_and_outside_image_paths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "smoke"):
            confirmed(tags={"smoke": "guessed", "overlap": "none"})
        _, labels, catalog = fixture()
        labels["samples"][0]["frame"] = "../another_project/image.jpg"
        with self.assertRaisesRegex(ValueError, "relative"):
            build_review_queue(labels, catalog)


if __name__ == "__main__":
    unittest.main()
