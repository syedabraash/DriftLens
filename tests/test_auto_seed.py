"""Semantic tests for automatic seed selection and conservative abstention."""
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from driftlens import auto_seed
except ImportError:
    spec = importlib.util.spec_from_file_location("auto_seed", Path(__file__).with_name("auto_seed.py"))
    auto_seed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auto_seed)


def fixture(ids=(99, 3), *, moving=True, with_parked=False, overlap=False):
    frames, observations = [], []
    for index in range(60):
        time = index / 10
        frames.append({"frame_index": index, "clip_time": time, "source_time": time, "shot_index": 0})
        movement = time * 20 if moving else 0
        for ident, x in zip(ids, (180, 80 if not overlap else 177)):
            observations.append({"frame_index": index, "clip_time": time, "source_time": time, "shot_index": 0,
                                 "track_id": ident, "x1": x + movement, "y1": 80, "x2": x + movement + 40,
                                 "y2": 110, "confidence": .8, "observed": True})
        if with_parked:
            observations.append({"frame_index": index, "clip_time": time, "source_time": time, "shot_index": 0,
                                 "track_id": 1, "x1": 400, "y1": 50, "x2": 440, "y2": 80,
                                 "confidence": .8, "observed": True})
    return frames, observations


class IdentityBackground:
    decoded_frames = 0
    def __init__(self, *args):
        pass
    def close(self):
        pass
    def estimate(self, *args):
        return {"valid": True, "affine": [[1, 0, 0], [0, 1, 0]], "background_inliers": 60}


class MissingBackground(IdentityBackground):
    def estimate(self, *args):
        return {"valid": False, "reason": "Weak background support"}


class PanningBackground(IdentityBackground):
    def estimate(self, *args):
        return {"valid": True, "affine": [[1, 0, 20], [0, 1, 0]], "background_inliers": 60}


class ConflictingBackground(IdentityBackground):
    def __init__(self, *args):
        self.count = 0
    def estimate(self, *args):
        self.count += 1
        # One valid motion direction followed by its opposite: cannot order roles.
        shift = 0 if self.count % 2 else 40
        return {"valid": True, "affine": [[1, 0, shift], [0, 1, 0]], "background_inliers": 60}


class TestAutomaticSeed(unittest.TestCase):
    def run_proposal(self, frames, rows, background=IdentityBackground):
        with patch.object(auto_seed, "_BackgroundMotion", background):
            return auto_seed.propose_automatic_seed("unused.mp4", frames, rows, 6)

    def test_order_depends_on_travel_not_id_numbers(self):
        frames, rows = fixture()
        result = self.run_proposal(frames, rows)
        self.assertEqual(result["status"], "seed_ready")
        self.assertEqual(result["seed"]["lead_id"], 99)
        self.assertEqual(result["seed"]["chase_id"], 3)
        self.assertEqual(result["evidence"]["review_status"], "automatic_geometry_unverified")
        self.assertIn("no user or expert", result["seed"]["review_basis"])

    def test_parked_track_does_not_compete_with_moving_pair(self):
        frames, rows = fixture(with_parked=True)
        result = self.run_proposal(frames, rows)
        self.assertEqual(result["status"], "seed_ready")
        self.assertNotIn(1, result["pair_ids"])

    def test_stationary_scene_abstains(self):
        frames, rows = fixture(moving=False)
        result = self.run_proposal(frames, rows)
        self.assertEqual(result["status"], "unavailable")
        self.assertIsNone(result["seed"])

    def test_camera_pan_does_not_turn_parked_cars_into_tandem(self):
        frames, rows = fixture()
        result = self.run_proposal(frames, rows, PanningBackground)
        self.assertEqual(result["status"], "unavailable")
        self.assertIsNone(result["seed"])

    def test_unknown_camera_motion_never_infers_raw_screen_order(self):
        frames, rows = fixture()
        result = self.run_proposal(frames, rows, MissingBackground)
        self.assertEqual(result["status"], "pair_only")
        self.assertCountEqual(result["pair_ids"], [99, 3])
        self.assertIsNone(result["seed"])

    def test_conflicting_travel_order_is_withheld(self):
        frames, rows = fixture()
        result = self.run_proposal(frames, rows, ConflictingBackground)
        self.assertEqual(result["status"], "pair_only")
        self.assertIsNone(result["seed"])

    def test_competing_pairs_abstain_instead_of_choosing_lowest_ids(self):
        frames, rows = fixture()
        for frame in frames:
            time = frame["clip_time"]
            rows.append({**frame, "track_id": 55, "x1": 350 + time * 20, "x2": 390 + time * 20,
                         "y1": 160, "y2": 190, "observed": True})
        result = self.run_proposal(frames, rows)
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("Competing", result["reason"])

    def test_merged_seed_pair_is_not_accepted(self):
        frames, rows = fixture(overlap=True)
        result = self.run_proposal(frames, rows)
        self.assertEqual(result["status"], "pair_only")
        self.assertIn("clear seed", result["reason"])

    def test_missing_samples_do_not_create_synthetic_seed_observations(self):
        frames, rows = fixture()
        rows = [row for row in rows if row["track_id"] != 3 or row["frame_index"] % 2 == 0]
        result = self.run_proposal(frames, rows)
        self.assertEqual(result["status"], "unavailable")
        self.assertIsNone(result["seed"])

    def test_nonobserved_predictions_are_excluded(self):
        frames, rows = fixture()
        for row in rows:
            if row["track_id"] == 3:
                row["observed"] = False
        result = self.run_proposal(frames, rows)
        self.assertEqual(result["status"], "unavailable")

    def test_seed_and_motion_never_cross_camera_cut(self):
        frames, rows = fixture()
        for collection in (frames, rows):
            for row in collection:
                if row["clip_time"] >= 1.5:
                    row["shot_index"] = 1
        result = self.run_proposal(frames, rows)
        self.assertEqual(result["status"], "pair_only")
        self.assertIsNone(result["seed"])
        self.assertTrue(all(row["end_clip_seconds"] < 1.5 for row in result["evidence"]["motion_windows"]))

    def test_invalid_boxes_are_not_role_seeds(self):
        frames, rows = fixture()
        rows[0]["x1"] = float("nan")
        result = self.run_proposal(frames, rows)
        self.assertEqual(result["status"], "unavailable")

    def test_duplicate_current_track_is_rejected(self):
        frames, rows = fixture()
        rows.append(dict(rows[0]))
        result = self.run_proposal(frames, rows)
        self.assertEqual(result["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
