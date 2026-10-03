"""Observed-box identity association, cut resets, abstention and seed safety."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from driftlens.role_identity import (IdentitySettings, appearance_similarity,
                                    compress_decisions, describe_crop, suggest_role_intervals)


def make_car(role):
    image = np.full((70, 90, 3), (45, 45, 45) if role == 'lead' else (235, 235, 235), np.uint8)
    if role == 'lead':
        image[40:60, :35] = (180, 180, 20)
        image[55:, 40:] = (20, 20, 210)
    else:
        image[48:, :50] = (20, 180, 210)
        image[:15, 30:70] = (35, 35, 35)
    return image


def fixture(directory, variants=None):
    variants = variants or [{} for _ in range(8)]
    path = Path(directory) / 'source.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (320, 140))
    if not writer.isOpened():
        raise unittest.SkipTest('MJPG writer unavailable on this runtime.')
    frames, observations = [], []
    for index, variant in enumerate(variants):
        image = np.full((140, 320, 3), 120, np.uint8)
        frame = {'frame_index': index, 'clip_time': index / 10, 'source_time': index / 10,
                 'shot_index': variant.get('shot', 0 if index < 3 else 1)}
        frames.append(frame)
        roles = [('lead', variant.get('lead_id', 1 if index < 3 else 10001), (20, 30, 110, 100)),
                 ('chase', variant.get('chase_id', 2 if index < 3 else 10002), (190, 30, 280, 100))]
        for role, identity, box in roles:
            x1, y1, x2, y2 = box
            livery = variant.get(role + '_livery', role)
            crop = make_car(livery)
            if variant.get('flip'):
                crop = cv2.flip(crop, 1)
            image[y1:y2, x1:x2] = crop
            if variant.get('omit_' + role):
                continue
            observation = {**frame, 'track_id': identity, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                           'confidence': variant.get(role + '_confidence', .8), 'observed': True, 'class': 2}
            observations.append(observation)
            if variant.get('duplicate_' + role):
                duplicate = dict(observation, track_id=identity + 500)
                observations.append(duplicate)
        writer.write(image)
    writer.release()
    return path, frames, observations, [{'start_clip_seconds': 0, 'end_clip_seconds': .3, 'lead_id': 1, 'chase_id': 2}], len(frames) / 10


class IdentityAssociationTest(unittest.TestCase):
    def run_fixture(self, variants=None, change=None, settings=None):
        with tempfile.TemporaryDirectory() as directory:
            args = list(fixture(directory, variants))
            if change:
                change(args)
            # Single-sample fixtures isolate per-frame association and gates.
            return suggest_role_intervals(*args, settings=settings or IdentitySettings(min_track_support_samples=1))

    def test_reconnects_new_ids_after_cut_with_flipped_view(self):
        result = self.run_fixture([{}, {}, {}, {'flip': True}, {'flip': True}, {}, {}, {}])
        self.assertEqual((result['decisions'][3]['lead_id'], result['decisions'][3]['chase_id']), (10001, 10002))
        self.assertEqual(result['summary']['paired_frames'], 8)
        self.assertEqual(result['summary']['seed_observation_counts'], {'lead': 3, 'chase': 3})
        self.assertEqual(result['intervals'][1]['assignment_kind'], 'automatic_appearance')

    def test_fragmented_tracks_match_without_position_propagation(self):
        result = self.run_fixture([{}, {}, {}, {}, {'lead_id': 33, 'chase_id': 34}, {'lead_id': 39, 'chase_id': 41}])
        self.assertEqual((result['decisions'][5]['lead_id'], result['decisions'][5]['chase_id']), (39, 41))

    def test_reused_old_track_number_does_not_copy_old_role(self):
        result = self.run_fixture([{}, {}, {}, {'lead_id': 2, 'chase_id': 1}])
        self.assertEqual((result['decisions'][3]['lead_id'], result['decisions'][3]['chase_id']), (2, 1))

    def test_missing_observation_stays_unknown(self):
        result = self.run_fixture([{}, {}, {}, {'omit_chase': True}, {'omit_lead': True}])
        self.assertIsNone(result['decisions'][3]['chase_id'])
        self.assertIsNone(result['decisions'][4]['lead_id'])
        self.assertNotEqual(result['decisions'][3]['chase_reason'], 'appearance_matches_selected_role')

    def test_low_confidence_current_box_is_withheld(self):
        result = self.run_fixture([{}, {}, {}, {'lead_confidence': .19}])
        self.assertIsNone(result['decisions'][3]['lead_id'])
        self.assertEqual(result['decisions'][3]['chase_id'], 10002)

    def test_same_livery_candidates_are_ambiguous(self):
        result = self.run_fixture([{}, {}, {}, {'duplicate_lead': True}])
        self.assertIsNone(result['decisions'][3]['lead_id'])
        self.assertEqual(result['decisions'][3]['lead_reason'], 'multiple_similar_candidates')

    def test_matching_templates_never_update_from_automatic_matches(self):
        result = self.run_fixture()
        self.assertEqual(result['summary']['exemplar_counts'], {'lead': 3, 'chase': 3})
        self.assertIn('never update', result['summary']['exemplar_update_policy'])

    def test_old_role_is_withheld_when_same_track_changes_livery(self):
        result = self.run_fixture([{}, {}, {}, {}, {}, {}, {'lead_livery': 'chase'}])
        self.assertEqual(result['decisions'][5]['lead_id'], 10001)
        self.assertIsNone(result['decisions'][6]['lead_id'])

    def test_default_withholds_short_lived_role_hypotheses(self):
        result = self.run_fixture([{}, {}, {}, {'lead_id': 19}, {}, {}, {}], settings=IdentitySettings())
        self.assertIsNone(result['decisions'][3]['lead_id'])
        self.assertEqual(result['decisions'][3]['lead_reason'], 'insufficient_temporal_support')
        self.assertEqual(result['decisions'][4]['lead_id'], 10001)

    def test_enclosing_boxes_cannot_be_reported_as_observed_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            args = list(fixture(directory, [{}, {}, {}, {}]))
            features = {}
            for row in args[2]:
                role = 'lead' if row['track_id'] in (1, 10001) else 'chase'
                features[(row['frame_index'], row['track_id'])] = describe_crop(make_car(role))
                if row['frame_index'] == 3 and role == 'lead':
                    row.update(x1=0, y1=0, x2=320, y2=140)
            with patch('driftlens.role_identity._read_features', return_value=features):
                result = suggest_role_intervals(*args)
            self.assertIsNone(result['decisions'][3]['lead_id'])
            self.assertIsNone(result['decisions'][3]['chase_id'])
            self.assertEqual(result['decisions'][3]['lead_reason'], 'overlapping_or_merged_pair_boxes')

    def test_similar_seed_pair_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'too similar'):
            self.run_fixture([{'chase_livery': 'lead'} for _ in range(5)])

    def test_less_than_three_clear_pair_seed_samples_rejected(self):
        with self.assertRaisesRegex(ValueError, 'at least three'):
            self.run_fixture([{}, {'omit_chase': True}, {}, {}])

    def test_overlapping_seed_boxes_do_not_enter_templates(self):
        def change(args):
            for row in args[2]:
                if row['frame_index'] < 3 and row['track_id'] == 2:
                    row.update(x1=20, y1=30, x2=110, y2=100)
        with self.assertRaisesRegex(ValueError, 'at least three'):
            self.run_fixture(change=change)

    def test_duplicate_track_on_one_frame_rejected(self):
        def change(args):
            args[2].append(copy.deepcopy(args[2][0]))
        with self.assertRaisesRegex(ValueError, 'unique current track'):
            self.run_fixture(change=change)

    def test_disagreeing_observation_times_rejected(self):
        def change(args):
            args[2][0]['clip_time'] = .2
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            self.run_fixture(change=change)

    def test_bad_source_is_reported(self):
        def change(args):
            args[0] = Path(args[0]).with_name('missing.avi')
        with self.assertRaisesRegex(ValueError, 'cannot be opened'):
            self.run_fixture(change=change)

    def test_nonfinite_coordinates_rejected(self):
        def change(args):
            args[2][0]['x1'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'must be finite'):
            self.run_fixture(change=change)

    def test_input_observations_are_never_modified(self):
        with tempfile.TemporaryDirectory() as directory:
            args = fixture(directory)
            before = copy.deepcopy(args[2])
            suggest_role_intervals(*args)
            self.assertEqual(args[2], before)

    def test_interval_compression_splits_same_ids_at_camera_cut(self):
        rows = [{'clip_time': time, 'shot_index': shot, 'lead_id': 1, 'chase_id': 2, 'assignment_kind': 'automatic_appearance'}
                for time, shot in [(0, 0), (.1, 0), (.2, 1), (.3, 1)]]
        intervals = compress_decisions(rows, .4)
        self.assertEqual(len(intervals), 2)
        self.assertEqual(intervals[0]['end_clip_seconds'], .2)
        self.assertEqual(intervals[1]['start_clip_seconds'], .2)

    def test_similarity_is_view_flip_invariant_and_not_probability(self):
        left = describe_crop(make_car('lead'))
        flipped = describe_crop(cv2.flip(make_car('lead'), 1))
        self.assertAlmostEqual(appearance_similarity(left, flipped), 1)
        self.assertLess(appearance_similarity(left, describe_crop(make_car('chase'))), .96)

    def test_invalid_settings_rejected(self):
        with self.assertRaises(ValueError):
            IdentitySettings(min_similarity=float('nan'))

    def test_automatic_seed_is_never_reported_as_user_review(self):
        with tempfile.TemporaryDirectory() as directory:
            args = fixture(directory)
            result = suggest_role_intervals(*args, seed_origin='automatic_motion')
            self.assertEqual(result['decisions'][0]['assignment_kind'], 'automatic_seed')
            self.assertEqual(result['summary']['seed_origin'], 'automatic_motion')
            self.assertIn('unverified hypothesis', result['intervals'][0]['review_basis'])
            self.assertFalse(any(row['assignment_kind'] == 'user_seed' for row in result['decisions']))

    def test_invalid_seed_origin_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'Seed origin'):
                suggest_role_intervals(*fixture(directory), seed_origin='expert_reviewed')

    def test_hue_evidence_is_downweighted_when_neutral_crop_has_few_coloured_pixels(self):
        plain = np.full((70, 90, 3), 230, np.uint8)
        score = appearance_similarity(describe_crop(plain), describe_crop(plain))
        self.assertAlmostEqual(score, 1)

    def consensus_fixture(self, directory, count=8):
        args = fixture(directory, [{} for _ in range(count)])
        def feature(values):
            masses = np.zeros(15)
            for index, mass in values.items():
                masses[index] = mass
            chromatic = float(masses[:12].sum())
            return {'masses': masses, 'hues': masses[:12] / chromatic,
                    'chromatic_fraction': chromatic}
        seed = feature({6: .3, 11: .2, 12: .4, 13: .1})
        anchor = feature({6: .3, 11: .2, 12: .2, 13: .2, 14: .1})
        weak = feature({6: .25, 11: .2, 12: .1, 13: .25, 14: .2})
        chase = feature({1: .3, 12: .1, 13: .1, 14: .5})
        features = {}
        for row in args[2]:
            index = row['frame_index']
            identity = row['track_id']
            features[(index, identity)] = (chase if identity in (2, 10002) else
                                           seed if index < 3 else anchor if index < 6 else weak)
        return args, features

    def test_multiple_strong_hits_can_support_a_weaker_current_crop(self):
        with tempfile.TemporaryDirectory() as directory:
            args, features = self.consensus_fixture(directory)
            with patch('driftlens.role_identity._read_features', return_value=features):
                result = suggest_role_intervals(*args)
            decision = result['decisions'][6]
            self.assertEqual(decision['lead_id'], 10001)
            self.assertLess(decision['lead_similarity'], .9)
            self.assertGreater(decision['lead_consensus_similarity'], .93)
            self.assertEqual(decision['lead_reason'], 'appearance_and_frozen_local_track_consensus')

    def test_consensus_cannot_recursively_extend_its_own_support_bank(self):
        with tempfile.TemporaryDirectory() as directory:
            args, features = self.consensus_fixture(directory, 32)
            with patch('driftlens.role_identity._read_features', return_value=features):
                result = suggest_role_intervals(*args)
            self.assertEqual(result['decisions'][6]['lead_id'], 10001)
            self.assertEqual(result['decisions'][20]['lead_id'], 10001)
            self.assertIsNone(result['decisions'][21]['lead_id'])
            self.assertIsNone(result['decisions'][-1]['lead_id'])


if __name__ == '__main__':
    unittest.main()
