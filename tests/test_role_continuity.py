"""Role continuation publication, evidence alignment and manual precedence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from driftlens.clip_role_review import update_clip_role_intervals
from driftlens.role_continuity import follow_clip_roles
from driftlens.review import read_rows
from test_clip_role_review import fixture, interval, renderer


def suggested(root):
    frames = read_rows(root / 'frames.csv')
    decisions = [{**frame, 'lead_id': 1 if index < 3 else 10001,
                  'chase_id': 2 if index < 3 else 10002} for index, frame in enumerate(frames)]
    return {'intervals': [interval(0, .3, 1, 2), interval(.3, .6, 10001, 10002)],
            'decisions': decisions, 'summary': {'settings': {'min_similarity': .7}}}


class RoleContinuationTests(unittest.TestCase):
    def test_cross_view_roles_publish_with_uncertain_provenance_and_observed_only_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root, with_cut=True)
            evidence = {name: (root / name).read_bytes() for name in ('frames.csv', 'observations.csv', 'detections.csv')}
            with patch('driftlens.role_identity.suggest_role_intervals', return_value=suggested(root)), patch('driftlens.pipeline.render_run', side_effect=renderer):
                revised = follow_clip_roles(root, interval(0, .3, 1, 2))
            self.assertEqual(revised['role_review_status'], 'automatic_appearance_unverified')
            self.assertTrue(revised['role_continuation']['active'])
            self.assertEqual(revised['paired_frames'], 5)
            self.assertEqual(read_rows(root / 'frame_metrics.csv')[1]['separation_proxy'], '')
            self.assertEqual(len(json.loads((root / 'role_matches.json').read_text())['frames']), 6)
            report = json.loads((root / 'analysis.json').read_text())
            self.assertIn('unverified appearance matches', ' '.join(report['conclusions']))
            self.assertIn('not calibrated identity probabilities', ' '.join(report['limitations']))
            for name, contents in evidence.items():
                self.assertEqual((root / name).read_bytes(), contents)

    def test_invalid_seed_stops_before_matching_or_publishing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root, with_cut=True)
            before = {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}
            with patch('driftlens.role_identity.suggest_role_intervals') as match:
                with self.assertRaises(ValueError):
                    follow_clip_roles(root, interval(0, .3, 1, 1))
                match.assert_not_called()
            self.assertEqual(before, {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()})

    def test_match_evidence_cannot_disagree_with_published_roles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root, with_cut=True)
            value = suggested(root)
            value['decisions'][3]['lead_id'] = 10002
            with patch('driftlens.role_identity.suggest_role_intervals', return_value=value), patch('driftlens.pipeline.render_run') as render:
                with self.assertRaisesRegex(ValueError, 'interval map'):
                    follow_clip_roles(root, interval(0, .3, 1, 2))
                render.assert_not_called()
            self.assertEqual(list(root.glob('.clip_roles_*')), [])

    def test_locked_match_export_restores_every_published_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root, with_cut=True)
            before = {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}
            original_replace = Path.replace

            def lock_match_export(source, target):
                if source.name == 'role_matches.json' and source.parent.name.startswith('.clip_roles_'):
                    raise PermissionError('Match evidence export is locked')
                return original_replace(source, target)

            with patch('driftlens.role_identity.suggest_role_intervals', return_value=suggested(root)), patch('driftlens.pipeline.render_run', side_effect=renderer), patch.object(Path, 'replace', new=lock_match_export):
                with self.assertRaisesRegex(PermissionError, 'locked'):
                    follow_clip_roles(root, interval(0, .3, 1, 2))
            self.assertEqual(before, {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()})
            self.assertEqual(list(root.glob('.clip_roles_*')), [])

    def test_manual_corrections_override_and_deactivate_automatic_assignments(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root, with_cut=True)
            with patch('driftlens.role_identity.suggest_role_intervals', return_value=suggested(root)), patch('driftlens.pipeline.render_run', side_effect=renderer):
                follow_clip_roles(root, interval(0, .3, 1, 2))
                revised = update_clip_role_intervals(root, [interval(0, .3, 1, 2)])
            self.assertFalse(revised['role_continuation']['active'])
            self.assertEqual(revised['role_review_status'], 'user_assignment_unverified')
            self.assertEqual(revised['paired_frames'], 2)
            self.assertTrue((root / 'role_matches.json').is_file())
            report = json.loads((root / 'analysis.json').read_text())
            self.assertNotIn('role_continuation', report)


if __name__ == '__main__':
    unittest.main()
