"""Automatic workflow provenance, uncertainty and legacy camera repair."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from driftlens.automatic_roles import automatic_clip_roles, _saved_seed
from driftlens.role_continuity import follow_clip_roles
from driftlens.review import read_rows, derive_frame_metrics, write_metrics
from test_clip_role_review import fixture, interval, renderer
from test_role_continuity import suggested


class AutomaticRoleWorkflowTests(unittest.TestCase):
    def test_manual_seed_correction_takes_precedence_over_inactive_automatic_history(self):
        frames=[{'frame_index':index,'clip_time':index/10,'shot_index':0} for index in range(4)]
        observations=[{'frame_index':index,'track_id':identity,'observed':True} for index in range(4) for identity in (1,2)]
        summary={'duration_seconds':.4,'role_review_status':'user_assignment_unverified',
                 'role_intervals':[interval(0,.4,2,1)],
                 'role_continuation':{'active':False,'seed_origin':'automatic_motion',
                                      'seed_intervals':[interval(0,.4,1,2)]}}
        seed,origin=_saved_seed(summary,frames,observations)
        self.assertEqual((seed['lead_id'],seed['chase_id'],origin),(2,1,'user_reviewed'))

    def test_saved_seed_is_limited_to_current_camera_view(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            fixture(root,with_cut=True)
            frames,observations=read_rows(root/'frames.csv'),read_rows(root/'observations.csv')
            summary={'duration_seconds':.6,'role_review_status':'user_assignment_unverified',
                     'role_intervals':[interval(0,.6,1,2)]}
            # The fixture has only two jointly visible rows in this short view;
            # three current samples are required instead of crossing the cut.
            self.assertEqual(_saved_seed(summary,frames,observations),(None,None))
            observations.append({**observations[0],'frame_index':'1','track_id':'2'})
            seed,origin=_saved_seed(summary,frames,observations)
            self.assertEqual(seed['end_clip_seconds'],.3)
            self.assertEqual((seed['lead_id'],seed['chase_id'],origin),(1,2,'user_reviewed'))

    def test_automatic_seed_is_not_attributed_to_a_user(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            fixture(root,with_cut=True)
            proposal={'status':'seed_ready','seed':interval(0,.3,1,2),'pair_ids':[1,2],
                      'reason':'Observed motion proposal.','evidence':{'order_consistency':.9}}
            before=json.loads((root/'summary.json').read_text())
            before['automatic_tandem']={'status':'pair_only','reason':'Previous order was unclear.'}
            (root/'summary.json').write_text(json.dumps(before))
            value=suggested(root)
            value['summary']['seed_origin']='automatic_motion'
            with patch('driftlens.role_identity.suggest_role_intervals',return_value=value) as matcher, patch('driftlens.pipeline.render_run',side_effect=renderer):
                revised=follow_clip_roles(root,proposal['seed'],seed_origin='automatic_motion',automatic_proposal=proposal)
            self.assertEqual(matcher.call_args.kwargs['seed_origin'],'automatic_motion')
            self.assertEqual(revised['role_continuation']['automatic_seed_proposal'],proposal)
            self.assertEqual(revised['automatic_tandem'],proposal)
            self.assertIn('automatically proposed',revised['role_assignment_method'])
            report=json.loads((root/'analysis.json').read_text())
            self.assertIn('automatically proposed',' '.join(report['conclusions']))
            self.assertNotIn('reviewed by the user',' '.join(report['conclusions']))

    def test_ambiguous_order_records_proposal_without_making_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            fixture(root,with_cut=True)
            summary=json.loads((root/'summary.json').read_text())
            summary.update(visibility_revision=4,lead_id=None,chase_id=None,role_intervals=[],role_review_status='unassigned')
            (root/'summary.json').write_text(json.dumps(summary))
            write_metrics(root/'frame_metrics.csv',derive_frame_metrics(read_rows(root/'observations.csv'),read_rows(root/'frames.csv'),None,None,role_intervals=[]))
            proposal={'status':'pair_only','seed':None,'pair_ids':[1,2],
                      'reason':'Camera motion leaves travel order unclear.','evidence':{}}
            video=(root/'annotated.mp4').read_bytes()
            with patch('driftlens.auto_seed.propose_automatic_seed',return_value=proposal), patch('driftlens.role_continuity.follow_clip_roles') as follow:
                outcome=automatic_clip_roles(root)
            follow.assert_not_called()
            self.assertEqual(outcome['status'],'pair_only')
            self.assertEqual(outcome['summary']['automatic_tandem'],proposal)
            self.assertEqual((root/'annotated.mp4').read_bytes(),video)
            self.assertEqual(outcome['summary']['role_intervals'],[])

    def test_old_result_is_repaired_in_a_sibling_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as directory:
            original=Path(directory)/'original'
            original.mkdir()
            fixture(original,with_cut=True)
            summary=json.loads((original/'summary.json').read_text())
            summary.update(visibility_revision=3,role_review_status='unassigned')
            (original/'summary.json').write_text(json.dumps(summary))
            (original/'candidates.csv').write_text('saved current candidates')
            frozen={p.name:p.read_bytes() for p in original.iterdir() if p.is_file()}
            proposal={'status':'unavailable','seed':None,'pair_ids':[],
                      'reason':'No stable tandem pair.','evidence':{}}
            def repair(source,destination):
                self.assertEqual(source,original.resolve())
                destination.mkdir()
                fixture(destination,with_cut=True)
                result=json.loads((destination/'summary.json').read_text())
                result.update(visibility_revision=4,role_review_status='unassigned')
                (destination/'summary.json').write_text(json.dumps(result))
                write_metrics(destination/'frame_metrics.csv',derive_frame_metrics(read_rows(destination/'observations.csv'),read_rows(destination/'frames.csv'),None,None,role_intervals=[]))
                return result
            with patch('driftlens.retrack.retrack_cached_run',side_effect=repair), patch('driftlens.auto_seed.propose_automatic_seed',return_value=proposal):
                outcome=automatic_clip_roles(original)
            self.assertEqual(outcome['run_dir'].name,'original_auto')
            self.assertEqual(frozen,{p.name:p.read_bytes() for p in original.iterdir() if p.is_file()})


if __name__=='__main__':
    unittest.main()
