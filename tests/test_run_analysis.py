"""Evidence attribution, camera boundaries and report publication checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from driftlens.run_analysis import build_run_analysis, load_evidence, playback_seek, report_payloads


def sample(index, *, shot="a", playback=None, source=None, lead=1, chase=2,
           lead_observed=True, chase_observed=True, pair=True, separation=1.0,
           local_frame=None, local_shot=0, camera_cut=False):
    return {"frame_index": index, "local_frame_index": index if local_frame is None else local_frame,
            "playback_time": index / 10 if playback is None else playback,
            "source_time": 100 + index / 10 if source is None else source,
            "shot_id": shot, "local_shot_index": local_shot, "camera_cut": camera_cut,
            "lead_track_id": lead, "chase_track_id": chase,
            "lead_observed": lead_observed, "chase_observed": chase_observed,
            "pair_observed": pair, "separation_proxy": separation}


def metadata():
    return {"run_id": "test_run", "title": "Test review", "sampled_fps": 10,
            "shots": [{"shot_id": "a", "label": "First camera", "start_seconds": 100,
                       "end_seconds": 101, "run_start_seconds": 0, "run_end_seconds": 1}]}


class RunAnalysisTests(unittest.TestCase):
    def test_pair_requires_both_observed_distinct_assigned_roles_and_valid_geometry(self):
        rows = [sample(0), sample(1, lead=None), sample(2, lead=2),
                sample(3, lead_observed=False), sample(4, chase_observed=False),
                sample(5, pair=False), sample(6, separation=float("nan")),
                sample(7, separation=-1), sample(8, separation=0)]
        report = build_run_analysis(metadata(), rows)
        self.assertEqual(report["accepted_paired_samples"], 2)
        self.assertEqual(report["assigned_pair_samples"], 7)
        self.assertEqual(report["measurement_status_counts"]["role_unknown"], 2)
        for row in report["frame_statuses"][1:8]:
            self.assertIsNone(row["separation_proxy"])
        self.assertEqual(report["views"][0]["separation"]["minimum"], 0)
        self.assertNotIn("separation", report)  # No cross-view aggregate.
        json.loads(report_payloads(report)["analysis.json"])

    def test_unknown_roles_do_not_imply_detector_failure_even_with_visible_tracks(self):
        rows = [sample(0, lead=None, chase=None)]
        observation = {"frame_index": 0, "track_id": 10, "observed": True}
        evidence = {"a": {"detections": [{"frame_index": 0}], "observations": [observation]}}
        report = build_run_analysis(metadata(), rows, evidence_by_shot=evidence)
        self.assertEqual(report["frame_statuses"][0]["measurement_status"], "role_unknown")
        self.assertEqual(report["frame_statuses"][0]["observation_status"], "tracked_observations_present")
        self.assertEqual(report["frame_statuses"][0]["detector_candidate_count"], 1)
        self.assertIn("do not establish detector failure", " ".join(report["conclusions"]))
        self.assertNotIn("no_detector_candidates", report["observation_status_counts"])

    def test_detector_and_tracker_absence_require_separate_saved_evidence(self):
        rows = [sample(i, lead_observed=False, chase_observed=False) for i in range(3)]
        evidence = {"a": {"detections": [{"frame_index": 1}, {"frame_index": 1}], "observations": []}}
        report = build_run_analysis(metadata(), rows, evidence_by_shot=evidence)
        self.assertEqual([row["observation_status"] for row in report["frame_statuses"]],
                         ["no_detector_candidates", "detections_without_track_ids", "no_detector_candidates"])
        missing_detector = build_run_analysis(metadata(), rows, evidence_by_shot={"a": {"observations": []}})
        self.assertEqual(missing_detector["observation_status_counts"], {"no_tracked_observations_detector_unknown": 3})
        no_evidence = build_run_analysis(metadata(), rows)
        self.assertEqual(no_evidence["observation_status_counts"], {"evidence_unavailable": 3})

    def test_camera_scoped_evidence_does_not_join_repeated_local_frame_numbers(self):
        summary = metadata()
        summary["shots"].append({"shot_id": "b", "start_seconds": 101, "end_seconds": 102})
        rows = [sample(0), sample(1, shot="b", local_frame=0, camera_cut=True)]
        evidence = {"a": {"detections": [], "observations": []},
                    "b": {"detections": [{"frame_index": 0}], "observations": [{"frame_index": 0, "track_id": 1}]}}
        report = build_run_analysis(summary, rows, evidence_by_shot=evidence)
        self.assertEqual(report["frame_statuses"][0]["detector_candidate_count"], 0)
        self.assertEqual(report["frame_statuses"][1]["detector_candidate_count"], 1)
        self.assertEqual(len(report["views"]), 2)

    def test_unknown_spans_stop_at_cuts_role_changes_and_unsampled_gaps(self):
        rows = [sample(0, pair=False), sample(1, pair=False),
                sample(2, pair=False, local_shot=1, camera_cut=True),
                sample(3, pair=False, local_shot=1, lead=None),
                sample(4, pair=False, local_shot=1, lead=None, playback=.8)]
        report = build_run_analysis(metadata(), rows)
        self.assertEqual([span["sample_count"] for span in report["unknown_intervals"]], [2, 1, 1, 1])
        gap = report["longest_unknown_interval"]
        self.assertEqual(gap["frame_index"], 0)
        self.assertAlmostEqual(gap["sampled_duration_seconds"], .2)
        self.assertEqual(gap["last_sample_source_seconds"], 100.1)
        self.assertEqual(len(report["views"]), 2)

    def test_source_and_replay_end_times_remain_distinct(self):
        summary = metadata()
        summary["shots"][0].update(end_seconds=100.15, run_end_seconds=.2)
        report = build_run_analysis(summary, [sample(0, pair=False), sample(1, pair=False)])
        gap = report["longest_unknown_interval"]
        self.assertEqual(gap["end_playback_seconds"], .2)
        self.assertEqual(gap["end_source_seconds"], 100.15)
        self.assertEqual(gap["sample_count"], 2)

    def test_extreme_events_stay_inside_each_view_and_preserve_exact_timestamps(self):
        rows = [sample(0, separation=1), sample(1, separation=2),
                sample(2, local_shot=1, separation=100, camera_cut=True)]
        report = build_run_analysis(metadata(), rows)
        self.assertEqual(report["views"][0]["separation"]["maximum"], 2)
        self.assertEqual(report["views"][1]["separation"]["maximum"], 100)
        events = [event for event in report["events"] if event["kind"] == "minimum_separation"]
        self.assertEqual([event["frame_index"] for event in events], [0, 2])
        self.assertEqual(events[1]["playback_time"], .2)
        self.assertEqual(events[1]["source_time"], 100.2)

    def test_clip_metrics_use_half_open_interval_roles_and_keep_unknown_outside_map(self):
        summary = {"clip_id": "clip", "sampled_fps": 10, "start_seconds": 100, "end_seconds": 100.4,
                   "lead_id": 1, "chase_id": 2,
                   "role_intervals": [{"start_clip_seconds": 0, "end_clip_seconds": .2, "lead_id": 1, "chase_id": 2}]}
        rows = []
        for index in range(4):
            row = sample(index)
            for name in ("shot_id", "playback_time", "local_frame_index", "lead_track_id", "chase_track_id", "local_shot_index"):
                row.pop(name)
            row.update(clip_time=index / 10, shot_index=0)
            rows.append(row)
        report = build_run_analysis(summary, rows)
        self.assertEqual(report["accepted_paired_samples"], 2)
        self.assertEqual(report["frame_statuses"][2]["lead_status"], "role_unknown")
        self.assertEqual(report["unknown_intervals"][0]["start_playback_seconds"], .2)

    def test_invalid_timeline_is_rejected_instead_of_silent_reordering(self):
        for rows in ([sample(0), sample(1, playback=0)], [sample(1), sample(0)],
                     [sample(0, source=float("inf"))], [sample(0, playback=-1)]):
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    build_run_analysis(metadata(), rows)

    def test_native_seek_fallback_retains_requested_fractional_timestamp(self):
        self.assertEqual(playback_seek(8.266667), {"target_seconds": 8.266667, "player_start_seconds": 8,
                         "preroll_seconds": .266667, "exact_video_seek": False})
        self.assertTrue(playback_seek(4)["exact_video_seek"])
        for value in (-1, None, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                playback_seek(value)

    def test_empty_and_partial_evidence_files_are_different(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "detections.csv").write_text("frame_index,observed\n", encoding="utf-8")
            evidence = load_evidence(root, root)[root.name]
            self.assertEqual(evidence["detections"], [])
            self.assertIsNone(evidence["observations"])
            (root / "observations.csv").write_text("invalid_header\n", encoding="utf-8")
            self.assertIsNone(load_evidence(root, root)[root.name]["observations"])
            with self.assertRaises(ValueError):
                load_evidence(root, root, [{"shot_id": "../../../escape"}])

    def test_report_is_deterministic_without_mutating_evidence_or_inputs(self):
        summary, rows = metadata(), [sample(0), sample(1, pair=False)]
        original = copy.deepcopy((summary, rows))
        first = report_payloads(build_run_analysis(summary, rows))
        self.assertEqual(first, report_payloads(build_run_analysis(summary, rows)))
        self.assertEqual(original, (summary, rows))
        self.assertIn(b"source", first["report.txt"])

    def test_recovered_box_counts_preserve_original_class_name_and_score_meaning(self):
        summary = metadata()
        summary.update(recover_vehicle_classes=True, source_class_names={"2": "car", "7": "truck", "39": "bottle"})
        native = {"frame_index": 0, "track_id": 1, "class": 2, "source_class_id": 2,
                  "detection_method": "native_detector", "confidence": .8}
        recovered = {"frame_index": 0, "track_id": 2, "class": 2, "source_class_id": 39,
                     "detection_method": "temporal_appearance_class_recovery", "appearance_similarity": .92,
                     "recovery_anchor_track_id": 2, "confidence": .95}
        evidence = {"a": {"detections": [native, recovered], "observations": [native, recovered]}}
        original = copy.deepcopy(evidence)
        report = build_run_analysis(summary, [sample(0)], evidence_by_shot=evidence)
        provenance = report["box_provenance"]
        self.assertEqual(provenance["native_vehicle_prediction_boxes"], 1)
        self.assertEqual(provenance["recovered_selected_boxes"], 1)
        self.assertEqual(provenance["recovered_tracked_boxes"], 1)
        self.assertEqual(provenance["recovered_source_classes"], [{"source_class_id": 39, "source_class_name": "bottle", "selected_boxes": 1, "tracked_boxes": 1}])
        self.assertEqual(report["accepted_paired_samples_with_recovery"], 1)
        self.assertTrue(report["frame_statuses"][0]["pair_uses_recovered_box"])
        self.assertIn("bottle (class 39)", " ".join(report["conclusions"]))
        self.assertIn("not car confidence", report["recovery_note"])
        self.assertEqual(evidence, original)
        self.assertIn(b"original predicted class score", report_payloads(report)["report.txt"])

    def test_archived_missing_classes_remain_unspecified_and_do_not_become_recovery(self):
        archived = {"frame_index": 0, "track_id": 1, "confidence": .9}
        report = build_run_analysis(metadata(), [sample(0)], evidence_by_shot={"a": {"detections": [archived], "observations": [archived]}})
        self.assertEqual(report["box_provenance"]["unspecified_selected_box_provenance"], 1)
        self.assertEqual(report["box_provenance"]["unspecified_tracked_box_provenance"], 1)
        self.assertEqual(report["box_provenance"]["recovered_selected_boxes"], 0)
        self.assertNotIn("recovery_note", report)
        absent = build_run_analysis(metadata(), [sample(0)])
        self.assertEqual(absent["box_provenance"]["detector_exports_available"], 0)
        self.assertIsNone(absent["frame_statuses"][0]["recovered_selected_box_count"])

    def test_recovered_background_candidate_never_counts_as_recovery_in_an_accepted_pair(self):
        observations = [{"frame_index": 0, "track_id": 1, "class": 2, "source_class_id": 2, "detection_method": "native_detector"},
                        {"frame_index": 0, "track_id": 2, "class": 2, "source_class_id": 2, "detection_method": "native_detector"},
                        {"frame_index": 0, "track_id": 3, "class": 2, "source_class_id": 39, "detection_method": "temporal_appearance_class_recovery"}]
        report = build_run_analysis(metadata(), [sample(0)], evidence_by_shot={"a": {"detections": observations, "observations": observations}})
        self.assertEqual(report["box_provenance"]["recovered_tracked_boxes"], 1)
        self.assertEqual(report["accepted_paired_samples_with_recovery"], 0)
        self.assertFalse(report["frame_statuses"][0]["pair_uses_recovered_box"])


class ReportPublicationTests(unittest.TestCase):
    def test_interval_form_renders_and_rejects_duplicate_ids_before_rebuilding(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            child = root / "shots" / "a"
            child.mkdir(parents=True)
            (root / "data").mkdir()
            (root / "data" / "full_run_catalog.json").write_text("{}", encoding="utf-8")
            saved = {"duration_seconds": 1, "track_ids": [1, 2], "lead_id": 1, "chase_id": 1,
                     "role_intervals": [{"start_clip_seconds": 0, "end_clip_seconds": .5, "lead_id": 1, "chase_id": 1}]}
            original = json.dumps(saved).encode()
            (child / "summary.json").write_bytes(original)
            script = ("from pathlib import Path\nfrom driftlens.full_run_ui import _role_editor\n"
                      f"root = Path({str(root)!r})\n"
                      "_role_editor(root, root, {'run_id':'test'}, {'a': {'label':'Camera A'}}, None)\n")
            app = AppTest.from_string(script, default_timeout=20)
            with patch("driftlens.full_run.update_shot_roles") as rebuild:
                app.run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(app.radio[0].value, "Bounded intervals")
                app.button[0].click().run()
                self.assertEqual(len(app.exception), 0)
                self.assertIn("Lead and chase must be different", app.error[0].value)
                rebuild.assert_not_called()
            self.assertEqual((child / "summary.json").read_bytes(), original)

    def test_report_publication_rolls_back_partial_updates_and_removes_new_files(self):
        from driftlens.run_analysis_ui import save_shot_analysis
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original_summary = b'{"clip_id": "clip", "status": "complete"}'
            (root / "summary.json").write_bytes(original_summary)
            (root / "report.txt").write_bytes(b"prior report")
            real_replace = Path.replace
            report = build_run_analysis(metadata(), [sample(0)])

            def fail_summary(source, target):
                if source.name == "summary.json" and source.parent.name.startswith(".analysis_"):
                    raise PermissionError("summary locked")
                return real_replace(source, target)

            with patch("driftlens.run_analysis_ui._shot_report", return_value=report), patch.object(Path, "replace", new=fail_summary):
                with self.assertRaisesRegex(PermissionError, "locked"):
                    save_shot_analysis(root, {"clip_id": "clip", "status": "complete"})
            self.assertEqual((root / "summary.json").read_bytes(), original_summary)
            self.assertEqual((root / "report.txt").read_bytes(), b"prior report")
            self.assertFalse((root / "analysis.json").exists())
            self.assertEqual(list(root.glob(".analysis_*")), [])


if __name__ == "__main__":
    unittest.main()
