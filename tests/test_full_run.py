"""Full run coverage and camera scoped timeline regression checks."""

import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from driftlens import full_run, pipeline, review
from driftlens.full_run import build_timeline, validate_run_manifest


def manifest():
    return {
        "id": "complete_run",
        "title": "A complete tandem run",
        "source_start_seconds": 10.0,
        "source_end_seconds": 16.0,
        "coverage": "initiation_to_finish",
        "shots": [
            {"id": "shot01", "label": "Initiation", "start_seconds": 10.0, "end_seconds": 12.0},
            {"id": "shot02", "label": "Finish", "start_seconds": 12.0, "end_seconds": 16.0},
        ],
    }


def row(index, time, *, local_shot=0, separation=1.0, lead=True, chase=True, pair=True):
    return {
        "frame_index": index,
        "clip_time": index / 10,
        "source_time": time,
        "shot_index": local_shot,
        "lead_observed": lead,
        "chase_observed": chase,
        "pair_observed": pair,
        "separation_proxy": separation,
        "lead_confidence": 0.9 if lead else None,
        "chase_confidence": 0.8 if chase else None,
    }


class FullRunManifestTests(unittest.TestCase):
    def test_contiguous_shots_cover_exact_run_bounds(self):
        validate_run_manifest(manifest(), source_duration=20.0)
        # A tiny representation difference is permitted, without tolerating a
        # meaningful omitted video interval.
        data = manifest()
        data["shots"][1]["start_seconds"] += 0.0000001
        validate_run_manifest(data, source_duration=20.0)

    def test_gap_and_overlap_are_rejected(self):
        for shift in (0.1, -0.1):
            with self.subTest(shift=shift):
                data = manifest()
                data["shots"][1]["start_seconds"] += shift
                with self.assertRaises(ValueError):
                    validate_run_manifest(data)

    def test_missing_initiation_or_finish_cannot_claim_complete_coverage(self):
        for boundary in ("start", "finish"):
            with self.subTest(boundary=boundary):
                data = manifest()
                if boundary == "start":
                    data["shots"][0]["start_seconds"] = 10.1
                else:
                    data["shots"][-1]["end_seconds"] = 15.9
                with self.assertRaises(ValueError):
                    validate_run_manifest(data)

    def test_empty_duplicate_or_unordered_shots_are_rejected(self):
        changes = [
            lambda data: data.update(shots=[]),
            lambda data: data["shots"][1].update(id="shot01"),
            lambda data: data["shots"].reverse(),
        ]
        for change in changes:
            data = manifest()
            change(data)
            with self.assertRaises(ValueError):
                validate_run_manifest(data)

    def test_full_run_and_individual_shot_duration_limits(self):
        data = manifest()
        data["source_end_seconds"] = 191.0
        data["shots"] = [
            {"id": f"shot{index}", "label": "View", "start_seconds": start, "end_seconds": end}
            for index, (start, end) in enumerate(((10, 60), (60, 110), (110, 160), (160, 191)))
        ]
        with self.assertRaises(ValueError):
            validate_run_manifest(data, source_duration=200)
        data = manifest()
        data["source_end_seconds"] = 71.0
        data["shots"] = [{"id": "shot01", "label": "View", "start_seconds": 10.0, "end_seconds": 71.0}]
        with self.assertRaises(ValueError):
            validate_run_manifest(data, source_duration=100)

    def test_invalid_source_bounds_and_nonfinite_times_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_run_manifest(manifest(), source_duration=15.9)
        for value in (-1.0, float("nan"), float("inf")):
            data = manifest()
            data["source_start_seconds"] = value
            with self.assertRaises(ValueError):
                validate_run_manifest(data)
        data = manifest()
        data["shots"][0]["end_seconds"] = data["shots"][0]["start_seconds"]
        with self.assertRaises(ValueError):
            validate_run_manifest(data)

    def test_coverage_declaration_and_unsafe_identifiers_are_rejected(self):
        data = manifest()
        data["coverage"] = "selected_highlights"
        with self.assertRaises(ValueError):
            validate_run_manifest(data)
        for value in ("../outside", "..\\outside", "a/b"):
            data = manifest()
            data["shots"][0]["id"] = value
            with self.assertRaises(ValueError):
                validate_run_manifest(data)


class FullRunTimelineTests(unittest.TestCase):
    def test_playback_time_accumulates_encoded_frames_at_fractional_cuts(self):
        shots = [
            {"id": "view_a", "start_seconds": 100.0, "end_seconds": 100.2333333, "sampled_fps": 10},
            {"id": "view_b", "start_seconds": 100.2333333, "end_seconds": 100.5, "sampled_fps": 10},
        ]
        inputs = {
            "view_a": [row(0, 100.0), row(1, 100.1), row(2, 100.2)],
            "view_b": [row(0, 100.233333), row(1, 100.333333), row(2, 100.433333)],
        }
        timeline = build_timeline(shots, inputs, source_start=100.0)
        self.assertEqual(len(timeline), 6)
        for entry, expected in zip(timeline, (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)):
            self.assertAlmostEqual(entry["playback_time"], expected)
        self.assertAlmostEqual(timeline[3]["run_time"], 0.233333)
        self.assertAlmostEqual(timeline[3]["playback_time"], 0.3)
        self.assertTrue(timeline[3]["camera_cut"])
        # Encoded frames last 0.1 seconds each, so this replay lasts 0.6 seconds
        # while the source interval lasts 0.5 seconds. Neither clock is replaced
        # by the other, and no fake boundary frame is introduced.
        self.assertAlmostEqual(timeline[-1]["playback_time"] + 0.1, 0.6)

    def test_camera_cut_scopes_repeated_local_ids_and_preserves_measurements(self):
        shots = copy.deepcopy(manifest()["shots"])
        for shot in shots:
            shot.update(lead_id=1, chase_id=2)
        inputs = {"shot01": [row(0, 10, separation=0.5), row(1, 10.1, separation=0.6)], "shot02": [row(0, 12, separation=7.0)]}
        timeline = build_timeline(shots, inputs, source_start=10)
        self.assertEqual(len(timeline), 3)
        self.assertEqual([entry["frame_index"] for entry in timeline], [0, 1, 2])
        self.assertEqual([entry["local_frame_index"] for entry in timeline], [0, 1, 0])
        self.assertEqual([entry["shot_id"] for entry in timeline], ["shot01", "shot01", "shot02"])
        for entry, expected_time in zip(timeline, (0, 0.1, 2)):
            self.assertAlmostEqual(entry["run_time"], expected_time)
        self.assertFalse(timeline[0]["camera_cut"])
        self.assertFalse(timeline[1]["camera_cut"])
        self.assertTrue(timeline[2]["camera_cut"])
        self.assertEqual(timeline[2]["separation_proxy"], 7.0)
        self.assertTrue(timeline[2]["pair_observed"])

    def test_missing_observation_remains_an_unknown_gap(self):
        inputs = {"shot01": [row(0, 10), row(1, 10.1, separation=None, chase=False, pair=False), row(2, 10.2, separation=2)], "shot02": [row(0, 12)]}
        timeline = build_timeline(manifest()["shots"], inputs, source_start=10)
        self.assertEqual(len(timeline), 4)
        self.assertIsNone(timeline[1]["separation_proxy"])
        self.assertFalse(timeline[1]["pair_observed"])
        self.assertEqual(timeline[2]["separation_proxy"], 2)

    def test_csv_false_flags_cannot_create_a_measurement(self):
        inputs = {"shot01": [row(0, 10, separation=3, lead="True", chase="False", pair="True")], "shot02": [row(0, 12, separation="", lead="False", chase="False", pair="False")]}
        timeline = build_timeline(manifest()["shots"], inputs, source_start=10)
        self.assertTrue(timeline[0]["lead_observed"])
        self.assertFalse(timeline[0]["chase_observed"])
        self.assertFalse(timeline[0]["pair_observed"])
        self.assertIsNone(timeline[0]["separation_proxy"])
        self.assertIsNone(timeline[1]["separation_proxy"])

    def test_explicit_unknown_roles_clear_even_stale_measurements(self):
        shots = copy.deepcopy(manifest()["shots"])
        shots[0].update(lead_id=1, chase_id=2)
        shots[1].update(lead_id=None, chase_id=None)
        inputs = {"shot01": [row(0, 10)], "shot02": [row(0, 12, separation=5)]}
        timeline = build_timeline(shots, inputs, source_start=10)
        self.assertEqual(timeline[0]["separation_proxy"], 1)
        self.assertIsNone(timeline[1]["separation_proxy"])
        self.assertFalse(timeline[1]["pair_observed"])

    def test_internal_scene_change_is_marked_even_within_manual_shot(self):
        inputs = {"shot01": [row(0, 10, local_shot=0), row(1, 10.1, local_shot=1)], "shot02": [row(0, 12)]}
        timeline = build_timeline(manifest()["shots"], inputs, source_start=10)
        self.assertEqual(timeline[1]["local_shot_index"], 1)
        self.assertTrue(timeline[1]["camera_cut"])

    def test_duplicate_unordered_or_outside_frame_times_are_rejected(self):
        bad_sequences = [
            [row(0, 10), row(1, 10)],
            [row(0, 10.2), row(1, 10.1)],
            [row(0, 9.9)],
            [row(0, 12.1)],
        ]
        for sequence in bad_sequences:
            with self.subTest(source_times=[entry["source_time"] for entry in sequence]):
                with self.assertRaises(ValueError):
                    build_timeline(manifest()["shots"], {"shot01": sequence, "shot02": [row(0, 12)]}, source_start=10)


class IntervalRoleReviewTests(unittest.TestCase):
    def test_half_open_map_overrides_defaults_and_leaves_uncovered_time_unknown(self):
        intervals = [self.interval(0.0, 0.2, 1, 2), self.interval(0.4, 0.6, 2, 1)]
        summary = {"lead_id": 1, "chase_id": 2, "role_intervals": intervals}
        self.assertEqual(review.roles_at(summary, 0.199999), (1, 2))
        self.assertEqual(review.roles_at(summary, 0.2), (None, None))
        self.assertEqual(review.roles_at(summary, 0.4), (2, 1))
        self.assertEqual(review.roles_at(summary, 0.6), (None, None))
        self.assertEqual(review.roles_at({**summary, "role_intervals": []}, 0.1), (None, None))

    def test_adjacent_role_swap_recomputes_ids_and_breaks_chart_continuity(self):
        intervals = [self.interval(0.0, 0.2, 1, 2), self.interval(0.2, 0.4, 2, 1)]
        frames, observations = self.inputs((0.0, 0.1, 0.2, 0.3))
        metrics = review.derive_frame_metrics(observations, frames, 1, 2, role_intervals=intervals)
        self.assertEqual([entry["lead_track_id"] for entry in metrics], [1, 1, 2, 2])
        self.assertEqual([entry["chase_track_id"] for entry in metrics], [2, 2, 1, 1])
        self.assertAlmostEqual(metrics[1]["lead_confidence"], 0.9)
        self.assertAlmostEqual(metrics[2]["lead_confidence"], 0.6)
        self.assertTrue(all(entry["pair_observed"] for entry in metrics))
        shot = self.shot(intervals)
        timeline = build_timeline([shot], {"interval_shot": metrics}, source_start=10.0)
        self.assertFalse(timeline[1]["role_assignment_cut"])
        self.assertTrue(timeline[2]["role_assignment_cut"])
        self.assertFalse(timeline[2]["camera_cut"])
        self.assertIsNotNone(timeline[2]["separation_proxy"])
        self.assertFalse(timeline[3]["role_assignment_cut"])

    def test_ambiguous_interval_never_reuses_last_roles_despite_visible_cars(self):
        intervals = [self.interval(0.0, 0.2, 1, 2), self.interval(0.4, 0.6, 2, 1)]
        frames, observations = self.inputs((0.0, 0.1, 0.2, 0.3, 0.4, 0.5))
        metrics = review.derive_frame_metrics(observations, frames, 1, 2, role_intervals=intervals)
        timeline = build_timeline([self.shot(intervals)], {"interval_shot": metrics}, source_start=10.0)
        for index in (2, 3):
            self.assertIsNone(timeline[index]["lead_track_id"])
            self.assertIsNone(timeline[index]["chase_track_id"])
            self.assertFalse(timeline[index]["pair_observed"])
            self.assertIsNone(timeline[index]["separation_proxy"])
        self.assertTrue(timeline[2]["role_assignment_cut"])
        self.assertTrue(timeline[4]["role_assignment_cut"])
        self.assertTrue(timeline[4]["pair_observed"])

    def test_single_known_role_keeps_its_observation_without_separation(self):
        intervals = [self.interval(0.0, 0.2, None, 2)]
        frames, observations = self.inputs((0.0, 0.1))
        metrics = review.derive_frame_metrics(observations, frames, None, None, role_intervals=intervals)
        shot = self.shot(intervals)
        shot.update(lead_id=None, chase_id=None)
        timeline = build_timeline([shot], {"interval_shot": metrics}, source_start=10.0)
        self.assertFalse(timeline[0]["lead_observed"])
        self.assertTrue(timeline[0]["chase_observed"])
        self.assertEqual(timeline[0]["chase_track_id"], 2)
        self.assertAlmostEqual(timeline[0]["chase_confidence"], 0.6)
        self.assertFalse(timeline[0]["pair_observed"])
        self.assertIsNone(timeline[0]["separation_proxy"])

    def test_interval_validation_rejects_overlap_bad_bounds_and_invalid_roles(self):
        invalid_maps = [
            [self.interval(0.0, 0.3, 1, 2), self.interval(0.2, 0.4, 1, 2)],
            [self.interval(0.2, 0.3, 1, 2), self.interval(0.0, 0.1, 1, 2)],
            [self.interval(-0.1, 0.2, 1, 2)],
            [self.interval(0.0, 0.5, 1, 2)],
            [self.interval(0.2, 0.2, 1, 2)],
            [self.interval(float("nan"), 0.2, 1, 2)],
            [self.interval(0.0, float("inf"), 1, 2)],
            [self.interval(0.0, 0.2, 1, 1)],
            [self.interval(0.0, 0.2, 1, 99)],
            [self.interval(0.0, 0.2, 1, 2.5)],
        ]
        for intervals in invalid_maps:
            with self.subTest(intervals=intervals):
                with self.assertRaises(ValueError):
                    review.validate_role_intervals(intervals, duration=0.4, available_ids={1, 2})
        review.validate_role_intervals([self.interval(0.0, 0.2, 1, 2), self.interval(0.2, 0.4, 2, 1)], duration=0.4, available_ids={1, 2})

    @staticmethod
    def interval(start, end, lead, chase):
        return {"start_clip_seconds": start, "end_clip_seconds": end, "lead_id": lead, "chase_id": chase}

    @staticmethod
    def shot(intervals):
        return {"id": "interval_shot", "start_seconds": 10.0, "end_seconds": 10.6, "sampled_fps": 10, "lead_id": 1, "chase_id": 2, "role_intervals": intervals}

    @staticmethod
    def inputs(times):
        frames, observations = [], []
        for index, clip_time in enumerate(times):
            frame = {"frame_index": index, "clip_time": clip_time, "source_time": 10 + clip_time, "shot_index": 0}
            frames.append(frame)
            for identity, x, confidence in ((1, 10, 0.9), (2, 50, 0.6)):
                observations.append({**frame, "track_id": identity, "x1": x, "y1": 10, "x2": x + 20, "y2": 30, "confidence": confidence, "observed": True})
        return frames, observations


class FullRunExportSafetyTests(unittest.TestCase):
    def test_digest_failure_marks_processing_failed_before_inference(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1], prefix=".full_run_test_") as temporary:
            root = Path(temporary)
            data, manifest_path, run = self.prepare_complete_fixture(root)
            data["shots"][0]["role_intervals"] = [IntervalRoleReviewTests.interval(0.0, 0.1, 1, 2)]
            manifest_path.write_text(json.dumps(data), encoding="utf-8")
            with patch.object(full_run, "ROOT", root), patch.object(pipeline, "video_info", return_value={"duration_seconds": 20}), patch.object(full_run.hashlib, "file_digest", side_effect=OSError("Could not read input digest")), patch.object(pipeline, "analyze_video") as inference:
                with self.assertRaisesRegex(OSError, "input digest"):
                    full_run.process_full_run(manifest_path)
                inference.assert_not_called()
            summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "failed")
            self.assertIn("input digest", summary["error"])

    def test_assembly_rejects_mixed_processing_profiles_before_encoding(self):
        for key, value in (("tracker", "bytetrack"), ("imgsz", 416), ("sampled_fps", 5), ("agnostic_nms", False)):
            with self.subTest(changed_setting=key):
                with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1], prefix=".full_run_test_") as temporary:
                    root = Path(temporary)
                    data, _, run = self.prepare_complete_fixture(root)
                    child = run / "shots" / "shot02" / "summary.json"
                    inconsistent = json.loads(child.read_text(encoding="utf-8"))
                    inconsistent[key] = value
                    child.write_text(json.dumps(inconsistent), encoding="utf-8")
                    original = (run / "annotated.mp4").read_bytes()
                    with patch.object(full_run, "ROOT", root), patch.object(full_run.subprocess, "run") as encoder:
                        with self.assertRaisesRegex(ValueError, "inconsistent processing profiles"):
                            full_run.assemble_full_run(data, run)
                        encoder.assert_not_called()
                    self.assertEqual((run / "annotated.mp4").read_bytes(), original)

    def test_partial_assembly_publication_restores_already_replaced_outputs(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1], prefix=".full_run_test_") as temporary:
            root = Path(temporary)
            data, _, run = self.prepare_complete_fixture(root)
            original = {name: (run / name).read_bytes() for name in ("summary.json", "timeline.csv", "shots.json", "annotated.mp4")}
            original_replace = Path.replace

            def encoder_stub(command, **kwargs):
                Path(command[-1]).write_bytes(b"new combined replay")
                return SimpleNamespace(returncode=0, stderr=b"")

            def locked_timeline(source, target):
                if source.name == "timeline.csv" and source.parent.name.startswith(".publish_"):
                    self.assertEqual((run / "annotated.mp4").read_bytes(), b"new combined replay")
                    raise PermissionError("Timeline export is locked")
                return original_replace(source, target)

            with patch.object(full_run, "ROOT", root), patch("imageio_ffmpeg.get_ffmpeg_exe", return_value="ffmpeg"), patch.object(full_run.subprocess, "run", side_effect=encoder_stub), patch.object(Path, "replace", new=locked_timeline):
                with self.assertRaisesRegex(PermissionError, "locked"):
                    full_run.assemble_full_run(data, run)
            for name, contents in original.items():
                self.assertEqual((run / name).read_bytes(), contents, name)
            self.assertEqual(list(run.glob(".publish_*")), [])

    def test_changed_visibility_revision_does_not_reuse_historical_role_ids(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1], prefix=".full_run_test_") as temporary:
            root = Path(temporary)
            data, manifest_path, run = self.prepare_complete_fixture(root)
            data["source_sha256"] = hashlib.sha256((root / "source.mp4").read_bytes()).hexdigest()
            data["model_sha256"] = hashlib.sha256((root / "models/yolov8n.pt").read_bytes()).hexdigest()
            data["role_review_profile"] = {}
            for shot in data["shots"]:
                shot["role_intervals"] = [{"start_clip_seconds": 0, "end_clip_seconds": .1, "lead_id": 1, "chase_id": 2}]
                saved = run / "shots" / shot["id"] / "summary.json"
                summary = json.loads(saved.read_text(encoding="utf-8"))
                summary.update(visibility_revision=3, lead_id=None, chase_id=None, role_assignment_method=None, role_review_status="unassigned")
                saved.write_text(json.dumps(summary), encoding="utf-8")
            manifest_path.write_text(json.dumps(data), encoding="utf-8")
            with patch.object(full_run, "ROOT", root), patch.object(pipeline, "video_info", return_value={"duration_seconds": 20}), patch.object(pipeline, "analyze_video", side_effect=AssertionError("Unexpected fresh inference")), patch.object(pipeline, "render_run"), patch.object(full_run, "apply_reviewed_intervals") as apply_roles, patch.object(full_run, "assemble_full_run", return_value={"status": "complete"}):
                full_run.process_full_run(manifest_path)
            apply_roles.assert_not_called()
            for shot in data["shots"]:
                saved = json.loads((run / "shots" / shot["id"] / "summary.json").read_text(encoding="utf-8"))
                self.assertIsNone(saved["lead_id"])
                self.assertIsNone(saved["chase_id"])
                self.assertNotIn("role_intervals", saved)

    def test_changed_model_digest_forces_new_inference_instead_of_cached_shot(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1], prefix=".full_run_test_") as temporary:
            root = Path(temporary)
            _, manifest_path, run = self.prepare_complete_fixture(root)
            child_path = run / "shots" / "shot01" / "summary.json"
            cached = json.loads(child_path.read_text(encoding="utf-8"))
            cached["model_sha256"] = "previous_model_digest"
            child_path.write_text(json.dumps(cached), encoding="utf-8")
            current_digest = hashlib.sha256((root / "models" / "yolov8n.pt").read_bytes()).hexdigest()

            def fresh_inference(source, destination, *args, **kwargs):
                fresh = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
                fresh["model_sha256"] = current_digest
                return fresh

            with patch.object(full_run, "ROOT", root), patch.object(pipeline, "video_info", return_value={"duration_seconds": 20}), patch.object(pipeline, "analyze_video", side_effect=fresh_inference) as inference, patch.object(pipeline, "render_run"), patch.object(full_run, "assemble_full_run", return_value={"status": "complete"}):
                full_run.process_full_run(manifest_path)
            inference.assert_called_once()
            self.assertEqual(Path(inference.call_args.args[1]), run / "shots" / "shot01")

    def test_failed_combined_export_restores_shot_and_parent_outputs(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1], prefix=".full_run_test_") as temporary:
            root = Path(temporary)
            data, manifest_path, run = self.prepare_complete_fixture(root)
            child = run / "shots" / "shot01"
            all_paths = [child / name for name in ("summary.json", "frame_metrics.csv", "annotated.mp4")]
            all_paths += [run / name for name in ("summary.json", "timeline.csv", "shots.json", "annotated.mp4")]
            original = {path: path.read_bytes() for path in all_paths}

            def render_stub(destination, *arguments):
                (destination / "annotated.mp4").write_bytes(b"revised shot replay")

            def fail_during_combined_publication(*arguments):
                revised = json.loads((child / "summary.json").read_text(encoding="utf-8"))
                self.assertEqual((revised["lead_id"], revised["chase_id"]), (2, 1))
                self.assertEqual(revised["role_review_status"], "user_assignment_unverified")
                self.assertNotEqual((child / "frame_metrics.csv").read_bytes(), original[child / "frame_metrics.csv"])
                for name in ("summary.json", "timeline.csv", "shots.json", "annotated.mp4"):
                    (run / name).write_bytes(b"incomplete combined publication")
                raise RuntimeError("Combined replay publication failed")

            with patch.object(full_run, "ROOT", root), patch.object(pipeline, "render_run", side_effect=render_stub), patch.object(full_run, "assemble_full_run", side_effect=fail_during_combined_publication):
                with self.assertRaisesRegex(RuntimeError, "publication failed"):
                    full_run.update_shot_roles(manifest_path, run, "shot01", 2, 1)
            for path, contents in original.items():
                self.assertEqual(path.read_bytes(), contents, str(path.relative_to(root)))
            self.assertEqual(list(run.glob(".role_backup_*")), [])

    def test_assembly_rejects_shot_from_another_source_before_encoding(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1], prefix=".full_run_test_") as temporary:
            root = Path(temporary)
            data, _, run = self.prepare_complete_fixture(root)
            child_summary = run / "shots" / "shot02" / "summary.json"
            wrong = json.loads(child_summary.read_text(encoding="utf-8"))
            wrong["source_path"] = str(root / "different_video.mp4")
            child_summary.write_text(json.dumps(wrong), encoding="utf-8")
            original_summary = (run / "summary.json").read_bytes()
            with patch.object(full_run, "ROOT", root), patch.object(full_run.subprocess, "run") as encoder:
                with self.assertRaisesRegex(ValueError, "different source"):
                    full_run.assemble_full_run(data, run)
                encoder.assert_not_called()
            self.assertEqual((run / "summary.json").read_bytes(), original_summary)

    def test_assembly_rejects_incomplete_metrics_before_encoding(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1], prefix=".full_run_test_") as temporary:
            root = Path(temporary)
            data, _, run = self.prepare_complete_fixture(root)
            child_summary = run / "shots" / "shot01" / "summary.json"
            inconsistent = json.loads(child_summary.read_text(encoding="utf-8"))
            inconsistent["frame_count"] = 2
            child_summary.write_text(json.dumps(inconsistent), encoding="utf-8")
            original_summary = (run / "summary.json").read_bytes()
            with patch.object(full_run, "ROOT", root), patch.object(full_run.subprocess, "run") as encoder:
                with self.assertRaisesRegex(ValueError, "frame count"):
                    full_run.assemble_full_run(data, run)
                encoder.assert_not_called()
            self.assertEqual((run / "summary.json").read_bytes(), original_summary)

    @staticmethod
    def prepare_complete_fixture(root):
        data = manifest()
        data["source"] = {"path": "source.mp4"}
        data["source_end_seconds"] = 10.2
        data["shots"][0]["end_seconds"] = 10.1
        data["shots"][1].update(start_seconds=10.1, end_seconds=10.2)
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(data), encoding="utf-8")
        (root / "source.mp4").write_bytes(b"unused local source placeholder")
        (root / "models").mkdir()
        (root / "models" / "yolov8n.pt").write_bytes(b"unused detector weights placeholder")
        model_digest = hashlib.sha256((root / "models" / "yolov8n.pt").read_bytes()).hexdigest()
        signature = {"size_bytes": (root / "source.mp4").stat().st_size, "modified_ns": (root / "source.mp4").stat().st_mtime_ns}
        run = root / "outputs" / "full_runs" / data["id"]
        for shot in data["shots"]:
            child = run / "shots" / shot["id"]
            child.mkdir(parents=True)
            frame = {"frame_index": 0, "clip_time": 0.0, "source_time": shot["start_seconds"], "shot_index": 0}
            observations = [{"clip_id": shot["id"], **frame, "track_id": identity, "x1": x, "y1": 10, "x2": x + 20, "y2": 30, "confidence": confidence, "observed": True} for identity, x, confidence in ((1, 10, 0.9), (2, 50, 0.6))]
            for name, fields, rows in (("frames.csv", pipeline.FRAME_FIELDS, [frame]), ("observations.csv", pipeline.OBSERVATION_FIELDS, observations)):
                with (child / name).open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
            review.write_metrics(child / "frame_metrics.csv", review.derive_frame_metrics(observations, [frame], 1, 2))
            summary = {"status": "complete", "pipeline_revision": 2, "source_path": str(root / "source.mp4"), "source_signature": signature, "model": "yolov8n.pt", "model_sha256": model_digest, "start_seconds": shot["start_seconds"], "end_seconds": shot["end_seconds"], "duration_seconds": shot["end_seconds"] - shot["start_seconds"], "sampled_fps": 10, "frame_count": 1, "track_ids": [1, 2], "shot_count": 1, "lead_id": 1, "chase_id": 2, "paired_frames": 1, "pair_coverage": 1.0, "tracker": "botsort", "imgsz": 640, "agnostic_nms": True, "confidence_threshold": 0.15, "processing_seconds": 1.0, "role_assignment_method": "Previous assistant assignment"}
            (child / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            (child / "annotated.mp4").write_bytes(b"original shot replay")
        (run / "summary.json").write_text(json.dumps({"status": "complete", "run_id": data["id"]}), encoding="utf-8")
        (run / "timeline.csv").write_bytes(b"original full run timeline")
        (run / "shots.json").write_text(json.dumps(data["shots"]), encoding="utf-8")
        (run / "annotated.mp4").write_bytes(b"original combined replay")
        return data, manifest_path, run


if __name__ == "__main__":
    unittest.main()
