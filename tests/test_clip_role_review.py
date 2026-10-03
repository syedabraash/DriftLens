"""Interval role evidence, missing samples and transactional export safety."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from driftlens.clip_role_review import update_clip_role_intervals
from driftlens.review import read_rows


def interval(start, end, lead, chase):
    return {"start_clip_seconds": start, "end_clip_seconds": end,
            "lead_id": lead, "chase_id": chase, "review_basis": "Visible livery and travel order."}


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fixture(root: Path, *, with_cut=False):
    summary = {"clip_id": "uploaded_clip", "status": "complete", "duration_seconds": .6,
               "source_path": str(root / "source.mp4"), "sampled_fps": 10,
               "frame_count": 6, "tracker": "botsort", "track_ids": [1, 2, 3],
               "lead_id": None, "chase_id": None, "files": {"video": "annotated.mp4"}}
    frames, observations = [], []
    for index in range(6):
        shot = 1 if with_cut and index >= 3 else 0
        frame = {"frame_index": index, "clip_time": round(index / 10, 6), "source_time": 100 + index / 10, "shot_index": shot}
        frames.append(frame)
        identities = (10001, 10002) if shot else (1, 2 if index < 3 else 3)
        for identity in identities:
            if index == 1 and identity == 2:
                continue  # A missing observed chase sample must remain missing.
            observations.append({**frame, "track_id": identity, "x1": 10 if identity in (1, 10001) else 40,
                                 "y1": 10, "x2": 30 if identity in (1, 10001) else 60,
                                 "y2": 30, "confidence": .8, "observed": True})
    if with_cut:
        summary["track_ids"] = [1, 2, 10001, 10002]
        summary["shot_count"] = 2
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    write_csv(root / "frames.csv", frames, list(frames[0]))
    write_csv(root / "observations.csv", observations, list(observations[0]))
    detections = [{key: value for key, value in row.items() if key != "track_id"} for row in observations]
    write_csv(root / "detections.csv", detections, list(detections[0]))
    (root / "frame_metrics.csv").write_bytes(b"prior metrics")
    (root / "annotated.mp4").write_bytes(b"prior video")
    return summary


def renderer(destination, summary, observations, frames):
    (destination / "annotated.mp4").write_bytes(b"revised observed-only replay")


class ClipIntervalTests(unittest.TestCase):
    def test_repeated_first_view_ids_error_names_current_interval_ids_and_matching_action(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root, with_cut=True)
            originals = {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}
            with patch("driftlens.pipeline.render_run") as render:
                with self.assertRaises(ValueError) as raised:
                    update_clip_role_intervals(root, [interval(.3, .6, 1, 2)])
                self.assertIn("Observed IDs in this interval: 10001, 10002", str(raised.exception))
                self.assertIn("Follow lead and chase across views", str(raised.exception))
                render.assert_not_called()
            self.assertEqual(originals, {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()})

    def test_fragmented_track_intervals_preserve_missing_samples_and_rebuild_all_exports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = fixture(root)
            evidence = {name: (root / name).read_bytes() for name in ("frames.csv", "observations.csv", "detections.csv")}
            with patch("driftlens.pipeline.render_run", side_effect=renderer) as render:
                revised = update_clip_role_intervals(root, [interval(0, .3, 1, 2), interval(.4, .6, 1, 3)])
            self.assertEqual(revised["source_path"], original["source_path"])
            self.assertEqual(revised["role_review_status"], "user_assignment_unverified")
            self.assertEqual(revised["paired_frames"], 4)
            metrics = read_rows(root / "frame_metrics.csv")
            self.assertEqual(metrics[1]["chase_observed"], "False")
            self.assertEqual(metrics[1]["separation_proxy"], "")
            self.assertEqual(metrics[3]["lead_track_id"], "")  # Uncovered time is unknown.
            self.assertEqual(metrics[4]["chase_track_id"], "3")
            report = json.loads((root / "analysis.json").read_text())
            self.assertEqual(report["accepted_paired_samples"], 4)
            self.assertEqual(report["measurement_status_counts"]["role_unknown"], 1)
            self.assertEqual(report["run_id"], "uploaded_clip")
            self.assertIn("source", (root / "report.txt").read_text())
            render.assert_called_once()
            for name, content in evidence.items():
                self.assertEqual((root / name).read_bytes(), content)
            self.assertEqual(list(root.glob(".clip_roles_*")), [])

    def test_camera_cut_offsets_require_independently_bounded_roles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root, with_cut=True)
            with patch("driftlens.pipeline.render_run", side_effect=renderer):
                update_clip_role_intervals(root, [interval(0, .3, 1, 2), interval(.3, .6, 10001, 10002)])
            metrics = read_rows(root / "frame_metrics.csv")
            self.assertEqual(metrics[2]["lead_track_id"], "1")
            self.assertEqual(metrics[3]["lead_track_id"], "10001")
            self.assertEqual(metrics[3]["shot_index"], "1")
            with patch("driftlens.pipeline.render_run") as render:
                with self.assertRaisesRegex(ValueError, "camera cut"):
                    update_clip_role_intervals(root, [interval(0, .6, 1, 2)])
                render.assert_not_called()

    def test_bad_bounds_duplicate_roles_or_ids_absent_inside_interval_do_not_mutate_files(self):
        invalid = [[interval(0, .2, 1, 1)], [interval(-.1, .2, 1, 2)],
                   [interval(0, .6001, 1, 2)], [interval(0, .3, 1, 2), interval(.2, .5, 1, 3)],
                   [interval(.4, .6, 1, 2)], [interval(0, .3, 1, 3)],
                   [interval(0, .1, 1, 999)], [interval(0, .2, 1, 2.5)],
                   [interval(float("nan"), .2, 1, 2)], [interval(0, .2, True, 2)]]
        for mapping in invalid:
            with self.subTest(mapping=mapping), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture(root)
                originals = {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}
                with patch("driftlens.pipeline.render_run") as render:
                    with self.assertRaises(ValueError):
                        update_clip_role_intervals(root, mapping)
                    render.assert_not_called()
                self.assertEqual(originals, {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()})
                self.assertEqual(list(root.glob(".clip_roles_*")), [])

    def test_id_at_excluded_end_or_unobserved_box_does_not_validate_role(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            observations = read_rows(root / "observations.csv")
            for row in observations:
                if row["track_id"] == "2":
                    row["observed"] = "False"
            write_csv(root / "observations.csv", observations, list(observations[0]))
            with patch("driftlens.pipeline.render_run") as render:
                with self.assertRaisesRegex(ValueError, "observed"):
                    update_clip_role_intervals(root, [interval(0, .3, 1, 2)])
                with self.assertRaisesRegex(ValueError, "within interval"):
                    update_clip_role_intervals(root, [interval(0, .3, 1, 3)])
                render.assert_not_called()

    def test_empty_map_preserves_unknown_roles_even_when_cars_are_observed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            with patch("driftlens.pipeline.render_run", side_effect=renderer):
                revised = update_clip_role_intervals(root, [])
            self.assertEqual(revised["role_intervals"], [])
            self.assertEqual(revised["paired_frames"], 0)
            self.assertIsNone(revised["lead_id"])
            self.assertTrue(all(row["separation_proxy"] == "" for row in read_rows(root / "frame_metrics.csv")))

    def test_encoding_failure_leaves_all_original_exports_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            originals = {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}
            with patch("driftlens.pipeline.render_run", side_effect=RuntimeError("encoder failed")):
                with self.assertRaisesRegex(RuntimeError, "encoder failed"):
                    update_clip_role_intervals(root, [interval(0, .3, 1, 2)])
            self.assertEqual(originals, {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()})
            self.assertEqual(list(root.glob(".clip_roles_*")), [])

    def test_partial_publication_restores_video_metrics_reports_and_removes_new_exports(self):
        for existing_reports in (False, True):
            with self.subTest(existing_reports=existing_reports), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture(root)
                if existing_reports:
                    (root / "analysis.json").write_bytes(b"prior analysis")
                    (root / "report.txt").write_bytes(b"prior readable report")
                originals = {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}
                real_replace = Path.replace

                def fail_summary(source, target):
                    if source.name == "summary.json" and source.parent.name.startswith(".clip_roles_"):
                        self.assertEqual((root / "annotated.mp4").read_bytes(), b"revised observed-only replay")
                        raise PermissionError("summary locked during publication")
                    return real_replace(source, target)

                with patch("driftlens.pipeline.render_run", side_effect=renderer), patch.object(Path, "replace", new=fail_summary):
                    with self.assertRaisesRegex(PermissionError, "publication"):
                        update_clip_role_intervals(root, [interval(0, .3, 1, 2)])
                self.assertEqual(originals, {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()})
                self.assertEqual(list(root.glob(".clip_roles_*")), [])

    def test_missing_detector_export_stays_unknown_in_rebuilt_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            (root / "detections.csv").unlink()
            with patch("driftlens.pipeline.render_run", side_effect=renderer):
                update_clip_role_intervals(root, [interval(0, .3, 1, 2)])
            report = json.loads((root / "analysis.json").read_text())
            self.assertTrue(all(row["detector_candidate_count"] is None for row in report["frame_statuses"]))
            self.assertNotIn("no_detector_candidates", report["observation_status_counts"])

    def test_incomplete_frame_export_is_rejected_before_rendering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            frames = read_rows(root / "frames.csv")[:-1]
            write_csv(root / "frames.csv", frames, list(frames[0]))
            with patch("driftlens.pipeline.render_run") as render:
                with self.assertRaisesRegex(ValueError, "frame count"):
                    update_clip_role_intervals(root, [interval(0, .3, 1, 2)])
                render.assert_not_called()

    def test_ui_validates_existing_invalid_intervals_without_publishing(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = fixture(root)
            summary["role_intervals"] = [interval(0, .3, 1, 1)]
            script = ("import json\nfrom pathlib import Path\nfrom driftlens.review import read_rows\n"
                      "from driftlens.clip_role_review import render_clip_role_editor\n"
                      f"root=Path({str(root)!r})\n"
                      f"render_clip_role_editor(root,{summary!r},read_rows(root/'observations.csv'))\n")
            app = AppTest.from_string(script, default_timeout=20).run()
            self.assertEqual(len(app.exception), 0)
            with patch("driftlens.pipeline.render_run") as render:
                app.button[0].click().run()
                self.assertEqual(len(app.exception), 0)
                self.assertIn("Lead and chase must be different", app.error[0].value)
                render.assert_not_called()


if __name__ == "__main__":
    unittest.main()
