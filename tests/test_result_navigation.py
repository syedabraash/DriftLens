"""Exercise the real result widget, deep links and queued result transitions."""
import ast
import os
from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


APP_SOURCE = Path(os.environ.get("DRIFTLENS_NAV_APP_SOURCE", str(Path(__file__).resolve().parents[1]/"app.py")))
source = APP_SOURCE.read_text(encoding="utf-8")
tree = ast.parse(source)
names = {"number", "queue_review_run", "sync_requested_review_run", "selected_review_run_changed", "select_review_run"}
functions = "\n\n".join(ast.get_source_segment(source, node) for node in tree.body
                         if isinstance(node, ast.FunctionDef) and node.name in names)
if {node.name for node in tree.body if isinstance(node, ast.FunctionDef)} & names != names:
    raise RuntimeError("The tested result selector helpers are missing from app.py.")

OLD = str(Path("fixture_results")/"test")
NEW = str(Path("fixture_results")/"test_continuity")
script = ("import math\nfrom pathlib import Path\nfrom typing import Any\nimport streamlit as st\n"
          + functions
          + f"\nruns=[(Path({OLD!r}),{{'clip_id':'test','tracker':'botsort','duration_seconds':37}}),"
            f"(Path({NEW!r}),{{'clip_id':'test_continuity','tracker':'botsort','duration_seconds':37}})]\n"
          + "linked=sync_requested_review_run(runs)\n"
            "selected=select_review_run({str(path):(path,summary) for path,summary in runs})\n"
            "st.text('Linked result: '+str(linked))\n"
            "if st.button('Open repaired result'):\n"
            "    queue_review_run(runs[1][0])\n"
            "    st.rerun()\n")


class ResultNavigationTests(unittest.TestCase):
    def app(self):
        return AppTest.from_string(script, default_timeout=20)

    def assert_selected(self, app, path, query_name):
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.selectbox[0].value, path)
        self.assertEqual(app.query_params.get("run"), [query_name])
        self.assertIn(f"Saved result: {Path(path).name}", [item.value for item in app.caption])

    def test_existing_same_deep_link_repairs_stale_old_selected_widget(self):
        app = self.app()
        app.query_params["run"] = "test_continuity"
        app.session_state["selected_run"] = OLD
        app.session_state["linked_preview_run"] = NEW
        app.run()
        self.assert_selected(app, NEW, "test_continuity")

    def test_dropdown_choice_updates_address_and_survives_future_reruns(self):
        app = self.app()
        app.query_params["run"] = "test_continuity"
        app.run()
        app.selectbox[0].select(OLD).run()
        self.assert_selected(app, OLD, "test")
        app.run()
        self.assert_selected(app, OLD, "test")
        app.query_params["run"] = "test_continuity"
        app.run()
        self.assert_selected(app, NEW, "test_continuity")

    def test_newly_processed_pending_result_takes_precedence_over_previous_address(self):
        app = self.app()
        app.query_params["run"] = "test"
        app.session_state["pending_selected_run"] = NEW
        app.run()
        self.assert_selected(app, NEW, "test_continuity")
        self.assertNotIn("pending_selected_run", app.session_state)

    def test_open_result_after_widget_render_does_not_mutate_instantiated_widget(self):
        app = self.app()
        app.query_params["run"] = "test"
        app.run()
        app.button[0].click().run()
        self.assert_selected(app, NEW, "test_continuity")

    def test_no_link_and_unknown_link_do_not_invent_or_open_arbitrary_results(self):
        app = self.app().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.selectbox[0].value, OLD)
        self.assertNotIn("run", app.query_params)
        self.assertIn("Linked result: None", [item.value for item in app.text])
        app.query_params["run"] = "../../outside"
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.selectbox[0].value, OLD)
        self.assertIn("Linked result: None", [item.value for item in app.text])


if __name__ == "__main__":
    unittest.main()
