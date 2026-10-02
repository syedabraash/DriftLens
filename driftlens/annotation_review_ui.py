"""Explicit human source-frame review without altering frozen references."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from .annotation_review import (confirm_sample, export_reviewed_labels, load_review_queue,
                                save_review_queue)


def render_annotation_review(root: Path, catalog: dict) -> None:
    path = root / "outputs/annotation_review/queue.json"
    with st.expander("Review reference labels for future training"):
        st.caption("These are assistant drafted boxes awaiting your review. A saved review records your own attestation; it does not certify expert ground truth. Original test references stay frozen.")
        if not path.is_file():
            st.info("The review queue has not been prepared yet.")
            return
        try:
            queue = load_review_queue(path)
        except (ValueError, OSError) as error:
            st.error(f"Could not load the review queue: {error}")
            return
        records = queue["records"]
        if not records:
            st.info("There are no source frames in this review queue yet.")
            return
        reviewed = sum(record.get("review") is not None for record in records)
        st.write(f"{reviewed} of {len(records)} source frames have a saved human review.")
        options = {record["sample_id"]: record for record in records}
        identifier = st.selectbox("Reference frame", list(options),
                                  format_func=lambda key: f"{options[key]['clip_id']} · {options[key]['time_seconds']:.2f}s · {options[key]['split']} · {'Reviewed' if options[key]['review'] else 'Pending'}")
        record = options[identifier]
        clips = {clip["id"]: clip for clip in catalog.get("clips", [])}
        image_path = (root / record.get("image", "")).resolve()
        try:
            image_path.relative_to(root)
        except ValueError:
            st.error("The reference image path is outside this project.")
            return
        frame = cv2.imread(str(image_path)) if image_path.is_file() else None
        if frame is None:
            try:
                source = (root / catalog["source"]["path"]).resolve()
                source.relative_to(root)
                clip = clips[record["clip_id"]]
            except (ValueError, KeyError) as error:
                st.error(f"The source context for this frame is unavailable: {error}")
                return
            capture = cv2.VideoCapture(str(source))
            try:
                seconds = float(clip["start_seconds"]) + float(record["time_seconds"])
                capture.set(cv2.CAP_PROP_POS_FRAMES, round(seconds * capture.get(cv2.CAP_PROP_FPS)))
                success, frame = capture.read()
                if not success:
                    frame = None
            finally:
                capture.release()
        if frame is None:
            st.error("The original source frame could not be loaded. Review cannot be confirmed.")
            return
        if frame.shape[:2] != (record["height"], record["width"]):
            st.error("The displayed image dimensions differ from the source label coordinates. Review cannot be confirmed.")
            return
        preview = frame.copy()
        for car in record["cars"]:
            if car.get("bbox"):
                x1, y1, x2, y2 = map(int, car["bbox"])
                cv2.rectangle(preview, (x1, y1), (x2, y2), (96, 255, 215), 2)
                cv2.putText(preview, car["identity"], (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, .6, (96, 255, 215), 2)
        original_column, draft_column = st.columns(2)
        original_column.image(frame, channels="BGR", caption="Original source frame", width="stretch")
        draft_column.image(preview, channels="BGR", caption="Current reference boxes", width="stretch")
        st.caption("Coordinates use the original image pixels. Keep each identity consistent within this camera shot. Hidden cars have no bounding box.")
        table = []
        for car in record["cars"]:
            coordinates = car.get("bbox") or [None] * 4
            table.append({"identity": car["identity"], "class_id": car["class_id"], "visibility": car["visibility"],
                          **dict(zip(("x1", "y1", "x2", "y2"), coordinates))})
        with st.form(f"annotation_review_{identifier}"):
            edited = st.data_editor(pd.DataFrame(table, columns=["identity", "class_id", "visibility", "x1", "y1", "x2", "y2"]),
                                    num_rows="dynamic", hide_index=True, width="stretch",
                                    column_config={"visibility": st.column_config.SelectboxColumn(options=["visible", "partial", "hidden"]),
                                                   "class_id": st.column_config.SelectboxColumn(options=[2, 7])})
            left, right = st.columns(2)
            smoke = left.selectbox("Smoke", ["unknown", "clear", "light", "heavy"], index=["unknown", "clear", "light", "heavy"].index(record["tags"]["smoke"]))
            overlap = right.selectbox("Overlap", ["unknown", "none", "partial", "severe"], index=["unknown", "none", "partial", "severe"].index(record["tags"]["overlap"]))
            reviewer = st.text_input("Reviewer name", value=(record.get("review") or {}).get("reviewer", ""))
            image_checked = st.checkbox("I inspected the original frame")
            boxes_checked = st.checkbox("I checked and corrected the visible boxes")
            identities_checked = st.checkbox("I checked the identities within this camera shot")
            submitted = st.form_submit_button("Save my frame review")
        if submitted:
            try:
                cars = []
                for row in edited.to_dict("records"):
                    visibility = row["visibility"]
                    cars.append({"identity": row["identity"], "class_id": int(row["class_id"]), "visibility": visibility,
                                 "bbox": None if visibility == "hidden" else [float(row[key]) for key in ("x1", "y1", "x2", "y2")]})
                revised = confirm_sample(queue, identifier, cars, {"smoke": smoke, "overlap": overlap}, reviewer,
                                         {"image": image_checked, "boxes": boxes_checked, "identities": identities_checked})
                save_review_queue(path, revised)
                st.session_state["notice"] = "Your frame review was saved separately from the original references."
                st.rerun()
            except (ValueError, TypeError, KeyError, OSError) as error:
                st.error(f"Review was not saved: {error}")
        st.download_button("Export the review queue", json.dumps(queue, indent=2), file_name="annotation_review_queue.json", mime="application/json")
        if reviewed:
            exported = export_reviewed_labels(queue)
            st.download_button("Export confirmed labels", json.dumps(exported, indent=2), file_name="human_reviewed_labels.json", mime="application/json")
