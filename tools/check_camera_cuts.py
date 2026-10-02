"""Read-only adjacent-frame camera cut diagnostic; exports numbers, not footage.

Use expected boundary times from prior source inspection. Results are local
regression diagnostics, not a generalisation benchmark or expert annotations.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from driftlens.camera_cuts import CAMERA_CUT_SETTINGS, CameraCutDetector


def scan(source: Path, start: float, end: float, target_fps: float, expected_cuts: list[float]) -> tuple[dict, list[dict]]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError("Could not open diagnostic source video.")
    fps = capture.get(cv2.CAP_PROP_FPS)
    total = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or start < 0 or end <= start or end > total / fps + .05:
        capture.release()
        raise ValueError("Choose an interval within the diagnostic source video.")
    first, last = round(start * fps), min(total, round(end * fps))
    stride = max(1, round(fps / target_fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    detector = CameraCutDetector()
    previous = None
    rows = []
    started = time.perf_counter()
    sample_count = 0
    try:
        for index in range(first, last):
            success, frame = capture.read()
            if not success:
                raise RuntimeError(f"Diagnostic source decoding stopped at frame {index}.")
            if (index-first) % stride:
                continue
            sample_count += 1
            cut = detector.update(frame)
            small = cv2.cvtColor(cv2.resize(frame, (96,54)),cv2.COLOR_BGR2GRAY)
            if detector.last_metrics is not None:
                histogram_first = cv2.calcHist([previous], [0], None, [32], [0,256])
                histogram_second = cv2.calcHist([small], [0], None, [32], [0,256])
                old_difference = float(np.mean(cv2.absdiff(previous, small))) / 255
                old_correlation = float(cv2.compareHist(histogram_first, histogram_second,cv2.HISTCMP_CORREL))
                clip_time = (index-first) / fps
                expected = any(abs(clip_time-boundary) <= .5 * stride/fps + 1e-8 for boundary in expected_cuts)
                rows.append({
                    "clip_time": round(clip_time,6), "source_time": round(index/fps,6),
                    "expected_cut": expected, "old_is_cut": old_difference > .20 and old_correlation < .65,
                    "old_pixel_difference": old_difference, "old_histogram_correlation": old_correlation,
                    **detector.last_metrics,
                })
            previous = small
    finally:
        capture.release()
    detected = [row["clip_time"] for row in rows if row["is_cut"]]
    unexpected = [row["clip_time"] for row in rows if row["is_cut"] and not row["expected_cut"]]
    missing = [boundary for boundary in expected_cuts if not any(abs(boundary-detected_time) <= .5 * stride/fps + 1e-8 for detected_time in detected)]
    negatives = sorted((row for row in rows if not row["expected_cut"]),key=lambda row:row["structure_residual"],reverse=True)[:3]
    report = {
        "source_filename": source.name, "start_seconds": start, "end_seconds": end,
        "source_fps": fps, "sampled_fps": fps / stride, "sample_count": sample_count,
        "adjacent_comparisons": len(rows), "scan_seconds": round(time.perf_counter()-started,3),
        "settings": CAMERA_CUT_SETTINGS, "expected_cut_times": expected_cuts,
        "detected_cut_times": detected, "unexpected_cut_times": unexpected, "missing_expected_cut_times": missing,
        "old_detected_cut_times": [row["clip_time"] for row in rows if row["old_is_cut"]],
        "boundary_metrics": [row for row in rows if row["expected_cut"]],
        "largest_structure_change_negatives": negatives,
        "evidence_scope": "Same-event source inspection regression only. Boundary expectations are assistant visual decisions without human expert review. New upload was used to tune this safeguard; these counts do not establish independent accuracy on new events, arbitrary videos, low sample rates, flashes, or whip pans.",
    }
    return report, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    parser.add_argument("--start",type=float,default=0)
    parser.add_argument("--end",type=float,required=True)
    parser.add_argument("--fps",type=float,default=10)
    parser.add_argument("--expected-cut",type=float,action="append",default=[])
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--csv",type=Path)
    args=parser.parse_args()
    cv2.setNumThreads(2)
    report,rows=scan(args.source,args.start,args.end,args.fps,args.expected_cut)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    if args.csv:
        args.csv.parent.mkdir(parents=True,exist_ok=True)
        with args.csv.open("w",newline="",encoding="utf-8") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({key:report[key] for key in ("sample_count","detected_cut_times","unexpected_cut_times","missing_expected_cut_times","old_detected_cut_times","scan_seconds")},indent=2))


if __name__ == "__main__":
    main()
