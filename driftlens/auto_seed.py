"""Conservative automatic participant and role seed hypotheses from observations.

This is a proposal, not a reviewed identity or camera-invariant role classifier.
Raw image motion never establishes travel order without background compensation.
"""
from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import median

DEFAULTS = {
    "max_initial_seconds": 8.0,
    "seed_seconds": 2.0,
    "min_track_samples": 12,
    "min_track_fraction": 0.35,
    "min_joint_samples": 12,
    "min_joint_fraction": 0.60,
    "min_raw_motion_widths": 0.08,
    "min_pair_dominance_margin": 0.15,
    "min_seed_joint_fraction": 0.90,
    "max_seed_overlap_fraction": 0.08,
    "max_camera_windows": 8,
    "max_decoded_frames": 16,
    "camera_window_seconds": 1.0,
    "camera_step_seconds": 0.5,
    "min_background_inliers": 20,
    "min_background_inlier_fraction": 0.55,
    "min_motion_widths": 0.08,
    "min_motion_cosine": 0.70,
    "min_speed_ratio": 0.20,
    "min_order_separation_widths": 0.07,
    "max_order_overlap_fraction": 0.60,
    "min_order_votes": 2,
    "min_order_agreement": 1.0,
}


def _number(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Nonfinite observation value")
    return result


def _center(row):
    return ((_number(row["x1"]) + _number(row["x2"])) / 2,
            (_number(row["y1"]) + _number(row["y2"])) / 2)


def _width(row):
    return _number(row["x2"]) - _number(row["x1"])


def _overlap(a, b):
    width = max(0.0, min(_number(a["x2"]), _number(b["x2"])) - max(_number(a["x1"]), _number(b["x1"])))
    height = max(0.0, min(_number(a["y2"]), _number(b["y2"])) - max(_number(a["y1"]), _number(b["y1"])))
    areas = [_width(row) * (_number(row["y2"]) - _number(row["y1"])) for row in (a, b)]
    return width * height / max(1.0, min(areas))


def _length(vector):
    return math.hypot(*vector)


def _raw_motion(rows):
    ordered = sorted(rows, key=lambda row: _number(row["clip_time"]))
    edge = max(1, len(ordered) // 5)
    first = tuple(median(_center(row)[axis] for row in ordered[:edge]) for axis in (0, 1))
    last = tuple(median(_center(row)[axis] for row in ordered[-edge:]) for axis in (0, 1))
    return _length((last[0] - first[0], last[1] - first[1])) / max(1.0, median(_width(row) for row in ordered))


class _BackgroundMotion:
    """Decode only selected sampled frames and mask all detected vehicles."""

    def __init__(self, source_path, frames, by_frame, settings):
        import cv2
        self.cv2 = cv2
        self.capture = cv2.VideoCapture(str(source_path))
        self.frames = frames
        self.by_frame = by_frame
        self.settings = settings
        self.cache = {}
        self.decoded_frames = 0

    def close(self):
        self.capture.release()

    def _image(self, frame):
        import numpy as np
        key = int(frame["frame_index"])
        if key in self.cache:
            return self.cache[key]
        if not self.capture.isOpened() or self.decoded_frames >= self.settings["max_decoded_frames"]:
            return None
        # Seek by the stored original timestamp, never by local tracker ID.
        self.capture.set(self.cv2.CAP_PROP_POS_MSEC, _number(frame.get("source_time", frame["clip_time"])) * 1000)
        okay, bgr = self.capture.read()
        if not okay:
            return None
        self.decoded_frames += 1
        height, width = bgr.shape[:2]
        scale = 192 / max(width, height)
        small_width, small_height = max(16, round(width * scale)), max(16, round(height * scale))
        gray = self.cv2.cvtColor(self.cv2.resize(bgr, (small_width, small_height)), self.cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray)
        # Interior source image; exclude the lower player/control region in captures.
        mask[round(small_height * .05):round(small_height * .80), round(small_width * .05):round(small_width * .95)] = 255
        for row in self.by_frame.get(key, {}).values():
            x1, y1, x2, y2 = [_number(row[name]) for name in ("x1", "y1", "x2", "y2")]
            xa, xb = max(0, int(x1 * small_width / width) - 3), min(small_width, int(x2 * small_width / width) + 4)
            ya, yb = max(0, int(y1 * small_height / height) - 3), min(small_height, int(y2 * small_height / height) + 4)
            mask[ya:yb, xa:xb] = 0
        result = (gray, mask, width, height)
        self.cache[key] = result
        return result

    def estimate(self, first, last):
        import numpy as np
        a, b = self._image(first), self._image(last)
        if a is None or b is None or a[2:] != b[2:]:
            return {"valid": False, "reason": "Source frame decode unavailable"}
        old, mask, width, height = a
        new = b[0]
        points = self.cv2.goodFeaturesToTrack(old, 300, .01, 3, mask=mask)
        if points is None or len(points) < self.settings["min_background_inliers"]:
            return {"valid": False, "reason": "Insufficient unmasked background features"}
        followed, status, _ = self.cv2.calcOpticalFlowPyrLK(old, new, points, None, winSize=(21, 21), maxLevel=3)
        if followed is None or status is None:
            return {"valid": False, "reason": "Background flow unavailable"}
        returned, back_status, _ = self.cv2.calcOpticalFlowPyrLK(new, old, followed, None, winSize=(21, 21), maxLevel=3)
        if returned is None or back_status is None:
            return {"valid": False, "reason": "Reverse background flow unavailable"}
        good = ((status.ravel() > 0) & (back_status.ravel() > 0)
                & (np.linalg.norm(points - returned, axis=2).ravel() < 2.0))
        if int(good.sum()) < self.settings["min_background_inliers"]:
            return {"valid": False, "reason": "Insufficient consistent background flow"}
        affine, inliers = self.cv2.estimateAffinePartial2D(points[good], followed[good], method=self.cv2.RANSAC, ransacReprojThreshold=2)
        if affine is None or inliers is None or not np.isfinite(affine).all():
            return {"valid": False, "reason": "Background affine fit unavailable"}
        count = int(inliers.sum())
        fraction = count / int(good.sum())
        scale = math.hypot(float(affine[0, 0]), float(affine[1, 0]))
        rotation = math.atan2(float(affine[1, 0]), float(affine[0, 0]))
        evidence = {"background_inliers": count, "background_flow_points": int(good.sum()),
                    "background_inlier_fraction": round(fraction, 6), "scale": round(scale, 6),
                    "rotation_radians": round(rotation, 6)}
        if (count < self.settings["min_background_inliers"]
                or fraction < self.settings["min_background_inlier_fraction"]
                or not .8 <= scale <= 1.25 or abs(rotation) > .1):
            return {"valid": False, "reason": "Weak or excessive camera motion fit", **evidence}
        # Convert the fitted mapping from small-image coordinates to source pixels.
        transform = affine.copy()
        scale_x, scale_y = old.shape[1] / width, old.shape[0] / height
        transform[0, 1] *= scale_y / scale_x
        transform[1, 0] *= scale_x / scale_y
        transform[0, 2] /= scale_x
        transform[1, 2] /= scale_y
        return {"valid": True, "affine": transform.tolist(), **evidence}


def _order_vote(first_rows, last_rows, pair, camera, settings):
    """Rank current centres along their compensated shared travel direction."""
    if not camera.get("valid"):
        return {"accepted": False, "reason": camera.get("reason", "Camera motion unavailable")}
    affine = camera["affine"]
    motions = []
    widths = []
    for ident in pair:
        a, b = _center(first_rows[ident]), _center(last_rows[ident])
        mapped = tuple(affine[axis][0] * a[0] + affine[axis][1] * a[1] + affine[axis][2] for axis in (0, 1))
        motions.append((b[0] - mapped[0], b[1] - mapped[1]))
        widths.append((_width(first_rows[ident]) + _width(last_rows[ident])) / 2)
    speeds = [_length(motion) for motion in motions]
    motion_widths = [speeds[i] / max(1.0, widths[i]) for i in (0, 1)]
    if min(motion_widths) < settings["min_motion_widths"]:
        return {"accepted": False, "reason": "Participants stationary relative to background", "motion_widths": motion_widths}
    cosine = sum(motions[0][axis] * motions[1][axis] for axis in (0, 1)) / max(.000001, speeds[0] * speeds[1])
    if cosine < settings["min_motion_cosine"] or min(speeds) / max(speeds) < settings["min_speed_ratio"]:
        return {"accepted": False, "reason": "Participants do not share supported travel direction", "motion_cosine": round(cosine, 6)}
    if max(_overlap(first_rows[pair[0]], first_rows[pair[1]]), _overlap(last_rows[pair[0]], last_rows[pair[1]])) > settings["max_order_overlap_fraction"]:
        return {"accepted": False, "reason": "Overlapping boxes undermine travel order"}
    vector = ((motions[0][0] + motions[1][0]) / 2, (motions[0][1] + motions[1][1]) / 2)
    size = _length(vector)
    direction = (vector[0] / size, vector[1] / size)
    projections = []
    for rows in (first_rows, last_rows):
        a, b = _center(rows[pair[0]]), _center(rows[pair[1]])
        projections.append(((a[0] - b[0]) * direction[0] + (a[1] - b[1]) * direction[1]) / max(1.0, sum(widths) / 2))
    if min(abs(value) for value in projections) < settings["min_order_separation_widths"] or projections[0] * projections[1] <= 0:
        return {"accepted": False, "reason": "Travel order weak or changes within window", "order_separation_widths": projections}
    lead = pair[0] if projections[0] > 0 else pair[1]
    return {"accepted": True, "lead_id": lead, "chase_id": next(ident for ident in pair if ident != lead),
            "motion_cosine": round(cosine, 6), "motion_widths": [round(value, 6) for value in motion_widths],
            "order_separation_widths": [round(value, 6) for value in projections]}


def propose_automatic_seed(source_path, frames, observations, duration_seconds, *, settings=None):
    """Return a bounded automatic seed hypothesis, pair-only suggestion or abstention.

    No role is inferred from an ID number. Weak background compensation, competing
    moving pairs, stationary cars and inconsistent order prevent an automatic seed.
    """
    config = {**DEFAULTS, **(settings or {})}
    evidence = {"method": "dominant_pair_camera_compensated_order_v1", "settings": config,
                "review_status": "automatic_geometry_unverified",
                "score_note": "Gates are heuristic evidence, not calibrated identity probabilities.",
                "scope": "Initial participant and travel-order proposal; perspective, camera motion, turns and similar participants can invalidate it."}
    def result(status, reason, pair=(), seed=None):
        return {"status": status, "seed": seed, "pair_ids": list(pair), "reason": reason, "evidence": evidence}
    try:
        duration = _number(duration_seconds)
        ordered = sorted((dict(row) for row in frames), key=lambda row: (float(row["clip_time"]), int(row["frame_index"])))
        if duration <= 0 or not ordered:
            return result("unavailable", "No valid sampled interval")
        first_shot = int(ordered[0].get("shot_index", 0))
        first_time = _number(ordered[0]["clip_time"])
        cutoff = min(duration, first_time + config["max_initial_seconds"])
        initial = []
        for row in ordered:
            if int(row.get("shot_index", 0)) != first_shot or _number(row["clip_time"]) >= cutoff:
                break
            initial.append(row)
        if len(initial) < config["min_joint_samples"]:
            return result("unavailable", "First camera view has insufficient samples")
        keys = {int(row["frame_index"]) for row in initial}
        by_frame = defaultdict(dict)
        tracks = defaultdict(list)
        for original in observations:
            row = dict(original)
            if str(row.get("observed", True)).lower() not in {"true", "1", "1.0"}:
                continue
            key = int(row["frame_index"])
            if key not in keys or int(row.get("shot_index", first_shot)) != first_shot:
                continue
            ident = int(row["track_id"])
            if _width(row) <= 0 or _number(row["y2"]) <= _number(row["y1"]):
                continue
            if ident in by_frame[key]:
                return result("unavailable", "Duplicate current ID within a sampled frame")
            by_frame[key][ident] = row
            tracks[ident].append(row)
        candidates = [ident for ident, rows in tracks.items() if len(rows) >= config["min_track_samples"]
                      and len(rows) / len(initial) >= config["min_track_fraction"]
                      and _raw_motion(rows) >= config["min_raw_motion_widths"]]
        candidates.sort(key=lambda ident: (-len(tracks[ident]), ident))
        if len(candidates) < 2:
            return result("unavailable", "Fewer than two persistent moving candidate tracks")
        pairs = []
        for pair in combinations(candidates[:6], 2):
            joint = [frame for frame in initial if set(pair).issubset(by_frame.get(int(frame["frame_index"]), {}))]
            score = len(joint) / len(initial)
            if len(joint) >= config["min_joint_samples"] and score >= config["min_joint_fraction"]:
                pairs.append((score, pair, joint))
        pairs.sort(key=lambda item: (-item[0], item[1]))
        if not pairs:
            return result("unavailable", "No sufficiently observed candidate pair")
        score, pair, joint = pairs[0]
        competitor = pairs[1][0] if len(pairs) > 1 else 0.0
        evidence["participant_pair"] = {"joint_samples": len(joint), "initial_samples": len(initial),
                                        "joint_fraction": round(score, 6), "competing_joint_fraction": round(competitor, 6),
                                        "track_samples": {str(ident): len(tracks[ident]) for ident in pair}}
        if score - competitor < config["min_pair_dominance_margin"]:
            return result("unavailable", "Competing persistent moving pairs prevent participant selection")
        step = median([_number(b["clip_time"]) - _number(a["clip_time"]) for a, b in zip(initial, initial[1:])])
        if step <= 0:
            return result("unavailable", "Invalid sample timing")
        first_cut = min(duration, _number(initial[-1]["clip_time"]) + step)
        # Earliest clear seed, wholly bounded inside this camera view.
        seed_range = None
        for start_frame in joint:
            start = _number(start_frame["clip_time"])
            end = min(first_cut, start + config["seed_seconds"])
            inside = [frame for frame in initial if start <= _number(frame["clip_time"]) < end]
            valid = [frame for frame in inside if set(pair).issubset(by_frame.get(int(frame["frame_index"]), {}))
                     and _overlap(by_frame[int(frame["frame_index"])][pair[0]], by_frame[int(frame["frame_index"])][pair[1]]) <= config["max_seed_overlap_fraction"]]
            if len(valid) >= config["min_joint_samples"] and len(valid) / max(1, len(inside)) >= config["min_seed_joint_fraction"]:
                seed_range = (start, end)
                break
        if seed_range is None:
            return result("pair_only", "Dominant pair found but no sufficiently clear seed interval", pair)
        motion = _BackgroundMotion(source_path, initial, by_frame, config)
        votes = []
        try:
            for index in range(config["max_camera_windows"]):
                start = first_time + index * config["camera_step_seconds"]
                end = start + config["camera_window_seconds"]
                if end >= first_cut:
                    break
                a = min(joint, key=lambda row: abs(_number(row["clip_time"]) - start))
                b = min(joint, key=lambda row: abs(_number(row["clip_time"]) - end))
                if max(abs(_number(a["clip_time"]) - start), abs(_number(b["clip_time"]) - end)) > step * 2:
                    continue
                try:
                    camera = motion.estimate(a, b)
                except Exception:
                    camera = {"valid": False, "reason": "Camera motion unavailable for these sampled frames"}
                vote = _order_vote(by_frame[int(a["frame_index"])], by_frame[int(b["frame_index"])], pair, camera, config)
                votes.append({"start_clip_seconds": _number(a["clip_time"]), "end_clip_seconds": _number(b["clip_time"]),
                              "camera": {key:value for key,value in camera.items() if key != "affine"}, **vote})
        finally:
            motion.close()
        evidence["motion_windows"] = votes
        evidence["decoded_frames"] = motion.decoded_frames
        accepted = [vote for vote in votes if vote["accepted"]]
        valid_camera_votes = [vote for vote in votes if vote["camera"].get("valid")]
        if (len(valid_camera_votes) >= config["min_order_votes"]
                and all(vote.get("reason") == "Participants stationary relative to background" for vote in valid_camera_votes)):
            return result("unavailable", "Dominant boxes are not two moving participants after camera compensation")
        if len(accepted) < config["min_order_votes"]:
            return result("pair_only", "Dominant pair found; camera-compensated travel order is insufficient", pair)
        tally = {ident: sum(vote["lead_id"] == ident for vote in accepted) for ident in pair}
        lead = max(tally, key=tally.get)
        if tally[lead] / len(accepted) < config["min_order_agreement"]:
            return result("pair_only", "Camera-compensated travel order conflicts between windows", pair)
        chase = next(ident for ident in pair if ident != lead)
        seed = {"start_clip_seconds": seed_range[0], "end_clip_seconds": seed_range[1],
                "lead_id": lead, "chase_id": chase,
                "review_basis": "Automatic dominant-pair and camera-compensated travel-order hypothesis; no user or expert identity confirmation."}
        evidence["accepted_order_windows"] = len(accepted)
        return result("seed_ready", "Consistent compensated motion supports an unverified initial lead and chase proposal", pair, seed)
    except (ValueError, TypeError, KeyError, OSError) as error:
        return result("unavailable", f"Automatic seed unavailable: {error}")
