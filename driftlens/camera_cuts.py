"""Small causal cut safeguard based on structure and camera motion continuity.

Brightness histograms can stay alike when a broadcast changes cameras. Check
whether source features still follow one affine camera motion instead. This is
a cut heuristic, not a camera identity estimator or cross-camera car matcher.
"""
from __future__ import annotations

import cv2
import numpy as np


CAMERA_CUT_SETTINGS = {
    "method": "brightness_residual_and_interior_motion_continuity_v2",
    "sample_size": [192, 108],
    "min_structure_residual": 0.10,
    "max_motion_inlier_fraction": 0.10,
    "max_corners": 180,
    "min_corner_distance": 5,
    "max_forward_backward_error_pixels": 1.5,
    "ransac_error_pixels": 2.0,
    "optical_flow_pyramid_levels": [3, 1],
    "motion_support_region": [0.05, 0.05, 0.95, 0.80],
}


def _sample(frame: np.ndarray) -> np.ndarray:
    """Work with actual source pixels at a fixed, inexpensive analysis size."""
    if frame.ndim not in (2, 3) or not frame.size:
        raise ValueError("Camera cut detection needs a nonempty source frame.")
    small = cv2.resize(frame, tuple(CAMERA_CUT_SETTINGS["sample_size"]), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small


def compare_samples(previous: np.ndarray, current: np.ndarray) -> dict:
    """Compare two sampled grayscale frames without assuming car visibility.

    Subtract the median brightness shift so a simple exposure change does not
    count as a structural change. Forward/backward checked feature tracks must
    also fit a robust affine transform, which accepts ordinary pans and zooms.
    The support denominator is all previous corners, not only surviving tracks:
    a few coincidental matches after a cut must not establish camera continuity.
    """
    if previous.shape != current.shape or previous.ndim != 2:
        raise ValueError("Compare two grayscale camera samples of the same shape.")
    pixel_change = current.astype(np.float32) - previous.astype(np.float32)
    brightness_shift = float(np.median(pixel_change))
    structure_residual = float(np.mean(np.abs(pixel_change - brightness_shift))) / 255
    metrics = {
        "structure_residual": structure_residual,
        "brightness_shift": brightness_shift / 255,
        "motion_inlier_fraction": 0.0,
        "motion_checked": False,
        "previous_corners": 0,
        "consistent_matches": 0,
    }
    # Most samples are ordinary motion and need no optical-flow computation.
    if structure_residual < CAMERA_CUT_SETTINGS["min_structure_residual"]:
        metrics["is_cut"] = False
        return metrics
    metrics["motion_checked"] = True
    # Player controls, operating system taskbars and broadcast edge overlays can
    # stay perfectly still across a real camera change. They must not establish
    # continuity of the video content. Keep a broad fixed normalized interior;
    # detector coordinates and the saved replay remain the original full frame.
    h, w = previous.shape
    x1, y1, x2, y2 = CAMERA_CUT_SETTINGS["motion_support_region"]
    mask = np.zeros_like(previous, dtype=np.uint8)
    mask[round(y1*h):round(y2*h), round(x1*w):round(x2*w)] = 255
    corners = cv2.goodFeaturesToTrack(
        previous, maxCorners=CAMERA_CUT_SETTINGS["max_corners"],
        qualityLevel=0.01, minDistance=CAMERA_CUT_SETTINGS["min_corner_distance"],
        mask=mask,
    )
    if corners is not None:
        metrics["previous_corners"] = len(corners)
    if corners is not None and len(corners) >= 6:
        # A large foreground occlusion can corrupt coarse pyramid estimates.
        # Retry nearer the original sample pixels before declaring lost motion.
        for max_level in CAMERA_CUT_SETTINGS["optical_flow_pyramid_levels"]:
            following, status, _ = cv2.calcOpticalFlowPyrLK(
                previous, current, corners, None, winSize=(21, 21), maxLevel=max_level,
            )
            if following is None or status is None:
                continue
            returned, reverse_status, _ = cv2.calcOpticalFlowPyrLK(
                current, previous, following, None, winSize=(21, 21), maxLevel=max_level,
            )
            if returned is None or reverse_status is None:
                continue
            valid = (
                status.ravel().astype(bool)
                & reverse_status.ravel().astype(bool)
                & (np.linalg.norm(returned - corners, axis=2).ravel()
                   < CAMERA_CUT_SETTINGS["max_forward_backward_error_pixels"])
            )
            metrics["consistent_matches"] = max(metrics["consistent_matches"], int(valid.sum()))
            if valid.sum() >= 6:
                _, inliers = cv2.estimateAffinePartial2D(
                    corners[valid], following[valid], method=cv2.RANSAC,
                    ransacReprojThreshold=CAMERA_CUT_SETTINGS["ransac_error_pixels"],
                )
                if inliers is not None:
                    support = float(inliers.sum()) / len(corners)
                    metrics["motion_inlier_fraction"] = max(metrics["motion_inlier_fraction"], support)
            if metrics["motion_inlier_fraction"] >= CAMERA_CUT_SETTINGS["max_motion_inlier_fraction"]:
                break
    metrics["is_cut"] = metrics["motion_inlier_fraction"] < CAMERA_CUT_SETTINGS["max_motion_inlier_fraction"]
    return metrics


class CameraCutDetector:
    """Retain only the previous source sample; never infer hidden car positions."""

    def __init__(self):
        self.previous: np.ndarray | None = None
        self.last_metrics: dict | None = None

    def reset(self) -> None:
        self.previous = None
        self.last_metrics = None

    def update(self, frame: np.ndarray) -> bool:
        current = _sample(frame)
        self.last_metrics = compare_samples(self.previous, current) if self.previous is not None else None
        self.previous = current
        return bool(self.last_metrics and self.last_metrics["is_cut"])
