"""The 12-joint primary representation (protocol Sections 6.1-6.5; D-08, D-09, D-11).

Primary joint order used throughout the project (named by the subject's ANATOMICAL side):

    index  name             COCO index   Penn column (0-based)  Penn label (1-based)
    0      left_shoulder    5            2                      Right shoulder (3)
    1      right_shoulder   6            1                      Left shoulder (2)
    2      left_elbow       7            4                      Right elbow (5)
    3      right_elbow      8            3                      Left elbow (4)
    4      left_wrist       9            6                      Right wrist (7)
    5      right_wrist      10           5                      Left wrist (6)
    6      left_hip         11           8                      Right hip (9)
    7      right_hip        12           7                      Left hip (8)
    8      left_knee        13           10                     Right knee (11)
    9      right_knee       14           9                      Left knee (10)
    10     left_ankle       15           12                     Right ankle (13)
    11     right_ankle      16           11                     Left ankle (12)

Penn Action columns labelled LEFT hold the subject's anatomical RIGHT (D-09), so each primary
joint takes the Penn column of the OPPOSITE label. The head (Penn column 0) is excluded (D-08).
The order follows COCO keypoints 5-16, so the estimator output maps without reordering.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

PRIMARY_JOINTS = (
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
)
N_PRIMARY = len(PRIMARY_JOINTS)

#: Penn Action's 13 column labels in array-column order (D-11), as labelled in the dataset.
PENN_COLUMN_LABELS = (
    "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
    "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
)

#: The 17 COCO keypoint names in RTMPose output order.
COCO_KEYPOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


def _opposite(name: str) -> str:
    side, part = name.split("_", 1)
    return ("right_" if side == "left" else "left_") + part


#: For each primary joint, the Penn column holding it (opposite-named column, D-09).
PENN_COLUMN_FOR_PRIMARY = tuple(PENN_COLUMN_LABELS.index(_opposite(j)) for j in PRIMARY_JOINTS)
#: For each primary joint, its COCO keypoint index (same anatomical name).
COCO_INDEX_FOR_PRIMARY = tuple(COCO_KEYPOINTS.index(j) for j in PRIMARY_JOINTS)
#: For each primary joint, the index of its left/right partner in the primary order.
PARTNER = tuple(PRIMARY_JOINTS.index(_opposite(j)) for j in PRIMARY_JOINTS)


def penn_to_primary(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Penn arrays (frames, 13) -> primary joints (frames, 12, 2) in anatomical order."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(PENN_COLUMN_LABELS) or y.shape != x.shape:
        raise ValueError(f"expected two (frames, 13) arrays, got {x.shape} and {y.shape}")
    cols = list(PENN_COLUMN_FOR_PRIMARY)
    return np.stack([x[:, cols], y[:, cols]], axis=-1)


def penn_visibility_to_primary(visibility: np.ndarray) -> np.ndarray:
    """Penn visibility (frames, 13) -> (frames, 12), same joint order as ``penn_to_primary``."""
    visibility = np.asarray(visibility)
    if visibility.ndim != 2 or visibility.shape[1] != len(PENN_COLUMN_LABELS):
        raise ValueError(f"expected (frames, 13), got {visibility.shape}")
    return visibility[:, list(PENN_COLUMN_FOR_PRIMARY)]


def coco_to_primary(keypoints: np.ndarray, keypoint_names: Sequence[str] = COCO_KEYPOINTS) -> np.ndarray:
    """COCO keypoints (..., 17, D) -> (..., 12, D), checked against the estimator's names."""
    if tuple(keypoint_names) != COCO_KEYPOINTS:
        raise ValueError("estimator keypoint names do not match the expected COCO-17 order")
    keypoints = np.asarray(keypoints)
    if keypoints.shape[-2] != len(COCO_KEYPOINTS):
        raise ValueError(f"expected 17 keypoints on axis -2, got shape {keypoints.shape}")
    return keypoints[..., list(COCO_INDEX_FOR_PRIMARY), :]
