"""Shared helpers for the test suite (standard-library unittest).

Run from the repository root:  python -m unittest discover -s tests -v
Tests that need the local dataset are skipped if it is not present.
"""

import functools
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from penn_pose import labels  # noqa: E402

DATA_ROOT = labels.DEFAULT_DATA_ROOT
HAVE_DATA = (DATA_ROOT / "labels").is_dir()
requires_data = unittest.skipUnless(HAVE_DATA, f"Penn Action labels not found under {DATA_ROOT}")


@functools.lru_cache(maxsize=1)
def training_videos():
    """All training-partition videos, loaded once per test run (test partition never loaded)."""
    return {v.video_id: v for v in labels.iter_training_videos(DATA_ROOT)}


# Batch D1: training-partition RTMPose-M outputs and recogniser inputs.
POSE_DIR = DATA_ROOT / "poses" / "rtmpose_m" / "train"
HAVE_POSES = HAVE_DATA and POSE_DIR.is_dir()
requires_poses = unittest.skipUnless(HAVE_POSES, f"RTMPose training outputs not found under {POSE_DIR}")


@functools.lru_cache(maxsize=1)
def training_inputs():
    """(gt, est) recogniser inputs for all 1,258 training videos, built once per test run."""
    from penn_pose import dataset
    vids = training_videos()
    ids = sorted(vids)
    gt = dataset.build_inputs("gt", ids, DATA_ROOT, POSE_DIR, videos=vids)
    est = dataset.build_inputs("est", ids, DATA_ROOT, POSE_DIR, videos=vids)
    return gt, est
