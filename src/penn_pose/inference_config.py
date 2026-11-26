"""RTMPose-M inference settings (protocol D-03, D-04) and the smoke-test video rule.

This module has no MMPose dependency, so it can be imported and tested from the main environment.
The settings below restate the protocol; they are not tunable options.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from . import labels

SETUP_DIR = Path(r"C:\venvs\penn-action-rtmpose-setup")
CONFIG_PATH = SETUP_DIR / "mmpose-v1.3.2" / "configs" / "body_2d_keypoint" / "rtmpose" / "coco" / "rtmpose-m_8xb256-420e_coco-256x192.py"
CHECKPOINT_PATH = SETUP_DIR / "checkpoints" / "rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth"
CONFIG_SHA256 = "f27da50a81c3d3b5c426262118b9a030d15c9dbddccda7100b8cc0bcd07f6fff"
CHECKPOINT_SHA256 = "929f00f2d84764aa0297decf13a84556642e43cbf5fca8cfb16e7326d92dfd97"

SETTINGS = {
    "model": "RTMPose-M",
    "mmpose_version": "1.3.2",
    "input_size_wh": [192, 256],
    "flip_test": True,
    "precision": "fp32",
    "bbox_format": "xyxy",
    "image_input": "file_path",
    "boxes_per_call": 1,
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "cudnn_allow_tf32": False,
    "matmul_allow_tf32": False,
    "bbox_rules": "D-05 (protocol v0.9): validity incl. diagonal >= 0.05 x image diagonal; nearest valid box (earlier on ties); full image if none",
    "validity_rule": "D-34a: invalid if score <= 0, score non-finite, or any coordinate non-finite",
    "primary_order": "COCO keypoints 5-16 (protocol v0.12)",
}

SMOKE_N = 5


def apply_determinism(torch_module) -> dict:
    """Apply the D-04 numerical settings to a torch module; return the resulting flags."""
    torch_module.backends.cudnn.benchmark = SETTINGS["cudnn_benchmark"]
    torch_module.backends.cudnn.deterministic = SETTINGS["cudnn_deterministic"]
    torch_module.backends.cudnn.allow_tf32 = SETTINGS["cudnn_allow_tf32"]
    torch_module.backends.cuda.matmul.allow_tf32 = SETTINGS["matmul_allow_tf32"]
    return {
        "cudnn_benchmark": bool(torch_module.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch_module.backends.cudnn.deterministic),
        "cudnn_allow_tf32": bool(torch_module.backends.cudnn.allow_tf32),
        "matmul_allow_tf32": bool(torch_module.backends.cuda.matmul.allow_tf32),
    }


def smoke_test_video_ids(fitting_ids: Iterable[str], n: int = SMOKE_N) -> List[str]:
    """Deterministic, quality-blind selection: n evenly spaced positions in the sorted fitting list
    (positions floor(k * (len - 1) / (n - 1)) for k = 0..n-1)."""
    ids = sorted(fitting_ids)
    if len(ids) < n:
        raise ValueError("not enough fitting videos")
    return [ids[(k * (len(ids) - 1)) // (n - 1)] for k in range(n)]


def check_inference_ids(video_ids: Iterable[str], root=labels.DEFAULT_DATA_ROOT) -> List[str]:
    """Raise TestPartitionAccessError if any requested video is in the test partition."""
    ids = list(video_ids)
    for vid in ids:
        if labels.read_partition_flag(vid, root) != labels.TRAIN_FLAG:
            raise labels.TestPartitionAccessError(
                f"Video {vid} is in the test partition; inference on test videos is not authorised.")
    return ids
