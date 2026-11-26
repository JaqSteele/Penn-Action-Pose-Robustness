"""Estimator outputs: D-34a validity flags and the per-video output format.

Validity (protocol D-34a, Section 5.5.4): an estimated keypoint is INVALID if its score is <= 0,
its score is not finite, or either coordinate is not finite. A finite coordinate with a score
<= 0 is still invalid. Detection never uses the apparent location of a keypoint.

Per-video output (outside the repository): ``<out_dir>/<video_id>.npz`` holding the arrays
below, plus ``<video_id>.json`` with metadata (settings, checkpoint hash, versions, flags).
``np.savez`` is used without pickled objects, so files load with ``allow_pickle=False``.
Existing outputs are never overwritten.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np

from . import mapping

PathLike = Union[str, Path]

#: name -> (dtype kind, shape builder given n_frames)
ARRAY_SPEC = {
    "frame_index": ("i", lambda n: (n,)),          # 1-based frame numbers
    "image_size": ("i", lambda n: (2,)),           # actual [width, height] used for inference
    "bbox_annotation": ("f", lambda n: (n, 4)),    # aligned annotation box (NaN where no row)
    "bbox_used": ("f", lambda n: (n, 4)),          # box given to the estimator (D-05)
    "bbox_replaced": ("b", lambda n: (n,)),        # True where the frame's own box was invalid
    "keypoints17": ("f", lambda n: (n, 17, 2)),    # raw RTMPose output, image pixels
    "scores17": ("f", lambda n: (n, 17)),
    "valid17": ("b", lambda n: (n, 17)),           # D-34a
    "keypoints12": ("f", lambda n: (n, mapping.N_PRIMARY, 2)),   # canonical COCO 5-16 order
    "scores12": ("f", lambda n: (n, mapping.N_PRIMARY)),
    "valid12": ("b", lambda n: (n, mapping.N_PRIMARY)),
}


class OutputFormatError(ValueError):
    pass


def keypoint_validity(keypoints: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """D-34a validity mask (..., K) from keypoints (..., K, 2) and scores (..., K)."""
    keypoints = np.asarray(keypoints)
    scores = np.asarray(scores)
    if keypoints.shape[:-1] != scores.shape or keypoints.shape[-1] != 2:
        raise ValueError(f"shape mismatch: keypoints {keypoints.shape}, scores {scores.shape}")
    with np.errstate(invalid="ignore"):
        return np.isfinite(scores) & (scores > 0) & np.all(np.isfinite(keypoints), axis=-1)


def output_paths(out_dir: PathLike, video_id: str) -> Tuple[Path, Path]:
    out_dir = Path(out_dir)
    return out_dir / f"{video_id}.npz", out_dir / f"{video_id}.json"


def validate_arrays(arrays: Dict[str, np.ndarray]) -> int:
    """Check names, dtypes and shapes; return the number of frames."""
    missing = sorted(set(ARRAY_SPEC) - set(arrays))
    extra = sorted(set(arrays) - set(ARRAY_SPEC))
    if missing or extra:
        raise OutputFormatError(f"missing arrays {missing}, unexpected arrays {extra}")
    n = int(np.asarray(arrays["frame_index"]).shape[0])
    for name, (kind, shape_fn) in ARRAY_SPEC.items():
        a = np.asarray(arrays[name])
        if a.dtype.kind != kind:
            raise OutputFormatError(f"{name}: dtype kind {a.dtype.kind!r}, expected {kind!r}")
        if a.shape != shape_fn(n):
            raise OutputFormatError(f"{name}: shape {a.shape}, expected {shape_fn(n)}")
    if not np.array_equal(arrays["frame_index"], np.arange(1, n + 1)):
        raise OutputFormatError("frame_index must be 1..n_frames in order")
    if not np.array_equal(arrays["valid17"], keypoint_validity(arrays["keypoints17"], arrays["scores17"])):
        raise OutputFormatError("valid17 does not match the D-34a rule")
    # equal_nan: an invalid (non-finite) keypoint must still match its own copy.
    if not np.array_equal(arrays["keypoints12"], mapping.coco_to_primary(arrays["keypoints17"]), equal_nan=True):
        raise OutputFormatError("keypoints12 is not the COCO 5-16 selection of keypoints17")
    if not np.array_equal(arrays["valid12"], arrays["valid17"][:, list(mapping.COCO_INDEX_FOR_PRIMARY)]):
        raise OutputFormatError("valid12 does not match valid17")
    if not np.array_equal(arrays["scores12"], arrays["scores17"][:, list(mapping.COCO_INDEX_FOR_PRIMARY)], equal_nan=True):
        raise OutputFormatError("scores12 does not match scores17")
    return n


def save_video_output(out_dir: PathLike, video_id: str, arrays: Dict[str, np.ndarray], meta: dict) -> None:
    validate_arrays(arrays)
    npz_path, json_path = output_paths(out_dir, video_id)
    if npz_path.exists() or json_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output for {video_id} in {out_dir}")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    with open(npz_path, "xb") as fh:
        np.savez(fh, **{k: np.asarray(v) for k, v in arrays.items()})
    with open(json_path, "x", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)


def load_video_output(out_dir: PathLike, video_id: str):
    """Return (arrays, meta) for one video, validated."""
    npz_path, json_path = output_paths(out_dir, video_id)
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    validate_arrays(arrays)
    with open(json_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    if meta.get("video_id") != video_id:
        raise OutputFormatError(f"{json_path.name}: video_id mismatch")
    return arrays, meta
