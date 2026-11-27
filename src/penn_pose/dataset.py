"""Recogniser inputs for Release 1.0 (protocol Sections 9.4, 9.5.1; D-07b, D-12, D-20, D-34).

Per video, for source ``"gt"`` (ground truth) or ``"est"`` (RTMPose-M estimates):

    mapping (12 joints, COCO 5-16 order) -> unreliable coordinates (D-07b for "gt";
    D-34a/b for "est") -> per-video normalisation (D-12) -> resampling to 64 frames (D-20)

giving one array of shape (24, 64), float32.

**Channel layout (fixed for D1):** interleaved x/y. Channel ``2j`` is the x coordinate and
channel ``2j + 1`` the y coordinate of primary joint ``j``, in the canonical order
``mapping.PRIMARY_JOINTS`` (COCO keypoints 5-16). ``CHANNEL_NAMES`` lists them.

**Class map:** the dataset's own ``action`` strings, sorted with Python's ``sorted`` (code-point
order, as in the D-02 split), indexed 0-14. ``CLASS_NAMES`` records them; the build checks
them against the training labels and persists them next to the inputs.

Only training-partition videos are accepted. Every identifier's partition flag is checked
first (reading only the ``train`` variable), so a test-partition identifier raises
``TestPartitionAccessError`` before anything else of it is read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np

from . import est_preprocess, gt_preprocess, labels, mapping, pose_io, splits, temporal

PathLike = Union[str, Path]

SOURCES = ("gt", "est")
N_CHANNELS = 2 * mapping.N_PRIMARY
CHANNEL_NAMES = tuple(f"{j}_{axis}" for j in mapping.PRIMARY_JOINTS for axis in ("x", "y"))

#: The 15 action strings of the dataset, sorted; index = class label.
CLASS_NAMES = (
    "baseball_pitch", "baseball_swing", "bench_press", "bowl", "clean_and_jerk", "golf_swing",
    "jump_rope", "jumping_jacks", "pullup", "pushup", "situp", "squat", "strum_guitar",
    "tennis_forehand", "tennis_serve",
)
CLASS_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}


class DatasetError(ValueError):
    """The requested inputs violate an identity, split or alignment requirement."""


def class_names_from_actions(actions: Iterable[str]) -> List[str]:
    """Sorted distinct action strings; must equal ``CLASS_NAMES``."""
    names = sorted(set(actions))
    if tuple(names) != CLASS_NAMES:
        raise DatasetError(f"training action strings {names} do not match CLASS_NAMES {list(CLASS_NAMES)}")
    return names


def to_channels(joints: np.ndarray) -> np.ndarray:
    """(T, 12, 2) -> (24, T) with interleaved channels: 2j = x, 2j+1 = y of joint j."""
    joints = np.asarray(joints)
    if joints.ndim != 3 or joints.shape[1:] != (mapping.N_PRIMARY, 2):
        raise ValueError(f"expected (T, {mapping.N_PRIMARY}, 2), got {joints.shape}")
    return joints.reshape(joints.shape[0], N_CHANNELS).T


def from_channels(x: np.ndarray) -> np.ndarray:
    """Inverse of ``to_channels``: (24, T) -> (T, 12, 2)."""
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[0] != N_CHANNELS:
        raise ValueError(f"expected ({N_CHANNELS}, T), got {x.shape}")
    return x.T.reshape(x.shape[1], mapping.N_PRIMARY, 2)


def video_input(normalised: np.ndarray, length: int = temporal.T_FRAMES) -> np.ndarray:
    """Normalised (frames, 12, 2) -> resampled float32 (24, length)."""
    x = to_channels(temporal.resample_to_length(normalised, length)).astype(np.float32)
    if not np.all(np.isfinite(x)):
        raise DatasetError("non-finite value in a recogniser input")
    return np.ascontiguousarray(x)


def training_split(split_dir: PathLike = splits.DEFAULT_SPLIT_DIR) -> Dict[str, str]:
    """{video_id: 'fitting' | 'validation'} from the stored, hash-checked split lists."""
    fitting = splits.load_split("fitting", split_dir)
    validation = splits.load_split("validation", split_dir)
    overlap = set(fitting) & set(validation)
    if overlap:
        raise DatasetError(f"fitting and validation overlap: {sorted(overlap)[:5]}")
    out = {v: "fitting" for v in fitting}
    out.update({v: "validation" for v in validation})
    return out


def check_ids(video_ids: Sequence[str], root: PathLike = labels.DEFAULT_DATA_ROOT,
              split_dir: PathLike = splits.DEFAULT_SPLIT_DIR) -> Dict[str, str]:
    """Check identifiers: unique, training partition (lock first), and in the stored split.

    Returns {video_id: subset}.
    """
    ids = list(video_ids)
    if not ids:
        raise DatasetError("no video identifiers given")
    if len(set(ids)) != len(ids):
        raise DatasetError("video identifiers are not unique")
    for vid in ids:
        if labels.read_partition_flag(vid, root) != labels.TRAIN_FLAG:
            raise labels.TestPartitionAccessError(
                f"Video {vid} belongs to the official test partition (protocol Section 4.3).")
    membership = training_split(split_dir)
    missing = [v for v in ids if v not in membership]
    if missing:
        raise DatasetError(f"identifiers not in the stored fitting/validation lists: {missing[:5]}")
    return {v: membership[v] for v in ids}


@dataclass
class InputSet:
    source: str
    ids: np.ndarray                  # (n,) '<U4', in the requested order
    X: np.ndarray                    # (n, 24, 64) float32
    y: np.ndarray                    # (n,) int64 class indices into CLASS_NAMES
    subset: np.ndarray               # (n,) '<U10', 'fitting' or 'validation'
    n_frames: np.ndarray             # (n,) int64 original frame counts
    reports: Dict[str, object] = field(default_factory=dict)   # per-video preprocessing reports


def build_inputs(source: str, video_ids: Sequence[str], root: PathLike = labels.DEFAULT_DATA_ROOT,
                 pose_dir: PathLike = est_preprocess.DEFAULT_POSE_DIR,
                 split_dir: PathLike = splits.DEFAULT_SPLIT_DIR,
                 videos: Optional[Mapping[str, labels.PennVideo]] = None) -> InputSet:
    """Build recogniser inputs for training-partition videos.

    ``videos`` optionally supplies already-loaded annotations (each must be a training video
    with the requested identifier); otherwise they are loaded through the locked loader.
    """
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    membership = check_ids(video_ids, root, split_dir)
    ids = list(video_ids)
    X = np.empty((len(ids), N_CHANNELS, temporal.T_FRAMES), dtype=np.float32)
    y = np.empty(len(ids), dtype=np.int64)
    n_frames = np.empty(len(ids), dtype=np.int64)
    reports: Dict[str, object] = {}
    for i, vid in enumerate(ids):
        video = videos[vid] if videos is not None else labels.load_video(vid, root)
        if video.video_id != vid or not video.is_train:
            raise DatasetError(f"{vid}: supplied annotations are not the training video {vid}")
        if video.action not in CLASS_INDEX:
            raise DatasetError(f"{vid}: unknown action {video.action!r}")
        if source == "gt":
            prepared = gt_preprocess.prepare_gt(video)
            normalised = prepared.joints_normalised
        else:
            arrays, _meta = pose_io.load_video_output(pose_dir, vid)
            prepared = est_preprocess.prepare_estimated_from(video, arrays)
            normalised = prepared.keypoints_normalised
        X[i] = video_input(normalised)
        y[i] = CLASS_INDEX[video.action]
        n_frames[i] = video.n_frames
        reports[vid] = prepared.report
    return InputSet(source=source, ids=np.array(ids, dtype="<U4"), X=X, y=y,
                    subset=np.array([membership[v] for v in ids], dtype="<U10"),
                    n_frames=n_frames, reports=reports)


def check_input_set(s: InputSet) -> None:
    """Shapes, dtypes, finiteness, label range and identifier uniqueness of one set."""
    n = len(s.ids)
    if s.X.shape != (n, N_CHANNELS, temporal.T_FRAMES) or s.X.dtype != np.float32:
        raise DatasetError(f"{s.source}: X has {s.X.dtype} {s.X.shape}")
    if s.y.shape != (n,) or s.y.dtype != np.int64 or (n and not (0 <= s.y.min() and s.y.max() < len(CLASS_NAMES))):
        raise DatasetError(f"{s.source}: bad labels")
    if s.n_frames.shape != (n,) or s.subset.shape != (n,):
        raise DatasetError(f"{s.source}: bad metadata shapes")
    if len(set(s.ids.tolist())) != n:
        raise DatasetError(f"{s.source}: duplicate identifiers")
    if not np.all(np.isfinite(s.X)):
        raise DatasetError(f"{s.source}: non-finite input values")


def check_alignment(gt: InputSet, est: InputSet) -> None:
    """Ground-truth and estimated sets must describe the same videos, in the same order, with
    the same labels, subsets and original frame counts."""
    check_input_set(gt)
    check_input_set(est)
    if gt.source != "gt" or est.source != "est":
        raise DatasetError("expected a 'gt' set and an 'est' set")
    for name in ("ids", "y", "subset", "n_frames"):
        if not np.array_equal(getattr(gt, name), getattr(est, name)):
            raise DatasetError(f"ground-truth and estimated sets differ in {name}")
