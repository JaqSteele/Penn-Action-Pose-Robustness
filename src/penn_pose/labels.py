"""Loading and validating Penn Action annotation files, with a test-partition lock.

Each video has one MATLAB file ``labels/<id>.mat``. The official partition is given by its
``train`` variable: 1 for the training partition, -1 for the test partition.

Test-partition lock (protocol Section 4.3):
- Before any other field of a file is read, only its ``train`` variable is read.
- Loading a test-partition video raises ``TestPartitionAccessError`` unless the caller passes
  ``allow_test=True`` explicitly. Listing test-partition identifiers needs the same opt-in.
- No other variable of a test-partition file is read without that opt-in.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import scipy.io as sio

PathLike = Union[str, Path]

#: Local default dataset location (outside the repository). Override with PENN_ACTION_ROOT.
DEFAULT_DATA_ROOT = Path(os.environ.get("PENN_ACTION_ROOT", r"C:\datasets\penn-action"))

N_PENN_JOINTS = 13
TRAIN_FLAG = 1
TEST_FLAG = -1

#: Widths recorded in ``dimensions`` that correspond to 480-pixel-wide frames (protocol D-05, v0.9).
_WIDTH_CORRECTIONS = {481: 480, 482: 480}


class LabelFormatError(ValueError):
    """An annotation file is missing, malformed or internally inconsistent."""


class TestPartitionAccessError(PermissionError):
    """Raised when test-partition data is requested without explicit opt-in."""


@dataclass(frozen=True)
class PennVideo:
    """One Penn Action video's annotations, with fields cast to consistent types.

    Arrays use Penn Action's original 13-column order (protocol Section 6.2). ``bbox`` keeps
    its original number of rows, which may differ from the number of frames; see
    ``boxes.align_boxes``.
    """

    video_id: str
    action: str
    pose: str
    is_train: bool
    x: np.ndarray            # (n_frames, 13) float64, image pixels
    y: np.ndarray            # (n_frames, 13) float64, image pixels
    visibility: np.ndarray   # (n_frames, 13) bool
    bbox: np.ndarray         # (n_rows, 4) float64, [x1, y1, x2, y2]
    n_frames: int
    image_width: int         # actual frame width (481/482 in ``dimensions`` corrected to 480)
    image_height: int
    dims_width: int          # width as recorded in ``dimensions``


def labels_dir(root: PathLike = DEFAULT_DATA_ROOT) -> Path:
    return Path(root) / "labels"


def _label_path(video_id: str, root: PathLike) -> Path:
    if not (isinstance(video_id, str) and len(video_id) == 4 and video_id.isdigit()):
        raise ValueError(f"video_id must be a four-digit string such as '0001', got {video_id!r}")
    path = labels_dir(root) / f"{video_id}.mat"
    if not path.is_file():
        raise LabelFormatError(f"Annotation file not found: {path}")
    return path


def _scalar(value, name: str, path: Path) -> float:
    arr = np.asarray(value).ravel()
    if arr.size != 1:
        raise LabelFormatError(f"{path.name}: '{name}' should be a scalar, got shape {np.shape(value)}")
    return arr[0]


def _string(value, name: str, path: Path) -> str:
    arr = np.asarray(value).ravel()
    if arr.size != 1:
        raise LabelFormatError(f"{path.name}: '{name}' should be a single string")
    return str(arr[0])


def read_partition_flag(video_id: str, root: PathLike = DEFAULT_DATA_ROOT) -> int:
    """Return 1 (training) or -1 (test), reading only the file's ``train`` variable."""
    path = _label_path(video_id, root)
    data = sio.loadmat(path, variable_names=["train"])
    if "train" not in data:
        raise LabelFormatError(f"{path.name}: missing 'train' variable")
    flag = int(_scalar(data["train"], "train", path))
    if flag not in (TRAIN_FLAG, TEST_FLAG):
        raise LabelFormatError(f"{path.name}: 'train' must be 1 or -1, got {flag}")
    return flag


def all_video_ids(root: PathLike = DEFAULT_DATA_ROOT) -> List[str]:
    """All four-digit video identifiers that have an annotation file, in ascending order."""
    directory = labels_dir(root)
    if not directory.is_dir():
        raise LabelFormatError(f"Labels directory not found: {directory}")
    ids = sorted(p.stem for p in directory.glob("*.mat"))
    bad = [i for i in ids if not (len(i) == 4 and i.isdigit())]
    if bad:
        raise LabelFormatError(f"Unexpected annotation file names: {bad[:5]}")
    return ids


def partition_flags(root: PathLike = DEFAULT_DATA_ROOT) -> Dict[str, int]:
    """Partition flag of every video. Reads only the ``train`` variable of each file."""
    return {vid: read_partition_flag(vid, root) for vid in all_video_ids(root)}


def list_video_ids(partition: str = "train", root: PathLike = DEFAULT_DATA_ROOT,
                   allow_test: bool = False) -> List[str]:
    """Video identifiers of one partition ('train', 'test' or 'all'), in ascending order.

    Listing 'test' or 'all' requires ``allow_test=True``.
    """
    if partition not in ("train", "test", "all"):
        raise ValueError("partition must be 'train', 'test' or 'all'")
    if partition in ("test", "all") and not allow_test:
        raise TestPartitionAccessError(
            f"Listing the '{partition}' partition requires allow_test=True (protocol Section 4.3).")
    flags = partition_flags(root)
    if partition == "train":
        return [v for v, f in flags.items() if f == TRAIN_FLAG]
    if partition == "test":
        return [v for v, f in flags.items() if f == TEST_FLAG]
    return list(flags)


def load_video(video_id: str, root: PathLike = DEFAULT_DATA_ROOT,
               allow_test: bool = False) -> PennVideo:
    """Load and validate one video's annotations.

    The partition flag is checked first. A test-partition video raises
    ``TestPartitionAccessError`` unless ``allow_test=True``.
    """
    flag = read_partition_flag(video_id, root)
    if flag == TEST_FLAG and not allow_test:
        raise TestPartitionAccessError(
            f"Video {video_id} belongs to the official test partition. Loading it requires "
            "allow_test=True, which is reserved for the final, frozen evaluation (protocol Section 4.3).")

    path = _label_path(video_id, root)
    data = sio.loadmat(path)
    required = ["action", "pose", "train", "x", "y", "visibility", "bbox", "dimensions", "nframes"]
    missing = [k for k in required if k not in data]
    if missing:
        raise LabelFormatError(f"{path.name}: missing variables {missing}")

    x = np.asarray(data["x"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.float64)
    vis_raw = np.asarray(data["visibility"])
    if x.ndim != 2 or x.shape[1] != N_PENN_JOINTS:
        raise LabelFormatError(f"{path.name}: 'x' should have shape (frames, 13), got {x.shape}")
    if y.shape != x.shape or vis_raw.shape != x.shape:
        raise LabelFormatError(f"{path.name}: 'x', 'y' and 'visibility' shapes differ: "
                               f"{x.shape}, {y.shape}, {vis_raw.shape}")
    if not np.all(np.isin(vis_raw, (0, 1))):
        raise LabelFormatError(f"{path.name}: 'visibility' must contain only 0 and 1")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise LabelFormatError(f"{path.name}: non-finite joint coordinates")

    dims = np.asarray(data["dimensions"]).ravel()
    if dims.size != 3:
        raise LabelFormatError(f"{path.name}: 'dimensions' should hold [height, width, frames]")
    height, dims_width, dims_frames = (int(v) for v in dims)
    n_frames = x.shape[0]
    nframes = int(_scalar(data["nframes"], "nframes", path))
    if not (n_frames == nframes == dims_frames):
        raise LabelFormatError(f"{path.name}: frame counts disagree (joint rows {n_frames}, "
                               f"nframes {nframes}, dimensions {dims_frames})")
    if height <= 0 or dims_width <= 0:
        raise LabelFormatError(f"{path.name}: invalid image dimensions {dims.tolist()}")

    bbox = np.asarray(data["bbox"], dtype=np.float64)
    if bbox.size == 0:
        bbox = np.zeros((0, 4), dtype=np.float64)
    if bbox.ndim != 2 or bbox.shape[1] != 4:
        raise LabelFormatError(f"{path.name}: 'bbox' should have shape (rows, 4), got {bbox.shape}")

    return PennVideo(
        video_id=video_id,
        action=_string(data["action"], "action", path),
        pose=_string(data["pose"], "pose", path),
        is_train=(flag == TRAIN_FLAG),
        x=x, y=y, visibility=vis_raw.astype(bool), bbox=bbox,
        n_frames=n_frames,
        image_width=_WIDTH_CORRECTIONS.get(dims_width, dims_width),
        image_height=height,
        dims_width=dims_width,
    )


def load_training_actions(root: PathLike = DEFAULT_DATA_ROOT) -> Dict[str, str]:
    """Action label of every training-partition video. Test-partition files are not opened
    beyond their ``train`` variable."""
    actions = {}
    for vid in list_video_ids("train", root):
        path = _label_path(vid, root)
        data = sio.loadmat(path, variable_names=["action"])
        if "action" not in data:
            raise LabelFormatError(f"{path.name}: missing 'action'")
        actions[vid] = _string(data["action"], "action", path)
    return actions


def iter_training_videos(root: PathLike = DEFAULT_DATA_ROOT, ids: Optional[List[str]] = None):
    """Yield ``PennVideo`` objects for training-partition videos (all, or the given ids)."""
    for vid in (ids if ids is not None else list_video_ids("train", root)):
        yield load_video(vid, root, allow_test=False)
