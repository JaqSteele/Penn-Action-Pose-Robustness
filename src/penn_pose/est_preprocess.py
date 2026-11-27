"""Estimated-pose preparation: invalid-keypoint imputation (D-34a step 1, D-34b steps 2-4) and
the estimated-pose pipeline (protocol Sections 5.5.4, 9.4 and 9.4.3).

Imputation works on the 12 primary keypoints (canonical COCO 5-16 order) in original-image
pixel space, at the original frame rate, before normalisation. It uses ESTIMATOR OUTPUTS AND
IMAGE SIZE ONLY: never ground-truth joint coordinates, ground-truth visibility or box centres.

- Step 1 (D-34a): an invalid keypoint in a track that has at least one valid estimate is
  replaced by linear interpolation in time between that keypoint's nearest valid estimates,
  holding the nearest valid estimate at the start or end of the video.
- Step 2 (D-34b): a keypoint never valid in the video copies its left/right partner's track
  (after step 1), if the partner has at least one valid estimate.
- Step 3 (D-34b): if the partner is also never valid, the constant per-video mean of the
  ORIGINAL valid estimated primary observations (x and y averaged separately over all valid
  joint-frames, before any imputation).
- Step 4 (D-34b): if no primary keypoint is valid anywhere in the video, the image centre
  (W/2, H/2) from the actual image size; the video is flagged, retained and reported.

Normalisation then uses the per-video centre and scale computed from the video's ground-truth
annotation BOXES only (the documented D-05 exception to D-07a rule 4), obtained through the
locked annotation loader, identically to the ground-truth pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from . import boxes, labels, mapping, normalise, pose_io

PathLike = Union[str, Path]

#: Default location of the training-partition RTMPose-M outputs (outside the repository).
DEFAULT_POSE_DIR = labels.DEFAULT_DATA_ROOT / "poses" / "rtmpose_m" / "train"


class EstimateConsistencyError(ValueError):
    """The pose output and the annotation file disagree (frames, image size or boxes)."""


@dataclass
class ImputationReport:
    n_frames: int = 0
    n_invalid: int = 0                                         # invalid primary observations
    step1_points: int = 0                                      # observations filled by step 1
    step1_tracks: List[int] = field(default_factory=list)      # primary joint indices
    step2_tracks: List[int] = field(default_factory=list)      # copied from the partner
    step3_tracks: List[int] = field(default_factory=list)      # per-video mean of valid estimates
    step3_mean: Optional[Tuple[float, float]] = None
    step4_video: bool = False                                  # image centre used for every track

    @property
    def step2_points(self) -> int:
        return len(self.step2_tracks) * self.n_frames

    @property
    def step3_points(self) -> int:
        return len(self.step3_tracks) * self.n_frames

    @property
    def step4_points(self) -> int:
        return self.n_frames * mapping.N_PRIMARY if self.step4_video else 0


def check_image_size(image_size) -> Tuple[int, int]:
    """Validate an image size [W, H]: two positive integers."""
    arr = np.asarray(image_size)
    if arr.shape != (2,) or arr.dtype.kind not in "iu":
        raise ValueError(f"image_size must be two integers [W, H], got {image_size!r}")
    w, h = int(arr[0]), int(arr[1])
    if w <= 0 or h <= 0:
        raise ValueError(f"image_size must be positive, got {[w, h]}")
    return w, h


def impute_estimates(keypoints12: np.ndarray, valid12: np.ndarray, image_size):
    """Apply D-34a step 1 and D-34b steps 2-4. Returns (imputed copy, ImputationReport).

    ``keypoints12``: (frames, 12, 2) pixels; ``valid12``: (frames, 12) D-34a validity;
    ``image_size``: actual [W, H] of the frames. The inputs are not modified.
    """
    kp = np.asarray(keypoints12, dtype=np.float64)
    valid = np.asarray(valid12)
    if kp.ndim != 3 or kp.shape[1:] != (mapping.N_PRIMARY, 2) or kp.shape[0] < 1:
        raise ValueError(f"expected keypoints (frames, {mapping.N_PRIMARY}, 2), got {kp.shape}")
    if valid.dtype != np.bool_ or valid.shape != kp.shape[:2]:
        raise ValueError(f"expected a boolean validity mask of shape {kp.shape[:2]}, "
                         f"got {valid.dtype} {valid.shape}")
    if not np.all(np.isfinite(kp[valid])):
        raise ValueError("a keypoint marked valid has a non-finite coordinate (violates D-34a)")
    width, height = check_image_size(image_size)

    n_frames = kp.shape[0]
    out = kp.copy()
    report = ImputationReport(n_frames=n_frames, n_invalid=int((~valid).sum()))
    t = np.arange(n_frames, dtype=np.float64)

    # Step 4: no primary keypoint valid anywhere in the video.
    if not valid.any():
        out[:] = np.array([width / 2.0, height / 2.0])
        report.step4_video = True
        return out, report

    # Step 1: same-joint linear interpolation, holding the nearest valid value at the ends.
    for j in range(mapping.N_PRIMARY):
        ok = valid[:, j]
        if ok.any() and not ok.all():
            bad = ~ok
            for d in range(2):
                out[bad, j, d] = np.interp(t[bad], t[ok], kp[ok, j, d])
            report.step1_tracks.append(j)
            report.step1_points += int(bad.sum())

    # Steps 2 and 3: tracks never valid in the video.
    never = ~valid.any(axis=0)
    original_mean = kp[valid].mean(axis=0)          # original valid observations only
    for j in np.flatnonzero(never):
        p = mapping.PARTNER[j]
        if not never[p]:
            out[:, j, :] = out[:, p, :]             # the partner track after step 1
            report.step2_tracks.append(int(j))
        else:
            out[:, j, :] = original_mean
            report.step3_tracks.append(int(j))
    if report.step3_tracks:
        report.step3_mean = (float(original_mean[0]), float(original_mean[1]))

    if not np.all(np.isfinite(out)):
        raise AssertionError("imputation left a non-finite coordinate")
    return out, report


@dataclass
class PreparedEstimate:
    video_id: str
    keypoints_raw: np.ndarray        # (frames, 12, 2) raw estimator output (may hold invalid values)
    valid: np.ndarray                # (frames, 12) D-34a validity
    keypoints_imputed: np.ndarray    # (frames, 12, 2) pixels, after D-34a/b imputation
    keypoints_normalised: np.ndarray # (frames, 12, 2) D-12 normalised
    image_size: Tuple[int, int]      # actual [W, H]
    frame: boxes.VideoFrame
    report: ImputationReport


def check_consistency(video: labels.PennVideo, arrays: Dict[str, np.ndarray]) -> Tuple[int, int]:
    """Cross-check a pose output against its annotation file. Returns the image size (W, H).

    Uses only the frame count, the image size and the annotation boxes of ``video``.
    """
    vid = video.video_id
    n = int(arrays["frame_index"].shape[0])
    if n != video.n_frames:
        raise EstimateConsistencyError(f"{vid}: pose output has {n} frames, annotations {video.n_frames}")
    size = check_image_size(arrays["image_size"])
    if size != (video.image_width, video.image_height):
        raise EstimateConsistencyError(f"{vid}: pose-output image size {list(size)} differs from the "
                                       f"annotation image size {[video.image_width, video.image_height]}")
    aligned = boxes.align_boxes(video.bbox, video.n_frames)
    if not np.array_equal(arrays["bbox_annotation"], aligned, equal_nan=True):
        raise EstimateConsistencyError(f"{vid}: stored bbox_annotation differs from the annotation boxes")
    return size


def prepare_estimated_from(video: labels.PennVideo, arrays: Dict[str, np.ndarray]) -> PreparedEstimate:
    """Pipeline for already-loaded inputs. From ``video`` only the frame count, image size and
    boxes are used; its ground-truth joints and visibility are never read."""
    if not video.is_train:
        raise labels.TestPartitionAccessError(f"{video.video_id}: not a training-partition video")
    size = check_consistency(video, arrays)
    frame = boxes.video_centre_scale(video.bbox, video.n_frames, size[0], size[1])
    imputed, report = impute_estimates(arrays["keypoints12"], arrays["valid12"], size)
    return PreparedEstimate(
        video_id=video.video_id,
        keypoints_raw=np.asarray(arrays["keypoints12"], dtype=np.float64),
        valid=np.asarray(arrays["valid12"]),
        keypoints_imputed=imputed,
        keypoints_normalised=normalise.normalise(imputed, frame.centre, frame.scale),
        image_size=size,
        frame=frame,
        report=report,
    )


def prepare_estimated(video_id: str, root: PathLike = labels.DEFAULT_DATA_ROOT,
                      pose_dir: PathLike = DEFAULT_POSE_DIR) -> PreparedEstimate:
    """Load (locked) annotations and the validated pose output, then impute and normalise.

    The annotation file is loaded first, through the test-partition lock, so a test-partition
    identifier fails before any pose file is opened.
    """
    video = labels.load_video(video_id, root)
    arrays, _meta = pose_io.load_video_output(pose_dir, video_id)
    return prepare_estimated_from(video, arrays)


def summarise_reports(reports: Dict[str, ImputationReport]) -> dict:
    return {
        "videos": len(reports),
        "frames": sum(r.n_frames for r in reports.values()),
        "invalid_observations": sum(r.n_invalid for r in reports.values()),
        "step1_points": sum(r.step1_points for r in reports.values()),
        "step1_tracks": sum(len(r.step1_tracks) for r in reports.values()),
        "step1_videos": sorted(v for v, r in reports.items() if r.step1_points),
        "step2_points": sum(r.step2_points for r in reports.values()),
        "step2_tracks": sum(len(r.step2_tracks) for r in reports.values()),
        "step2_videos": sorted(v for v, r in reports.items() if r.step2_tracks),
        "step3_points": sum(r.step3_points for r in reports.values()),
        "step3_tracks": sum(len(r.step3_tracks) for r in reports.values()),
        "step3_videos": sorted(v for v, r in reports.items() if r.step3_tracks),
        "step4_points": sum(r.step4_points for r in reports.values()),
        "step4_videos": sorted(v for v, r in reports.items() if r.step4_video),
    }
