"""Ground-truth bounding boxes: validity, fallbacks and per-video centre/scale (D-05, D-12).

Rules (protocol Section 5.2.2, v0.8 with the v0.9 amendment):
1. A box is valid if its row exists, its four values are finite, x2 > x1 and y2 > y1, and its
   diagonal is at least 0.05 x the actual image diagonal.
2. Box rows are aligned to frames from the first frame; frames without a row have an invalid
   box; surplus rows are ignored.
3. D-12 medians use valid boxes only.
4. A frame with an invalid box is cropped with the box of the nearest frame with a valid box
   in the same video (ties: the earlier frame).
5. A video with no valid box uses the full image [0, 0, W, H] for cropping, and the image centre
   and image diagonal for D-12.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_DIAGONAL_FRACTION = 0.05


@dataclass(frozen=True)
class VideoFrame:
    """Per-video normalisation frame (D-12)."""

    centre: np.ndarray      # (2,) [cx, cy] in pixels
    scale: float            # median valid-box diagonal, pixels
    used_fallback: bool     # True if no valid box existed (image centre and diagonal used)
    n_valid: int            # number of frames with a valid box


def align_boxes(bbox: np.ndarray, n_frames: int) -> np.ndarray:
    """(rows, 4) -> (n_frames, 4); missing rows become NaN, surplus rows are dropped."""
    bbox = np.asarray(bbox, dtype=np.float64).reshape(-1, 4)
    out = np.full((n_frames, 4), np.nan)
    k = min(n_frames, bbox.shape[0])
    out[:k] = bbox[:k]
    return out


def valid_box_mask(aligned: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    """Boolean mask of frames whose aligned box is valid under rule 1."""
    aligned = np.asarray(aligned, dtype=np.float64)
    finite = np.all(np.isfinite(aligned), axis=1)
    w = aligned[:, 2] - aligned[:, 0]
    h = aligned[:, 3] - aligned[:, 1]
    with np.errstate(invalid="ignore"):
        diag = np.hypot(w, h)
        min_diag = MIN_DIAGONAL_FRACTION * float(np.hypot(image_width, image_height))
        return finite & (w > 0) & (h > 0) & (diag >= min_diag)


def crop_boxes(bbox: np.ndarray, n_frames: int, image_width: int, image_height: int) -> np.ndarray:
    """Per-frame boxes to give the estimator (rules 4-5). Returns (n_frames, 4)."""
    aligned = align_boxes(bbox, n_frames)
    valid = valid_box_mask(aligned, image_width, image_height)
    if not valid.any():
        return np.tile(np.array([0.0, 0.0, float(image_width), float(image_height)]), (n_frames, 1))
    valid_idx = np.flatnonzero(valid)
    out = aligned.copy()
    for t in np.flatnonzero(~valid):
        dist = np.abs(valid_idx - t)
        nearest = valid_idx[np.argmin(dist)]   # argmin returns the first minimum: the earlier frame
        out[t] = aligned[nearest]
    return out


def video_centre_scale(bbox: np.ndarray, n_frames: int, image_width: int, image_height: int) -> VideoFrame:
    """D-12 centre (median of valid box centres) and scale (median valid-box diagonal)."""
    aligned = align_boxes(bbox, n_frames)
    valid = valid_box_mask(aligned, image_width, image_height)
    if not valid.any():
        return VideoFrame(centre=np.array([image_width / 2.0, image_height / 2.0]),
                          scale=float(np.hypot(image_width, image_height)),
                          used_fallback=True, n_valid=0)
    b = aligned[valid]
    centres = np.column_stack([(b[:, 0] + b[:, 2]) / 2.0, (b[:, 1] + b[:, 3]) / 2.0])
    diags = np.hypot(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])
    return VideoFrame(centre=np.median(centres, axis=0), scale=float(np.median(diags)),
                      used_fallback=False, n_valid=int(valid.sum()))
