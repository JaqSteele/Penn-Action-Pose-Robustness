"""Temporal preprocessing: resampling to a fixed length (protocol D-20, Section 9.5.1).

Every sequence of N frames is resampled to T = 64 frames by linear interpolation in time,
separately for each coordinate, at the positions

    t_k = k * (N - 1) / (T - 1),   k = 0, ..., T - 1

on the original 0-based frame-index scale. A single-frame sequence is repeated. Resampling is
the last preprocessing step, after normalisation, and is identical in every condition.
"""

from __future__ import annotations

import numpy as np

T_FRAMES = 64


def sample_positions(n_frames: int, length: int = T_FRAMES) -> np.ndarray:
    """The D-20 sample positions t_k on the 0-based frame-index scale."""
    if n_frames < 1:
        raise ValueError(f"n_frames must be at least 1, got {n_frames}")
    if length < 2:
        raise ValueError(f"length must be at least 2, got {length}")
    # Multiply before dividing so that positions which are whole numbers are exact.
    return np.arange(length, dtype=np.float64) * (n_frames - 1) / (length - 1)


def resample_to_length(seq: np.ndarray, length: int = T_FRAMES) -> np.ndarray:
    """Resample ``seq`` (N, ...) along axis 0 to (length, ...) by linear interpolation.

    The first and last output frames equal the first and last input frames exactly. With
    N == length the input is returned unchanged (as float64). With N == 1 the frame is repeated.
    """
    if length < 2:
        raise ValueError(f"length must be at least 2, got {length}")
    seq = np.asarray(seq, dtype=np.float64)
    if seq.ndim < 1 or seq.shape[0] < 1:
        raise ValueError(f"expected a sequence with at least one frame, got shape {seq.shape}")
    if not np.all(np.isfinite(seq)):
        raise ValueError("sequence contains non-finite values")
    n = seq.shape[0]
    if n == 1:
        return np.repeat(seq, length, axis=0)
    t = sample_positions(n, length)
    i0 = np.minimum(np.floor(t).astype(np.int64), n - 2)
    frac = (t - i0).reshape((length,) + (1,) * (seq.ndim - 1))
    return seq[i0] * (1.0 - frac) + seq[i0 + 1] * frac
