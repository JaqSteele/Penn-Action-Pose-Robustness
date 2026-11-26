"""Per-video coordinate normalisation (protocol D-12, Section 9.4).

normalised = (joint - c_v) / s_v, with one centre and one scale per video (from ``boxes``),
the same scale for x and y, and no per-frame recentring, rescaling, rotation or mirroring.
"""

from __future__ import annotations

import numpy as np


def _check(centre, scale):
    centre = np.asarray(centre, dtype=np.float64).reshape(2)
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"scale must be a positive finite number, got {scale}")
    if not np.all(np.isfinite(centre)):
        raise ValueError("centre must be finite")
    return centre, scale


def normalise(points: np.ndarray, centre, scale) -> np.ndarray:
    """Apply (p - c_v) / s_v to an array whose last axis is (x, y)."""
    centre, scale = _check(centre, scale)
    points = np.asarray(points, dtype=np.float64)
    if points.shape[-1] != 2:
        raise ValueError("last axis must hold (x, y)")
    return (points - centre) / scale


def denormalise(points: np.ndarray, centre, scale) -> np.ndarray:
    """Inverse of ``normalise``: p = u * s_v + c_v."""
    centre, scale = _check(centre, scale)
    points = np.asarray(points, dtype=np.float64)
    if points.shape[-1] != 2:
        raise ValueError("last axis must hold (x, y)")
    return points * scale + centre
