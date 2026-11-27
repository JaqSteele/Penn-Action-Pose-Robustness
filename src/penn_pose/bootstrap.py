"""Paired, class-stratified video-level bootstrap (protocol Sections 10.3, 10.7; D-26, D-27, D-28,
D-37, D-45).

- Resampling: within each class, videos are drawn with replacement, keeping each class's count.
  The same resampled indices are used for every condition and seed within a comparison.
- B = 10,000 resamples (D-26). Percentile intervals (D-27), computed with ``numpy.percentile``
  (default linear interpolation).
- Comparison A versus B with three seeds (Section 10.7): point estimate = mean over seeds of the
  per-seed macro-F1 differences on the full set; 95% interval = 2.5th and 97.5th percentiles of the
  seed-averaged difference over resamples; Bonferroni 97.5% interval = 1.25th and 98.75th.
  Seed variation is reported separately: the per-seed differences, their range and standard
  deviation, and each seed's own bootstrap interval.
- D-45: the bootstrap seed has **no default**. Its value is still open and must be fixed and
  committed before any test-partition output is opened; it is always passed explicitly.

Nothing here reads data; it operates on label and prediction arrays given by the caller.
"""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

from . import metrics

N_RESAMPLES = 10_000
LEVELS = {"ci95": (2.5, 97.5), "ci97_5": (1.25, 98.75)}


def _check_seed(seed) -> int:
    if seed is None or isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("a bootstrap seed must be given explicitly as a non-negative integer "
                         "(D-45: its value is fixed before any test-partition output is opened)")
    return int(seed)


def resample_indices(y_true: np.ndarray, n_resamples: int, seed) -> np.ndarray:
    """(n_resamples, n) indices; each row resamples every class with replacement, keeping its count."""
    y = metrics.check_labels(y_true, "y_true")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, (int, np.integer)) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    rng = np.random.default_rng(_check_seed(seed))
    groups = [np.flatnonzero(y == c) for c in range(metrics.N_CLASSES) if np.any(y == c)]
    out = np.empty((n_resamples, y.size), dtype=np.int64)
    for r in range(n_resamples):
        out[r] = np.concatenate([g[rng.integers(0, g.size, size=g.size)] for g in groups])
    return out


def _check_preds(y_true: np.ndarray, preds: Mapping[int, np.ndarray]) -> Dict[int, np.ndarray]:
    if not preds:
        raise ValueError("no predictions given")
    return {int(s): metrics._pair(y_true, p)[1] for s, p in sorted(preds.items())}


def _percentiles(values: np.ndarray) -> dict:
    return {name: [float(v) for v in np.percentile(values, q)] for name, q in LEVELS.items()}


def paired_comparison(y_true: np.ndarray, preds_a: Mapping[int, np.ndarray], preds_b: Mapping[int, np.ndarray],
                      seed, n_resamples: int = N_RESAMPLES) -> dict:
    """Macro-F1 difference A - B with the pre-specified three-seed procedure (Section 10.7)."""
    y = metrics.check_labels(y_true, "y_true")
    a, b = _check_preds(y, preds_a), _check_preds(y, preds_b)
    if list(a) != list(b):
        raise ValueError(f"conditions must share the same seeds, got {list(a)} and {list(b)}")
    idx = resample_indices(y, n_resamples, seed)
    per_seed_full = {s: metrics.macro_f1(y, a[s]) - metrics.macro_f1(y, b[s]) for s in a}
    per_seed_boot = {s: np.array([metrics.macro_f1(y[r], a[s][r]) - metrics.macro_f1(y[r], b[s][r]) for r in idx])
                     for s in a}
    averaged = np.mean(np.stack([per_seed_boot[s] for s in a]), axis=0)
    diffs = np.array([per_seed_full[s] for s in a])
    return {
        "metric": "macro_f1", "n_resamples": n_resamples, "bootstrap_seed": int(seed), "seeds": list(a),
        "point_estimate": float(np.mean(diffs)),
        **_percentiles(averaged),
        "seed_variation": {
            "per_seed_difference": {str(s): float(per_seed_full[s]) for s in a},
            "range": float(diffs.max() - diffs.min()),
            "sd": float(np.std(diffs, ddof=1)) if diffs.size > 1 else None,
            "per_seed_ci95": {str(s): _percentiles(per_seed_boot[s])["ci95"] for s in a},
        },
    }


def condition_summary(y_true: np.ndarray, preds: Mapping[int, np.ndarray], seed,
                      n_resamples: int = N_RESAMPLES) -> dict:
    """Per-condition macro-F1 and accuracy, per seed and as the seed mean, with percentile intervals."""
    y = metrics.check_labels(y_true, "y_true")
    p = _check_preds(y, preds)
    idx = resample_indices(y, n_resamples, seed)
    out = {"n_resamples": n_resamples, "bootstrap_seed": int(seed), "seeds": list(p)}
    for name, fn in (("macro_f1", metrics.macro_f1), ("accuracy", metrics.accuracy)):
        boot = {s: np.array([fn(y[r], p[s][r]) for r in idx]) for s in p}
        out[name] = {
            "per_seed": {str(s): {"value": fn(y, p[s]), **_percentiles(boot[s])} for s in p},
            "mean": {"value": float(np.mean([fn(y, p[s]) for s in p])),
                     **_percentiles(np.mean(np.stack([boot[s] for s in p]), axis=0))},
        }
    return out
