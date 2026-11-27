"""Classification metrics (protocol Section 10.1; D-37) and the majority-class baseline (Section 9.5.3).

D-37: macro-F1 always averages over the fixed label set 0-14 (all 15 classes). A class that is
never predicted scores F1 = 0 (``zero_division = 0``). Accuracy is the ordinary proportion
correct. All values are computed per video, in float64.

Per-class F1 is computed as 2 TP / (2 TP + FP + FN), which equals the harmonic mean of precision
and recall and gives 0 when that denominator is 0. This single implementation is used everywhere
(selection, descriptive results, bootstrap); the tests check it against scikit-learn's
``f1_score(labels=0..14, average="macro", zero_division=0)``.
"""

from __future__ import annotations

import numpy as np

N_CLASSES = 15
LABELS = tuple(range(N_CLASSES))


def check_labels(y: np.ndarray, name: str = "labels") -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 1 or y.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array, got shape {y.shape}")
    if y.dtype.kind not in "iu":
        raise ValueError(f"{name} must be integers, got {y.dtype}")
    if y.min() < 0 or y.max() >= N_CLASSES:
        raise ValueError(f"{name} must lie in 0..{N_CLASSES - 1}")
    return y.astype(np.int64, copy=False)


def _pair(y_true, y_pred):
    y_true = check_labels(y_true, "y_true")
    y_pred = check_labels(y_pred, "y_pred")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred differ in shape: {y_true.shape} vs {y_pred.shape}")
    return y_true, y_pred


def confusion(y_true, y_pred) -> np.ndarray:
    """15 x 15 counts; rows are true classes, columns predicted classes."""
    y_true, y_pred = _pair(y_true, y_pred)
    return np.bincount(y_true * N_CLASSES + y_pred, minlength=N_CLASSES * N_CLASSES).reshape(N_CLASSES, N_CLASSES)


def per_class_f1(y_true, y_pred) -> np.ndarray:
    cm = confusion(y_true, y_pred).astype(np.float64)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    return np.divide(2 * tp, denom, out=np.zeros(N_CLASSES), where=denom > 0)


def macro_f1(y_true, y_pred) -> float:
    return float(np.mean(per_class_f1(y_true, y_pred)))


def accuracy(y_true, y_pred) -> float:
    y_true, y_pred = _pair(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


def summary(y_true, y_pred) -> dict:
    """Macro-F1, accuracy, per-class F1 and the confusion matrix (reporting outputs)."""
    return {
        "macro_f1": macro_f1(y_true, y_pred),
        "accuracy": accuracy(y_true, y_pred),
        "per_class_f1": per_class_f1(y_true, y_pred).tolist(),
        "confusion": confusion(y_true, y_pred).tolist(),
        "n_videos": int(np.asarray(y_true).size),
    }


def majority_class(y_train) -> int:
    """The most frequent class in the run's training labels (ties: the smallest class index)."""
    counts = np.bincount(check_labels(y_train, "y_train"), minlength=N_CLASSES)
    return int(np.argmax(counts))


def majority_predictions(y_train, n: int) -> np.ndarray:
    return np.full(int(n), majority_class(y_train), dtype=np.int64)
