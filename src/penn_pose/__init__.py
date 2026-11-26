"""Foundations for the Penn Action pose-estimation robustness study.

Each module implements decisions recorded in ``docs/RESEARCH_PROTOCOL.md`` (v0.11):

- ``labels``        loading and validating Penn Action annotation files; test-partition lock
- ``splits``        the deterministic fitting/validation split (D-02)
- ``mapping``       the 12-joint primary representation and left/right correction (D-09, D-11)
- ``boxes``         ground-truth box validity, fallbacks and per-video centre/scale (D-05, D-12)
- ``gt_preprocess`` corner-placeholder repair for ground-truth poses (D-07b)
- ``normalise``     per-video coordinate normalisation (D-12)
- ``integrity``     annotation-file integrity checks against the SHA-256 manifest

No module modifies any dataset file.
"""

__version__ = "0.1.0"
