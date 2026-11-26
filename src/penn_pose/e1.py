"""Measurement E1: invalid estimated keypoints (protocol D-34b, Section 5.5.4).

E1 is defined on the fitting subset after inference. These functions take, per video, the
``valid12`` mask (frames x 12, canonical COCO 5-16 order) and report:
- invalid rate per primary joint and per video;
- the longest run of consecutive invalid frames per track;
- tracks never valid in their video (these would need D-34b step 2 or 3);
- frames with no valid primary joint;
- videos with no valid primary keypoint anywhere (these would need D-34b step 4).
The functions only measure; they do not apply or change any imputation rule.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from . import mapping


def longest_run(mask_1d: np.ndarray) -> int:
    """Length of the longest run of True values."""
    best = cur = 0
    for v in np.asarray(mask_1d, dtype=bool):
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def video_diagnostics(valid12: np.ndarray) -> dict:
    valid12 = np.asarray(valid12, dtype=bool)
    if valid12.ndim != 2 or valid12.shape[1] != mapping.N_PRIMARY:
        raise ValueError(f"expected (frames, {mapping.N_PRIMARY}), got {valid12.shape}")
    invalid = ~valid12
    never_valid = [mapping.PRIMARY_JOINTS[j] for j in range(mapping.N_PRIMARY) if not valid12[:, j].any()]
    partner_also_never = [j for j in never_valid
                          if not valid12[:, mapping.PARTNER[mapping.PRIMARY_JOINTS.index(j)]].any()]
    return {
        "n_frames": int(valid12.shape[0]),
        "n_invalid": int(invalid.sum()),
        "invalid_rate": float(invalid.mean()) if invalid.size else 0.0,
        "invalid_rate_by_joint": {n: float(invalid[:, j].mean()) for j, n in enumerate(mapping.PRIMARY_JOINTS)},
        "longest_invalid_run_by_joint": {n: longest_run(invalid[:, j]) for j, n in enumerate(mapping.PRIMARY_JOINTS)},
        "never_valid_tracks": never_valid,                       # D-34b step 2 or 3 would be needed
        "never_valid_tracks_partner_also_never": partner_also_never,  # step 3 would be needed
        "frames_without_valid_primary": int((~valid12.any(axis=1)).sum()),
        "needs_step4": bool(not valid12.any()),                  # no valid primary keypoint anywhere
    }


def summarise(per_video: Dict[str, dict]) -> dict:
    """Pool per-video diagnostics."""
    n_frames = sum(d["n_frames"] for d in per_video.values())
    n_cells = n_frames * mapping.N_PRIMARY
    n_invalid = sum(d["n_invalid"] for d in per_video.values())
    by_joint = {}
    for j, name in enumerate(mapping.PRIMARY_JOINTS):
        inv = sum(d["invalid_rate_by_joint"][name] * d["n_frames"] for d in per_video.values())
        by_joint[name] = inv / n_frames if n_frames else 0.0
    needing_23: List[str] = sorted(v for v, d in per_video.items() if d["never_valid_tracks"])
    return {
        "videos": len(per_video),
        "frames": n_frames,
        "invalid_keypoints": n_invalid,
        "invalid_rate": n_invalid / n_cells if n_cells else 0.0,
        "invalid_rate_by_joint": by_joint,
        "longest_invalid_run": max((max(d["longest_invalid_run_by_joint"].values()) for d in per_video.values()), default=0),
        "never_valid_tracks": sum(len(d["never_valid_tracks"]) for d in per_video.values()),
        "frames_without_valid_primary": sum(d["frames_without_valid_primary"] for d in per_video.values()),
        "videos_needing_steps_2_or_3": needing_23,
        "videos_needing_step_3": sorted(v for v, d in per_video.items() if d["never_valid_tracks_partner_also_never"]),
        "videos_needing_step_4": sorted(v for v, d in per_video.items() if d["needs_step4"]),
    }
