"""Ground-truth pose preparation: corner-placeholder repair (D-07b) and the full GT pipeline.

D-07b rules (protocol Section 5.5.3), applied to ground-truth poses only, in the 12-joint
representation, at the original frame rate. No visibility flags are used.
1. Invisible coordinates that are not corner placeholders are retained as released.
2. A corner placeholder is any point with x < 1 and y < 1 (this includes (0, 0)).
3. Partly placeholder track (some frames, not all): linear interpolation in time between the
   joint's own nearest non-placeholder observations, holding the nearest one at the ends.
4. Entirely placeholder track (every frame): rule 3 is applied to every partly placeholder
   track first; then copy the left/right partner's track if it contains no placeholder,
   otherwise use the constant video centre c_v. The choice is made once per track.
Both fallbacks are approximations, not recovered anatomical ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from . import boxes, mapping, normalise
from .labels import PennVideo

PLACEHOLDER_LIMIT = 1.0


@dataclass
class RepairReport:
    partly_tracks: List[int] = field(default_factory=list)        # primary joint indices (rule 3)
    partner_copied: List[int] = field(default_factory=list)       # rule 4, partner track copied
    centre_fallback: List[int] = field(default_factory=list)      # rule 4, constant c_v used
    n_placeholder_points: int = 0


def placeholder_mask(joints: np.ndarray) -> np.ndarray:
    """(frames, 12, 2) -> (frames, 12) True where x < 1 and y < 1."""
    joints = np.asarray(joints, dtype=np.float64)
    return (joints[..., 0] < PLACEHOLDER_LIMIT) & (joints[..., 1] < PLACEHOLDER_LIMIT)


def repair_placeholders(joints: np.ndarray, video_centre: np.ndarray):
    """Apply D-07b to ground-truth primary joints. Returns (repaired copy, RepairReport)."""
    joints = np.asarray(joints, dtype=np.float64)
    if joints.ndim != 3 or joints.shape[1:] != (mapping.N_PRIMARY, 2):
        raise ValueError(f"expected (frames, {mapping.N_PRIMARY}, 2), got {joints.shape}")
    centre = np.asarray(video_centre, dtype=np.float64).reshape(2)
    mask = placeholder_mask(joints)
    out = joints.copy()
    report = RepairReport(n_placeholder_points=int(mask.sum()))
    n_frames = joints.shape[0]
    t = np.arange(n_frames, dtype=np.float64)

    entire = mask.all(axis=0)
    partly = mask.any(axis=0) & ~entire

    # Rule 3: partly placeholder tracks, interpolated from the same joint's own observations.
    for j in np.flatnonzero(partly):
        keep = ~mask[:, j]
        for d in range(2):
            out[mask[:, j], j, d] = np.interp(t[mask[:, j]], t[keep], joints[keep, j, d])
        report.partly_tracks.append(int(j))

    # Rule 4: entirely placeholder tracks. After rule 3, a partner track contains a placeholder
    # only if the partner itself is an entirely placeholder track.
    for j in np.flatnonzero(entire):
        p = mapping.PARTNER[j]
        if not entire[p]:
            out[:, j, :] = out[:, p, :]
            report.partner_copied.append(int(j))
        else:
            out[:, j, :] = centre
            report.centre_fallback.append(int(j))
    return out, report


@dataclass
class PreparedGT:
    video_id: str
    joints_raw: np.ndarray          # (frames, 12, 2) pixels, after the mapping only
    joints_repaired: np.ndarray     # (frames, 12, 2) pixels, after D-07b
    joints_normalised: np.ndarray   # (frames, 12, 2) D-12 normalised
    visibility: np.ndarray          # (frames, 12) bool, for evaluation eligibility only (D-07a)
    frame: boxes.VideoFrame
    report: RepairReport


def prepare_gt(video: PennVideo) -> PreparedGT:
    """Mapping (D-09/D-11) -> placeholder repair (D-07b) -> per-video normalisation (D-12)."""
    joints = mapping.penn_to_primary(video.x, video.y)
    frame = boxes.video_centre_scale(video.bbox, video.n_frames, video.image_width, video.image_height)
    repaired, report = repair_placeholders(joints, frame.centre)
    return PreparedGT(
        video_id=video.video_id,
        joints_raw=joints,
        joints_repaired=repaired,
        joints_normalised=normalise.normalise(repaired, frame.centre, frame.scale),
        visibility=mapping.penn_visibility_to_primary(video.visibility),
        frame=frame,
        report=report,
    )


def summarise_reports(reports: Dict[str, RepairReport]) -> Dict[str, int]:
    return {
        "videos": len(reports),
        "partly_tracks": sum(len(r.partly_tracks) for r in reports.values()),
        "partner_copied": sum(len(r.partner_copied) for r in reports.values()),
        "centre_fallback": sum(len(r.centre_fallback) for r in reports.values()),
        "placeholder_points": sum(r.n_placeholder_points for r in reports.values()),
    }
