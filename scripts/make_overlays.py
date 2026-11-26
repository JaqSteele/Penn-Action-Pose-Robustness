"""Overlay images for the smoke test (main environment). Images are written outside the repository.

Usage:  .venv\\Scripts\\python.exe -B scripts\\make_overlays.py

For each smoke-test video, the first, middle and last frames are drawn at 2x size with:
- the box given to the estimator (yellow rectangle);
- RTMPose estimates: FILLED dots joined by a skeleton (orange = anatomical left, cyan = right);
  invalid estimates (D-34a) are not drawn;
- Penn Action ground truth: HOLLOW rings in the same side colours, visible joints only;
- a legend stating which marks are estimates and which are ground truth.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from penn_pose import frames, gt_preprocess, inference_config as ic, labels, mapping, pose_io, splits  # noqa: E402

ROOT = labels.DEFAULT_DATA_ROOT
SMOKE = ROOT / "poses" / "rtmpose_m" / "smoke"
LEFT, RIGHT, BOX = (0, 140, 255), (255, 200, 0), (0, 255, 255)   # BGR
BONES = [("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
         ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
         ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
         ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
         ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
         ("right_hip", "right_knee"), ("right_knee", "right_ankle")]
S = 2


def colour(name):
    return LEFT if name.startswith("left") else RIGHT


def draw(img, est, val, gt, gt_vis, box, title):
    img = cv2.resize(img, None, fx=S, fy=S, interpolation=cv2.INTER_LINEAR)
    x1, y1, x2, y2 = (box * S).round().astype(int)
    cv2.rectangle(img, (x1, y1), (x2, y2), BOX, 2)
    idx = {n: i for i, n in enumerate(mapping.PRIMARY_JOINTS)}
    for a, b in BONES:
        i, j = idx[a], idx[b]
        if val[i] and val[j]:
            c = colour(a) if a.split("_")[0] == b.split("_")[0] else (200, 200, 200)
            cv2.line(img, tuple((est[i] * S).round().astype(int)), tuple((est[j] * S).round().astype(int)), c, 2)
    for i, n in enumerate(mapping.PRIMARY_JOINTS):
        if gt_vis[i]:
            cv2.circle(img, tuple((gt[i] * S).round().astype(int)), 9, colour(n), 2)       # GT: hollow ring
        if val[i]:
            cv2.circle(img, tuple((est[i] * S).round().astype(int)), 4, colour(n), -1)     # estimate: filled
    lines = [title, "RTMPose estimate = filled dots + skeleton", "Ground truth (visible) = hollow rings",
             "orange = anatomical left, cyan = right, yellow = input box"]
    for k, t in enumerate(lines):
        cv2.putText(img, t, (8, 22 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, t, (8, 22 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    out = SMOKE / "overlays"
    out.mkdir(parents=True, exist_ok=True)
    for vid in ic.smoke_test_video_ids(splits.load_split("fitting")):
        arr, meta = pose_io.load_video_output(SMOKE / "run1", vid)
        v = labels.load_video(vid, ROOT)
        g = gt_preprocess.prepare_gt(v)
        n = v.n_frames
        for t in sorted({1, (n + 1) // 2, n}):
            img = cv2.imread(str(frames.frame_paths(vid, n, ROOT)[t - 1]))
            vis = g.visibility[t - 1] & ~gt_preprocess.placeholder_mask(g.joints_raw)[t - 1]
            im = draw(img, arr["keypoints12"][t - 1], arr["valid12"][t - 1], g.joints_raw[t - 1], vis,
                      arr["bbox_used"][t - 1], f"video {vid} ({v.action}) frame {t}/{n}")
            path = out / f"{vid}_{t:06d}.png"
            if path.exists():
                raise FileExistsError(path)
            cv2.imwrite(str(path), im)
            print(path)


if __name__ == "__main__":
    main()
