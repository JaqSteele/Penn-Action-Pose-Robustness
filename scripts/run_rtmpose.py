"""RTMPose-M inference on Penn Action TRAINING videos (protocol D-03, D-04, D-05, D-34a, v0.12).

Run with the separate RTMPose environment, from the repository root, for example:
    C:\\venvs\\penn-action-rtmpose\\Scripts\\python.exe -B scripts\\run_rtmpose.py --smoke --out-dir <dir>
    C:\\venvs\\penn-action-rtmpose\\Scripts\\python.exe -B scripts\\run_rtmpose.py --videos 0002 0003 --out-dir <dir>
    ... --device cpu --frame-step 25 --sample-only      (CPU cross-check on a sample of frames)

Safeguards:
- config and checkpoint SHA-256 are verified before loading; the model's flip test setting and
  keypoint names are checked against the protocol;
- only training-partition videos are accepted (test videos raise an error);
- one box per call, boxes from the D-05 rules; all 17 raw keypoints and scores are stored, with
  D-34a validity flags and the 12-joint selection in canonical COCO 5-16 order;
- existing outputs are never overwritten.
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from penn_pose import boxes, frames, inference_config as ic, integrity, labels, mapping, pose_io, splits  # noqa: E402


def git_state():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True).stdout.strip())
        return commit, dirty
    except OSError:
        return None, None


def load_model(device):
    import torch
    from mmpose.apis import init_model
    if integrity.sha256_file(ic.CONFIG_PATH) != ic.CONFIG_SHA256:
        sys.exit("Config SHA-256 mismatch; stopping.")
    if integrity.sha256_file(ic.CHECKPOINT_PATH) != ic.CHECKPOINT_SHA256:
        sys.exit("Checkpoint SHA-256 mismatch; stopping.")
    flags = ic.apply_determinism(torch)
    model = init_model(str(ic.CONFIG_PATH), str(ic.CHECKPOINT_PATH), device=device)  # cfg_options=None
    if model.cfg.model.test_cfg.get("flip_test") is not True:
        sys.exit("Model flip_test is not enabled; stopping.")
    names = model.dataset_meta["keypoint_id2name"]
    if tuple(names[i] for i in range(len(names))) != mapping.COCO_KEYPOINTS:
        sys.exit("Model keypoint names do not match COCO-17; stopping.")
    if next(model.parameters()).dtype != torch.float32:
        sys.exit("Model is not FP32; stopping.")
    return model, flags


def infer_video(model, video, frame_indices, used_boxes, root):
    from mmpose.apis import inference_topdown
    paths = frames.frame_paths(video.video_id, video.n_frames, root)
    kps = np.zeros((len(frame_indices), 17, 2), dtype=np.float32)
    scs = np.zeros((len(frame_indices), 17), dtype=np.float32)
    for k, t in enumerate(frame_indices):
        path = paths[t - 1]
        if not path.is_file():
            raise FileNotFoundError(path)
        res = inference_topdown(model, str(path), bboxes=used_boxes[t - 1][None, :], bbox_format="xyxy")
        if len(res) != 1:
            raise RuntimeError(f"{path}: expected one result, got {len(res)}")
        kps[k] = res[0].pred_instances.keypoints[0]
        scs[k] = res[0].pred_instances.keypoint_scores[0]
    return kps, scs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(labels.DEFAULT_DATA_ROOT))
    ap.add_argument("--out-dir", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true", help="the five deterministic fitting-subset videos")
    g.add_argument("--videos", nargs="+")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--frame-step", type=int, default=1)
    ap.add_argument("--sample-only", action="store_true", help="save a frame sample (CPU cross-check)")
    args = ap.parse_args()
    root, out_dir = Path(args.root), Path(args.out_dir)
    if args.frame_step != 1 and not args.sample_only:
        sys.exit("--frame-step requires --sample-only (full outputs must contain every frame).")

    fitting = splits.load_split("fitting")
    validation = set(splits.load_split("validation"))
    ids = ic.smoke_test_video_ids(fitting) if args.smoke else args.videos
    ids = ic.check_inference_ids(ids, root)

    import torch
    import mmcv
    import mmpose
    model, flags = load_model(args.device)
    commit, dirty = git_state()
    env = {"python": sys.version.split()[0], "torch": torch.__version__, "mmpose": mmpose.__version__,
           "mmcv": mmcv.__version__, "numpy": np.__version__, "platform": platform.platform(),
           "device": args.device,
           "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None}
    run = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "videos": ids, "out_dir": str(out_dir),
           "settings": ic.SETTINGS, "torch_flags": flags, "env": env, "git_commit": commit,
           "git_dirty": dirty, "config_sha256": ic.CONFIG_SHA256, "checkpoint_sha256": ic.CHECKPOINT_SHA256,
           "per_video": {}}
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t_run = time.time()
    for vid in ids:
        v = labels.load_video(vid, root)                       # raises for test videos
        W, H = v.image_width, v.image_height
        aligned = boxes.align_boxes(v.bbox, v.n_frames)
        valid_box = boxes.valid_box_mask(aligned, W, H)
        used = boxes.crop_boxes(v.bbox, v.n_frames, W, H)
        idx = list(range(1, v.n_frames + 1, args.frame_step))
        t0 = time.time()
        kps, scs = infer_video(model, v, idx, used, root)
        secs = time.time() - t0
        if args.sample_only:
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{vid}.sample.npz"
            with open(path, "xb") as fh:
                np.savez(fh, frame_index=np.asarray(idx, dtype=np.int32), keypoints17=kps, scores17=scs,
                         bbox_used=used[np.asarray(idx) - 1])
        else:
            valid17 = pose_io.keypoint_validity(kps, scs)
            arrays = {
                "frame_index": np.arange(1, v.n_frames + 1, dtype=np.int32),
                "image_size": np.array([W, H], dtype=np.int32),
                "bbox_annotation": aligned, "bbox_used": used, "bbox_replaced": ~valid_box,
                "keypoints17": kps, "scores17": scs, "valid17": valid17,
                "keypoints12": mapping.coco_to_primary(kps), "scores12": scs[:, list(mapping.COCO_INDEX_FOR_PRIMARY)],
                "valid12": valid17[:, list(mapping.COCO_INDEX_FOR_PRIMARY)],
            }
            meta = {"video_id": vid, "partition": "train",
                    "subset": "validation" if vid in validation else "fitting",
                    "action": v.action, "n_frames": v.n_frames,
                    "image_width": W, "image_height": H, "dimensions_width": v.dims_width,
                    "bbox_frames_replaced": int((~valid_box).sum()),
                    "bbox_full_image_fallback": bool(not valid_box.any()),
                    "n_invalid_keypoints17": int((~valid17).sum()),
                    "seconds": round(secs, 2), "settings": ic.SETTINGS, "torch_flags": flags, "env": env,
                    "config_sha256": ic.CONFIG_SHA256, "checkpoint_sha256": ic.CHECKPOINT_SHA256,
                    "git_commit": commit, "git_dirty": dirty, "primary_joints": list(mapping.PRIMARY_JOINTS),
                    "coco_keypoints": list(mapping.COCO_KEYPOINTS)}
            pose_io.save_video_output(out_dir, vid, arrays, meta)
        run["per_video"][vid] = {"frames": len(idx), "seconds": round(secs, 2),
                                 "ms_per_frame": round(1000 * secs / len(idx), 1)}
        print(f"{vid}: {len(idx)} frames in {secs:.1f} s ({1000 * secs / len(idx):.1f} ms/frame)", flush=True)
    run["seconds_total"] = round(time.time() - t_run, 1)
    if args.device.startswith("cuda"):
        run["peak_gpu_memory_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    run["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"run_{time.strftime('%Y%m%d-%H%M%S')}.json", "x", encoding="utf-8") as fh:
        json.dump(run, fh, indent=2)
    print(json.dumps({k: run[k] for k in ("seconds_total", "torch_flags") if k in run}), run.get("peak_gpu_memory_mib"))


if __name__ == "__main__":
    main()
