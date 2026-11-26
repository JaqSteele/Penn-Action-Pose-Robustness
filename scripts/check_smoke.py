"""Check the five-video RTMPose smoke test (main environment, read-only on all inputs).

Usage:  .venv\\Scripts\\python.exe -B scripts\\check_smoke.py

Expects, under <root>/poses/rtmpose_m/smoke/: run1/ and run2/ (two full GPU runs) and
gpu_sample/ and cpu_sample/ (every 10th frame on each device). Writes smoke/report.json.
Everything here concerns five FITTING-subset videos only; these are preliminary diagnostics,
not the training-partition E1 results.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from penn_pose import e1, frames, gt_preprocess, inference_config as ic, labels, mapping, pose_io, splits  # noqa: E402

ROOT = labels.DEFAULT_DATA_ROOT
SMOKE = ROOT / "poses" / "rtmpose_m" / "smoke"


def load_sample(directory, vid):
    with np.load(directory / f"{vid}.sample.npz", allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def main():
    ids = ic.smoke_test_video_ids(splits.load_split("fitting"))
    fitting = set(splits.load_split("fitting"))
    report = {"videos": ids, "all_in_fitting_subset": all(v in fitting for v in ids), "per_video": {}}
    swapped_cols = list(mapping.PARTNER)
    dets, err_px_all, err_sv_all, swap_sv_all = {}, [], [], []
    for vid in ids:
        a1, meta = pose_io.load_video_output(SMOKE / "run1", vid)       # validates shapes and D-34a
        a2, _ = pose_io.load_video_output(SMOKE / "run2", vid)
        v = labels.load_video(vid, ROOT)
        g = gt_preprocess.prepare_gt(v)
        n = v.n_frames
        W, H = int(a1["image_size"][0]), int(a1["image_size"][1])

        # GT joints eligible for comparison: visible and not corner placeholders (D-07a)
        elig = g.visibility & ~gt_preprocess.placeholder_mask(g.joints_raw)
        est, val = a1["keypoints12"].astype(np.float64), a1["valid12"]
        use = elig & val
        err = np.linalg.norm(est - g.joints_raw, axis=-1)
        err_sw = np.linalg.norm(est[:, swapped_cols] - g.joints_raw, axis=-1)   # if L/R were not corrected
        s_v = g.frame.scale
        err_px_all.append(err[use]); err_sv_all.append(err[use] / s_v); swap_sv_all.append(err_sw[use] / s_v)

        cpu, gpu = load_sample(SMOKE / "cpu_sample", vid), load_sample(SMOKE / "gpu_sample", vid)
        idx = gpu["frame_index"] - 1
        cpu_gpu_px = np.abs(cpu["keypoints17"].astype(np.float64) - gpu["keypoints17"])
        cpu_gpu_score = np.abs(cpu["scores17"].astype(np.float64) - gpu["scores17"])

        box = a1["bbox_used"]
        dets[vid] = e1.video_diagnostics(a1["valid12"])
        report["per_video"][vid] = {
            "action": v.action, "frames_expected": n, "frames_output": int(a1["frame_index"].shape[0]),
            "frame_files_present": all(p.is_file() for p in frames.frame_paths(vid, n, ROOT)),
            "all_coordinates_finite": bool(np.isfinite(a1["keypoints17"]).all()),
            "all_scores_finite": bool(np.isfinite(a1["scores17"]).all()),
            "score_min": float(a1["scores17"].min()), "score_median": float(np.median(a1["scores17"])),
            "score_max": float(a1["scores17"].max()),
            "invalid_keypoints17": int((~a1["valid17"]).sum()),
            "bbox_frames_replaced": int(a1["bbox_replaced"].sum()),
            "bbox_full_image_fallback": meta["bbox_full_image_fallback"],
            "bbox_inside_image": bool(np.all((box[:, 0] >= 0) & (box[:, 1] >= 0) & (box[:, 2] <= W) & (box[:, 3] <= H))),
            "bbox_used_equals_annotation_where_valid": bool(np.array_equal(box[~a1["bbox_replaced"]], a1["bbox_annotation"][~a1["bbox_replaced"]])),
            "run2_bit_identical": all(np.array_equal(a1[k], a2[k]) for k in a1),
            "gpu_sample_equals_run1": bool(np.array_equal(gpu["keypoints17"], a1["keypoints17"][idx]) and np.array_equal(gpu["scores17"], a1["scores17"][idx])),
            "cpu_vs_gpu_frames": int(idx.size),
            "cpu_vs_gpu_max_abs_px": float(cpu_gpu_px.max()),
            "cpu_vs_gpu_frac_keypoints_identical": float(np.mean(np.all(cpu_gpu_px == 0, axis=-1))),
            "cpu_vs_gpu_max_abs_score": float(cpu_gpu_score.max()),
            "eligible_gt_joints_compared": int(use.sum()),
            "median_error_px": float(np.median(err[use])) if use.any() else None,
            "median_error_sv": float(np.median(err[use] / s_v)) if use.any() else None,
            "median_error_sv_if_lr_swapped": float(np.median(err_sw[use] / s_v)) if use.any() else None,
            "s_v_px": s_v,
        }
    px, sv, sw = (np.concatenate(x) for x in (err_px_all, err_sv_all, swap_sv_all))
    by_joint = {}
    report["pooled_mapping_check"] = {
        "eligible_joint_instances": int(sv.size),
        "median_error_px": float(np.median(px)), "median_error_sv": float(np.median(sv)),
        "median_error_sv_if_lr_swapped": float(np.median(sw)),
        "note": "descriptive check of the mapping on five fitting videos; not a pose-error result",
    }
    report["preliminary_e1_five_videos"] = {"label": "PRELIMINARY: five fitting-subset videos only; not the training-partition E1",
                                            "summary": e1.summarise(dets), "per_video": dets}
    report["all_checks_pass"] = all(
        r["frames_output"] == r["frames_expected"] and r["frame_files_present"] and r["all_coordinates_finite"]
        and r["all_scores_finite"] and r["bbox_inside_image"] and r["bbox_used_equals_annotation_where_valid"]
        and r["run2_bit_identical"] and r["gpu_sample_equals_run1"] for r in report["per_video"].values())
    (SMOKE / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
