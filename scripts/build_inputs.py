"""Build and verify the Release 1.0 recogniser inputs for the training partition (Batch D1).

Builds ground-truth ("gt") and estimated ("est") inputs for all 1,258 training-partition videos
and writes them OUTSIDE the repository (default ``C:\\datasets\\penn-action\\inputs\\release1``):

    gt_train.npz, est_train.npz   X (n, 24, 64) float32, y, ids, subset, n_frames (+ est imputation counts)
    class_map.json                the persisted class map (dataset action strings, sorted)
    build_report.json             provenance, integrity checks, imputation summary, statistics
    inputs-sha256.txt             SHA-256 manifest of the four files above

The output directory must not exist: files are written to ``<out>.incomplete`` and renamed only
after every check has passed. Existing outputs are never overwritten. No test-partition file is
read beyond its ``train`` flag.

Usage (main environment, repository root):
    .venv\\Scripts\\python.exe -B scripts\\build_inputs.py            # build, then verify
    .venv\\Scripts\\python.exe -B scripts\\build_inputs.py --verify   # verify an existing build
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import scipy

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from penn_pose import dataset, est_preprocess, gt_preprocess, integrity, labels, splits, temporal  # noqa: E402

DEFAULT_OUT = labels.DEFAULT_DATA_ROOT / "inputs" / "release1"
POSE_MANIFEST = labels.DEFAULT_DATA_ROOT / "manifests" / "poses-rtmpose_m-train-sha256.txt"
POSE_MANIFEST_SHA256 = "505789ce7d800f749f40c2f15c27d6fc75cfd3189598af928995731c55f5e17f"
DATA_FILES = ("gt_train.npz", "est_train.npz", "class_map.json", "build_report.json")
MANIFEST_NAME = "inputs-sha256.txt"
#: Durable repository record of the accepted default build (manifest hash and file hashes).
CACHE_RECORD = REPO_ROOT / "configs" / "inputs" / "release1-cache.json"
SET_ARRAYS = ("X", "y", "ids", "subset", "n_frames")
EST_COUNT_ARRAYS = ("step1_points", "step2_points", "step3_points", "step4_points")
NPZ_KEYS = {"gt_train.npz": set(SET_ARRAYS), "est_train.npz": set(SET_ARRAYS) | set(EST_COUNT_ARRAYS)}

# Measurement E1 (protocol Section 5.5.4): the imputation must reproduce these exactly.
E1_EXPECTED = {
    "fitting": {"step1_points": 20, "step2_points": 0, "step3_points": 0, "step4_points": 0},
    "validation": {"step1_points": 0, "step2_points": 0, "step3_points": 0, "step4_points": 0},
}
E1_VIDEOS = ["0359", "0362", "0373", "0462", "0710", "1516", "1905"]


def git_state() -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    status = run("status", "--porcelain", "--untracked-files=all")
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(status),
            "changes": status.splitlines()}


def source_hashes() -> dict:
    files = sorted((REPO_ROOT / "src" / "penn_pose").glob("*.py")) + [Path(__file__).resolve()]
    return {f.relative_to(REPO_ROOT).as_posix(): integrity.sha256_file(f) for f in files}


def test_hashes() -> dict:
    files = sorted((REPO_ROOT / "tests").glob("*.py"))
    return {f.relative_to(REPO_ROOT).as_posix(): integrity.sha256_file(f) for f in files}


def check_labels(root: Path) -> dict:
    result = integrity.verify_labels(root)
    if not result.ok:
        raise SystemExit(f"Label files differ from the manifest: {result}")
    return {"labels_checked": result.checked, "labels_ok": result.ok}


def check_pose_outputs(pose_dir: Path) -> dict:
    if integrity.sha256_file(POSE_MANIFEST) != POSE_MANIFEST_SHA256:
        raise SystemExit(f"Pose manifest hash differs from the recorded value: {POSE_MANIFEST}")
    expected = integrity.read_manifest(POSE_MANIFEST)
    present = {p.name for p in pose_dir.iterdir() if p.is_file()}
    if present != set(expected):
        raise SystemExit(f"Pose directory contents differ from the manifest "
                         f"(missing {sorted(set(expected) - present)[:5]}, extra {sorted(present - set(expected))[:5]})")
    bad = [n for n, h in sorted(expected.items()) if integrity.sha256_file(pose_dir / n) != h]
    if bad:
        raise SystemExit(f"Pose outputs differ from the manifest: {bad[:5]}")
    return {"files": len(expected), "manifest_sha256": POSE_MANIFEST_SHA256}


def imputation_summary(est: dataset.InputSet) -> dict:
    out = {"training_partition": est_preprocess.summarise_reports(est.reports)}
    for subset in ("fitting", "validation"):
        ids = est.ids[est.subset == subset].tolist()
        out[subset] = est_preprocess.summarise_reports({v: est.reports[v] for v in ids})
    return out


def check_e1(summary: dict) -> None:
    problems = []
    for subset, expected in E1_EXPECTED.items():
        for key, value in expected.items():
            if summary[subset][key] != value:
                problems.append(f"{subset} {key}: {summary[subset][key]} (expected {value})")
    if summary["training_partition"]["step1_videos"] != E1_VIDEOS:
        problems.append(f"step-1 videos {summary['training_partition']['step1_videos']} (expected {E1_VIDEOS})")
    if problems:
        raise SystemExit("Imputation does not reproduce E1; stopping without writing:\n  " + "\n  ".join(problems))


def stats(s: dataset.InputSet) -> dict:
    x = s.X.astype(np.float64)
    return {
        "videos": int(len(s.ids)),
        "by_subset": dict(sorted(Counter(s.subset.tolist()).items())),
        "by_class": {dataset.CLASS_NAMES[k]: int(v) for k, v in sorted(Counter(s.y.tolist()).items())},
        "X_min": float(x.min()), "X_max": float(x.max()),
        "X_mean": float(x.mean()), "X_std": float(x.std()),
        "abs_gt_1_fraction": float(np.mean(np.abs(x) > 1.0)),
        "n_frames_min": int(s.n_frames.min()), "n_frames_median": float(np.median(s.n_frames)),
        "n_frames_max": int(s.n_frames.max()),
    }


def count_arrays(s: dataset.InputSet) -> dict:
    """Per-video imputation-step counts of an estimated set, in identifier order."""
    return {name: np.array([getattr(s.reports[v], name) for v in s.ids.tolist()], dtype=np.int64)
            for name in EST_COUNT_ARRAYS}


def save_set(path: Path, s: dataset.InputSet) -> None:
    arrays = {"X": s.X, "y": s.y, "ids": s.ids, "subset": s.subset, "n_frames": s.n_frames}
    if s.source == "est":
        arrays.update(count_arrays(s))
    with open(path, "xb") as fh:
        np.savez(fh, **arrays)


def write_json(path: Path, obj) -> None:
    with open(path, "x", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=2, sort_keys=False)
        fh.write("\n")


def build_all(root: Path, pose_dir: Path):
    """Build and check both sets in memory. Returns (gt, est, context)."""
    membership = dataset.training_split()
    ids = sorted(membership)
    if ids != labels.list_video_ids("train", root):
        raise SystemExit("The stored split lists do not cover exactly the training partition")
    class_names = dataset.class_names_from_actions(labels.load_training_actions(root).values())
    videos = {v.video_id: v for v in labels.iter_training_videos(root, ids)}
    gt = dataset.build_inputs("gt", ids, root, pose_dir, videos=videos)
    est = dataset.build_inputs("est", ids, root, pose_dir, videos=videos)
    dataset.check_alignment(gt, est)
    return gt, est, {"ids": ids, "class_names": class_names}


def build(out_dir: Path, root: Path, pose_dir: Path) -> None:
    staging = out_dir.with_name(out_dir.name + ".incomplete")
    for p in (out_dir, staging):
        if p.exists():
            raise SystemExit(f"Refusing to overwrite: {p} already exists")
    t0 = time.time()
    label_check = check_labels(root)
    pose_check = check_pose_outputs(pose_dir)
    git = git_state()

    gt, est, ctx = build_all(root, pose_dir)
    summary = imputation_summary(est)
    check_e1(summary)
    gt_report = gt_preprocess.summarise_reports(gt.reports)

    report = {
        "batch": "D1",
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol": "docs/RESEARCH_PROTOCOL.md v0.13",
        "git": git,
        "source_sha256": source_hashes(),
        "test_sha256": test_hashes(),
        "python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
        "platform": platform.platform(),
        "data_root": str(root), "pose_dir": str(pose_dir),
        "integrity": {**label_check, "poses": pose_check},
        "split_sha256": dict(splits.EXPECTED_SHA256),
        "settings": {
            "T": temporal.T_FRAMES, "channels": list(dataset.CHANNEL_NAMES),
            "channel_layout": "interleaved: channel 2j = x, 2j+1 = y of primary joint j (COCO 5-16 order)",
            "dtype": "float32", "order": "ids ascending (all 1,258 training-partition videos)",
            "gt_pipeline": "mapping D-09/D-11 -> D-07b -> D-12 -> D-20",
            "est_pipeline": "COCO 5-16 -> D-34a step 1 + D-34b steps 2-4 (pixels) -> D-12 (GT boxes, locked loader) -> D-20",
        },
        "gt_placeholder_repair": gt_report,
        "est_imputation": summary,
        "statistics": {"gt": stats(gt), "est": stats(est)},
        "elapsed_s": None,
    }

    staging.mkdir(parents=True)
    save_set(staging / "gt_train.npz", gt)
    save_set(staging / "est_train.npz", est)
    write_json(staging / "class_map.json", {
        "class_names": ctx["class_names"],
        "class_index": {n: i for i, n in enumerate(ctx["class_names"])},
        "rule": "the dataset's original action strings, sorted with Python sorted() (code-point order)",
    })
    report["elapsed_s"] = round(time.time() - t0, 1)
    write_json(staging / "build_report.json", report)
    lines = [f"{integrity.sha256_file(staging / n)} *{n}\n" for n in DATA_FILES]
    with open(staging / MANIFEST_NAME, "x", encoding="ascii", newline="\n") as fh:
        fh.writelines(lines)
    staging.rename(out_dir)
    print(f"Built {out_dir} in {report['elapsed_s']} s")
    print(json.dumps({"est_imputation": {k: {kk: summary[k][kk] for kk in ("step1_points", "step2_points", "step3_points", "step4_points", "step1_videos")}
                                         for k in summary},
                      "statistics": report["statistics"]}, indent=2))


def check_cache_record(out_dir: Path, manifest: dict) -> str:
    """Compare the cache manifest with the durable repository record (default build only)."""
    if out_dir.resolve() != DEFAULT_OUT.resolve():
        return "not applicable (non-default output directory)"
    if not CACHE_RECORD.is_file():
        raise SystemExit(f"Repository record not found: {CACHE_RECORD}")
    record = json.loads(CACHE_RECORD.read_text(encoding="utf-8"))
    manifest_sha = integrity.sha256_file(out_dir / MANIFEST_NAME)
    if manifest_sha != record["manifest_sha256"]:
        raise SystemExit(f"{MANIFEST_NAME} SHA-256 {manifest_sha} differs from the repository record "
                         f"{record['manifest_sha256']}")
    if manifest != record["files"]:
        raise SystemExit("Cache manifest entries differ from the repository record")
    return f"matches {CACHE_RECORD.relative_to(REPO_ROOT).as_posix()} ({manifest_sha})"


def verify(out_dir: Path, root: Path, pose_dir: Path) -> None:
    """Hashes, file set, repository record, label and pose manifests, formats, identifiers,
    split membership, class map, and an exact rebuild including the imputation counts."""
    manifest = integrity.read_manifest(out_dir / MANIFEST_NAME)
    present = {p.name for p in out_dir.iterdir()}
    if set(manifest) != set(DATA_FILES) or present != set(DATA_FILES) | {MANIFEST_NAME}:
        raise SystemExit(f"Unexpected cache contents: {sorted(present)}")
    bad = [n for n, h in manifest.items() if integrity.sha256_file(out_dir / n) != h]
    if bad:
        raise SystemExit(f"Hash mismatch: {bad}")
    print(f"Cache manifest hashes: OK; repository record: {check_cache_record(out_dir, manifest)}")
    labels_ok = check_labels(root)
    poses_ok = check_pose_outputs(pose_dir)
    print(f"Label manifest: OK ({labels_ok['labels_checked']} files); "
          f"pose manifest: OK ({poses_ok['files']} files)")

    class_map = json.loads((out_dir / "class_map.json").read_text(encoding="utf-8"))
    if tuple(class_map["class_names"]) != dataset.CLASS_NAMES:
        raise SystemExit("class_map.json differs from dataset.CLASS_NAMES")

    gt_new, est_new, ctx = build_all(root, pose_dir)
    membership = dataset.training_split()
    for name, rebuilt in (("gt_train.npz", gt_new), ("est_train.npz", est_new)):
        with np.load(out_dir / name, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        if set(arrays) != NPZ_KEYS[name]:
            raise SystemExit(f"{name}: arrays {sorted(arrays)}, expected {sorted(NPZ_KEYS[name])}")
        s = dataset.InputSet(source=rebuilt.source, ids=arrays["ids"], X=arrays["X"], y=arrays["y"],
                             subset=arrays["subset"], n_frames=arrays["n_frames"])
        dataset.check_input_set(s)
        if s.ids.tolist() != ctx["ids"] or len(s.ids) != 1258:
            raise SystemExit(f"{name}: identifiers are not the 1,258 training videos in ascending order")
        if [membership[v] for v in s.ids.tolist()] != s.subset.tolist():
            raise SystemExit(f"{name}: subset labels differ from the stored split")
        for key in SET_ARRAYS:
            if not np.array_equal(arrays[key], getattr(rebuilt, key)):
                raise SystemExit(f"{name}: {key} differs from an independent rebuild")
        if rebuilt.source == "est":
            for key, expected in count_arrays(rebuilt).items():
                if arrays[key].dtype != np.int64 or not np.array_equal(arrays[key], expected):
                    raise SystemExit(f"{name}: {key} differs from the independently rebuilt reports")
        print(f"{name}: OK ({s.X.shape}, {s.X.dtype}; "
              f"{int((s.subset == 'fitting').sum())} fitting / {int((s.subset == 'validation').sum())} validation)")
    print(f"Verified {out_dir}: manifest hashes, repository record, label and pose manifests, "
          "contents, class map and exact rebuild (including imputation counts) all match")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--root", type=Path, default=labels.DEFAULT_DATA_ROOT)
    ap.add_argument("--pose-dir", type=Path, default=est_preprocess.DEFAULT_POSE_DIR)
    ap.add_argument("--verify", action="store_true", help="verify an existing build only")
    args = ap.parse_args()
    try:
        args.out_dir.resolve().relative_to(REPO_ROOT)
        raise SystemExit("The output directory must be outside the repository")
    except ValueError:
        pass
    if not args.verify:
        build(args.out_dir, args.root, args.pose_dir)
    verify(args.out_dir, args.root, args.pose_dir)


if __name__ == "__main__":
    main()
