"""D-01 frame-level verification for one partition (main environment, read-only).

Usage:  .venv\\Scripts\\python.exe -B scripts\\verify_frames.py --partition train

Writes <root>/manifests/d01-frames-<partition>.json (full results) and prints a summary.
Nothing is repaired or modified.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from penn_pose import frames, integrity, labels  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(labels.DEFAULT_DATA_ROOT))
    ap.add_argument("--partition", choices=["train", "test"], required=True)
    ap.add_argument("--allow-test", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    if args.partition == "test" and not args.allow_test:
        sys.exit("Verifying test-partition frames requires --allow-test.")
    t0 = time.time()
    ids = labels.list_video_ids(args.partition, root, allow_test=args.allow_test)
    results, total_jpegs, width_corrected = [], 0, []
    for vid in ids:
        v = labels.load_video(vid, root, allow_test=args.allow_test)
        r = frames.verify_video_frames(v, root)
        total_jpegs += r.n_found
        if v.dims_width != v.image_width:
            width_corrected.append({"video_id": vid, "dimensions_width": v.dims_width,
                                    "jpeg_size": r.image_size})
        results.append({"video_id": vid, "ok": r.ok, "n_expected": r.n_expected, "n_found": r.n_found,
                        "image_size": r.image_size, "annotation_size": [v.image_width, v.image_height],
                        "issues": r.issues})
    folders = frames.frames_root(root)
    present = sorted(p.name for p in folders.iterdir()) if folders.is_dir() else []
    unexpected_folders = sorted(set(present) - set(labels.list_video_ids("train", root)) - (set(ids)))
    failing = [r for r in results if not r["ok"]]
    report = {
        "partition": args.partition,
        "videos_checked": len(results),
        "videos_passing": len(results) - len(failing),
        "videos_failing": len(failing),
        "failures": failing,
        "total_jpegs": total_jpegs,
        "width_corrected_videos": width_corrected,
        "unexpected_folders_in_frames_dir": unexpected_folders,
        "frame_manifest_sha256": integrity.sha256_file(root / "manifests" / f"frames-{args.partition}-sha256.txt"),
        "label_manifest_ok": integrity.verify_labels(root).ok,
        "seconds": round(time.time() - t0, 1),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "per_video": results,
    }
    out = root / "manifests" / f"d01-frames-{args.partition}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {k: v for k, v in report.items() if k not in ("per_video",)}
    summary["width_corrected_videos"] = len(width_corrected)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
