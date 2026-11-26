"""Extract Penn Action frames for one partition from the original archive (read-only on the archive).

Usage (main environment, from the repository root):
    .venv\\Scripts\\python.exe -B scripts\\extract_frames.py --partition train

- Verifies the archive SHA-256 first.
- Streams the archive and writes only regular files named Penn_Action/frames/<id>/<NNNNNN>.jpg
  whose video belongs to the requested partition. Members are written by name to a fixed
  destination (no path from the archive is used as a target path), so no member can escape it.
- Test-partition members are skipped unless --partition test (requires --allow-test).
- Refuses to write into a video folder that already contains files.
- Writes a SHA-256 manifest of every extracted frame and a JSON summary under <root>/manifests/.
"""

import argparse
import hashlib
import json
import re
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from penn_pose import integrity, labels  # noqa: E402
from penn_pose.frames import frames_root  # noqa: E402

MEMBER = re.compile(r"^Penn_Action/frames/(\d{4})/([^/]+)$")
FRAME = re.compile(r"^\d{6}\.jpg$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(labels.DEFAULT_DATA_ROOT))
    ap.add_argument("--partition", choices=["train", "test"], required=True)
    ap.add_argument("--allow-test", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    if args.partition == "test" and not args.allow_test:
        sys.exit("Extracting test-partition frames requires --allow-test.")
    ids = set(labels.list_video_ids(args.partition, root, allow_test=args.allow_test))

    archive = root / "Penn_Action.tar.gz"
    t0 = time.time()
    if not integrity.verify_archive(root):
        sys.exit("Archive SHA-256 does not match the recorded value; stopping.")
    t_hash = time.time() - t0

    dest_root = frames_root(root)
    for vid in ids:
        d = dest_root / vid
        if d.exists() and any(d.iterdir()):
            sys.exit(f"Refusing to overwrite existing frames in {d}")

    manifest_lines, unexpected, counts = [], [], {"written": 0, "skipped_other_partition": 0,
                                                    "skipped_non_frame_members": 0}
    t1 = time.time()
    with tarfile.open(archive, "r|gz") as tf:
        for m in tf:
            match = MEMBER.match(m.name)
            if not match:
                counts["skipped_non_frame_members"] += 1
                continue
            vid, name = match.groups()
            if vid not in ids:
                counts["skipped_other_partition"] += 1
                continue
            if not m.isfile() or not FRAME.match(name):
                unexpected.append({"member": m.name, "type": m.type.decode() if isinstance(m.type, bytes) else str(m.type)})
                continue
            data = tf.extractfile(m).read()
            out = dest_root / vid / name
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "xb") as fh:   # 'x': never overwrite
                fh.write(data)
            manifest_lines.append(f"{hashlib.sha256(data).hexdigest()} *{vid}/{name}")
            counts["written"] += 1
    t_extract = time.time() - t1

    manifests = root / "manifests"
    manifests.mkdir(exist_ok=True)
    manifest_path = manifests / f"frames-{args.partition}-sha256.txt"
    manifest_path.write_text("\n".join(sorted(manifest_lines)) + "\n", encoding="ascii", newline="\n")
    summary = {
        "partition": args.partition,
        "archive": str(archive),
        "archive_sha256_verified": integrity.ARCHIVE_SHA256,
        "videos_requested": len(ids),
        "video_folders_written": len({ln.split('*', 1)[1].split('/')[0] for ln in manifest_lines}),
        **counts,
        "unexpected_members_in_partition_folders": unexpected,
        "frame_manifest": str(manifest_path),
        "frame_manifest_sha256": integrity.sha256_file(manifest_path),
        "seconds_archive_hash": round(t_hash, 1),
        "seconds_extract": round(t_extract, 1),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version.split()[0],
    }
    (manifests / f"extraction-{args.partition}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "unexpected_members_in_partition_folders"}, indent=2))
    print("unexpected members:", len(unexpected))


if __name__ == "__main__":
    main()
