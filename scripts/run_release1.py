"""Release 1.0 recogniser experiments (protocol v0.14, Sections 4.4, 9.5 and 9.5.8).

Commands, to be run in this order and only with explicit approval for each:

    qualify      D-36 device qualification (GPU first, then CPU with one thread). Each repetition
                 runs in its own fresh Python process. Writes the qualification run (outside Git)
                 and configs/recognition/release1-device.json (commit it before Stage A).
    stage-a-c1   24 C1 runs (8 configurations x 3 seeds), ground-truth validation every 5 epochs.
    select       D-22 selection; writes configs/recognition/release1-selection.json and an aggregate
                 table. Commit the record before any of the next three commands (D-41).
    stage-a-c2   deterministic reruns to E (must reproduce exactly), then descriptive C2 evaluation.
    stage-a-c3   C3 Stage A, evaluated only at E (D-43).
    stage-b      final C1 and C3 models on all 1,258 training videos; no evaluation.

Every command except ``qualify`` requires the committed device record and the same device and
software, and the protected experimental code/configuration (runs.PROTECTED_PATHS) committed and
clean: Stage A and selection at one implementation commit, later steps at a descendant commit with an
identical protected tree. Every command that trains needs ``--confirm``. No command reads the test
partition.

Usage (main environment, repository root):
    .venv\\Scripts\\python.exe -B scripts\\run_release1.py <command> --confirm
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from penn_pose import determinism  # noqa: E402,F401  (sets CUBLAS_WORKSPACE_CONFIG before torch)
from penn_pose import experiment as ex  # noqa: E402
from penn_pose import runs  # noqa: E402

DEVICE_RECORD = ex.device_record_path()
TABLE = ex.TABLES_DIR / "release1_stage_a_c1_validation.csv"


def repetition_command(device: str) -> list:
    """A new Python interpreter running this script's internal ``qualify-repetition`` command."""
    return [sys.executable, "-B", str(Path(__file__).resolve()), "qualify-repetition", "--device", device, "--confirm"]


def _repetition_in_subprocess(device: str, command=repetition_command) -> dict:
    """Run one qualification repetition in a fresh process (never reusing an earlier one)."""
    proc = subprocess.run(command(device), cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"repetition on {device} failed: {proc.stderr.strip()[-2000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["qualify", "qualify-repetition", "stage-a-c1", "select",
                                        "stage-a-c2", "stage-a-c3", "stage-b"])
    ap.add_argument("--device", help="internal: device for qualify-repetition")
    ap.add_argument("--confirm", action="store_true", help="required for every command that trains")
    args = ap.parse_args()
    ex.load_config()
    ex.check_external(runs.EXPERIMENT_ROOT)
    if args.command != "select" and not args.confirm:
        raise SystemExit(f"'{args.command}' trains models; rerun with --confirm once it has been approved")

    if args.command == "qualify-repetition":
        X_fit, y_fit = ex.qualification_inputs(ex.load_inputs("gt"))   # fitting subset only
        print(json.dumps(ex.qualification_repetition(X_fit, y_fit, args.device)))
        return

    if args.command == "qualify":
        result = ex.qualify(_repetition_in_subprocess)
        ex.record_qualification(result, runs.EXPERIMENT_ROOT, ex.RECORDS_DIR)
        print(f"Qualified device: {result['qualified_device']}. Commit {DEVICE_RECORD.relative_to(REPO_ROOT)} "
              "before Stage A.")
        return

    device = ex.prepare_device(DEVICE_RECORD, REPO_ROOT)
    prov = ex.provenance(device, DEVICE_RECORD)
    gt = ex.load_inputs("gt")
    if args.command == "stage-a-c1":
        ex.stage_a_c1(gt, device, runs.EXPERIMENT_ROOT, prov=prov, repo_root=REPO_ROOT)
    elif args.command == "select":
        record = ex.select_and_record(runs.EXPERIMENT_ROOT, gt.y, ex.SELECTION_RECORD, TABLE, prov=prov,
                                      repo_root=REPO_ROOT)
        print(f"Selected {record['selected']['config']} at E = {record['selected']['epoch']}. Commit "
              f"{ex.SELECTION_RECORD.relative_to(REPO_ROOT)} before any C2, C3 or Stage B run.")
    elif args.command == "stage-a-c2":
        ex.stage_a_c2(gt, ex.load_inputs("est"), device, runs.EXPERIMENT_ROOT, ex.SELECTION_RECORD, REPO_ROOT,
                      ex.RECORDS_DIR, prov=prov)
    elif args.command == "stage-a-c3":
        ex.stage_a_c3(ex.load_inputs("est"), device, runs.EXPERIMENT_ROOT, ex.SELECTION_RECORD, REPO_ROOT,
                      ex.RECORDS_DIR, prov=prov)
    elif args.command == "stage-b":
        ex.stage_b(gt, ex.load_inputs("est"), device, runs.EXPERIMENT_ROOT, ex.SELECTION_RECORD, REPO_ROOT,
                   ex.RECORDS_DIR, prov=prov)


if __name__ == "__main__":
    main()
