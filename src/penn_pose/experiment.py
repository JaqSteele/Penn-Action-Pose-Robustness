"""Release 1.0 experiment orchestration (protocol Sections 4.4, 9.5.3-9.5.8; D-13, D-24, D-33,
D-36 to D-44).

Steps, in the order the protocol requires:
1. ``qualify``: deterministic device qualification (D-36), before Stage A. Two repetitions of a short
   training run on fitting-subset inputs; training-only metrics and state hashes must match exactly.
   No validation or test metric is computed. GPU first, then the CPU with one thread; else stop.
2. ``stage_a_c1``: 8 configurations x 3 seeds on ground-truth fitting inputs, ground-truth validation
   macro-F1 and the model-state hash at epochs 5, 10, ..., 150 (D-22, D-41).
3. ``select_and_record``: the D-22 selection on C1 scores only, written as the selection and freeze
   record. It must be committed to Git before steps 4-6 (D-41).
4. ``stage_a_c2``: deterministic reruns of the selected configuration to E, which must reproduce the
   recorded state hash and ground-truth validation macro-F1 exactly, then the descriptive C2
   evaluation on estimated validation inputs (D-41).
5. ``stage_a_c3``: estimated fitting inputs, E epochs, evaluated only at E (D-43).
6. ``stage_b``: all 1,258 training videos, E epochs, C1 (ground truth) and C3 (estimated) models,
   no evaluation; model weights saved outside Git (D-33, D-42).

Every step uses the qualified device unchanged (D-36). No step reads the test partition: inputs
come only from the verified D1 training-partition cache.
"""

from __future__ import annotations

import csv
import datetime
import io
import json
import os
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from . import dataset, determinism, integrity, metrics, runs, splits
from .labels import DEFAULT_DATA_ROOT
from .selection import RELEASE1, Plan, Score, Selection, select
from .training import BATCH_SIZE, TrainConfig, make_evaluator, predict, train_model

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "recognition" / "release1.json"
SELECTION_RECORD = REPO_ROOT / "configs" / "recognition" / "release1-selection.json"
#: Compact, committed records (device, selection, stage summaries). Bulky outputs never go here.
RECORDS_DIR = REPO_ROOT / "configs" / "recognition"
TABLES_DIR = REPO_ROOT / "results" / "tables"
CACHE_DIR = DEFAULT_DATA_ROOT / "inputs" / "release1"
CACHE_RECORD = REPO_ROOT / "configs" / "inputs" / "release1-cache.json"
PROTOCOL_VERSION = "v0.14"

DEVICE_CANDIDATES = ("cuda:0", "cpu")
QUALIFICATION_EPOCHS = 5
QUALIFICATION_CONFIG = TrainConfig(0.001, 0.0001, 0.3)
QUALIFICATION_SEED = 0


class QualificationError(RuntimeError):
    pass


class ReproductionError(RuntimeError):
    pass


# --------------------------------------------------------------------------------------------
# Configuration


def config_dict() -> dict:
    """The committed experiment configuration, as the code defines it."""
    return {
        "description": "Release 1.0 recogniser experiment configuration (protocol v0.14, Sections 9.5 and 9.5.8).",
        "protocol": PROTOCOL_VERSION,
        "plan": RELEASE1.as_dict(),
        "batch_size": BATCH_SIZE,
        "drop_last": False,
        "optimizer": "AdamW; defaults apart from learning rate and weight decay; weight decay on all parameters",
        "loss": "class-weighted cross-entropy, w_c = N / (K N_c) from the run's training labels, reduction='mean'",
        "device_policy": {"candidates": list(DEVICE_CANDIDATES), "cpu_threads": 1},
        "qualification": {"epochs": QUALIFICATION_EPOCHS, "config": QUALIFICATION_CONFIG.as_dict(),
                          "seed": QUALIFICATION_SEED, "inputs": "ground-truth fitting subset; training-only metrics"},
        "bootstrap": {"n_resamples": 10000, "seed": None,
                      "seed_status": "open (D-45): fixed and committed before any test-partition output is opened"},
        "experiment_root": str(runs.EXPERIMENT_ROOT),
        "inputs_record": "configs/inputs/release1-cache.json",
    }


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load the committed configuration and check it matches the code exactly."""
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = config_dict()
    if cfg != expected:
        diffs = sorted(k for k in set(cfg) | set(expected) if cfg.get(k) != expected.get(k))
        raise ValueError(f"{path} differs from the configuration defined in code: {diffs}")
    return cfg


# --------------------------------------------------------------------------------------------
# Inputs (verified D1 cache; training partition only)


@dataclass(frozen=True)
class Inputs:
    source: str
    X: np.ndarray
    y: np.ndarray
    ids: np.ndarray
    subset: np.ndarray

    def part(self, which: str):
        if which == "all":
            mask = np.ones(len(self.ids), dtype=bool)
        elif which in ("fitting", "validation"):
            mask = self.subset == which
        else:
            raise ValueError(f"unknown subset {which!r}")
        return np.ascontiguousarray(self.X[mask]), self.y[mask].copy(), self.ids[mask].copy()


def load_inputs(source: str, cache_dir: Path = CACHE_DIR, record_path: Path = CACHE_RECORD) -> Inputs:
    """Load ``gt`` or ``est`` inputs after checking the cache against its repository record."""
    if source not in dataset.SOURCES:
        raise ValueError(f"source must be one of {dataset.SOURCES}")
    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    manifest_path = Path(cache_dir) / "inputs-sha256.txt"
    if integrity.sha256_file(manifest_path) != record["manifest_sha256"]:
        raise ValueError("input cache manifest differs from its repository record")
    manifest = integrity.read_manifest(manifest_path)
    if manifest != record["files"]:
        raise ValueError("input cache manifest entries differ from the repository record")
    name = f"{source}_train.npz"
    if integrity.sha256_file(Path(cache_dir) / name) != manifest[name]:
        raise ValueError(f"{name} differs from the manifest")
    with np.load(Path(cache_dir) / name, allow_pickle=False) as data:
        X, y, ids, subset, n_frames = data["X"], data["y"], data["ids"], data["subset"], data["n_frames"]
    dataset.check_input_set(dataset.InputSet(source=source, ids=ids, X=X, y=y, subset=subset, n_frames=n_frames))
    membership = dataset.training_split()
    if ids.tolist() != sorted(membership) or [membership[v] for v in ids.tolist()] != subset.tolist():
        raise ValueError(f"{name}: identifiers or subsets differ from the stored split")
    return Inputs(source=source, X=X, y=y, ids=ids, subset=subset)


def check_paired(gt: Inputs, est: Inputs) -> None:
    if gt.source != "gt" or est.source != "est":
        raise ValueError("expected ground-truth and estimated inputs")
    for name in ("ids", "y", "subset"):
        if not np.array_equal(getattr(gt, name), getattr(est, name)):
            raise ValueError(f"ground-truth and estimated inputs differ in {name}")


# --------------------------------------------------------------------------------------------
# Provenance and the device lock


def check_external(path: Path, repo_root: Path = REPO_ROOT) -> Path:
    """Experiment outputs must be written outside the Git repository (D-42)."""
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(Path(repo_root).resolve())
    except ValueError:
        return resolved
    raise runs.OutputExistsError(f"experiment outputs must be outside the repository, got {resolved}")


class ProvenanceError(RuntimeError):
    """A run or step does not match the expected provenance, or provenance is missing."""


#: Keys every production provenance must contain.
REQUIRED_PROVENANCE = ("protocol", "environment", "git", "code", "device_record_sha256",
                       "split_sha256", "inputs_record_sha256")
#: Provenance fields that must match exactly (the environment is compared on LOCKED_FIELDS only,
#: and code identity separately: same commit within Stage A, descendant commit with an identical
#: protected tree afterwards).
LOCKED_PROVENANCE = ("protocol", "device_record_sha256", "split_sha256", "inputs_record_sha256")


def provenance(device: str, device_record: Optional[Path] = None, repo_root: Path = REPO_ROOT) -> dict:
    """Provenance of the current step. ``code`` identifies the committed implementation and is only
    available when the protected code/configuration scope is committed and clean."""
    return {
        "protocol": PROTOCOL_VERSION,
        "environment": determinism.environment_record(device),
        "git": runs.git_state(repo_root),
        "code": runs.code_identity(repo_root),
        "device_record_sha256": integrity.sha256_file(device_record) if device_record is not None else None,
        "split_sha256": dict(splits.EXPECTED_SHA256),
        "inputs_record_sha256": integrity.sha256_file(CACHE_RECORD) if CACHE_RECORD.is_file() else None,
    }


def validate_provenance(prov: Optional[dict], repo_root: Path) -> None:
    """Production steps need complete provenance that describes the repository as it is now: the
    protected scope committed and clean, HEAD and its protected tree as recorded, and a device record."""
    if not isinstance(prov, dict):
        raise ProvenanceError("provenance is required for every experiment step")
    missing = [k for k in REQUIRED_PROVENANCE if k not in prov]
    if missing:
        raise ProvenanceError(f"provenance is missing {missing}")
    problems = provenance_field_problems(prov)
    if problems:
        raise ProvenanceError(f"provenance is invalid: {problems}")
    code = prov["code"]
    live = runs.code_identity(repo_root)                              # always the canonical scope
    if code != live:
        raise ProvenanceError(f"provenance code identity {code} does not describe the repository now: {live}")


def _is_sha256(value) -> bool:
    """A SHA-256 written as 64 hexadecimal characters (either case)."""
    return isinstance(value, str) and len(value) == 64 and all(c in string.hexdigits for c in value)


def provenance_field_problems(prov: dict) -> List[str]:
    """Completeness and format of the provenance fields that later checks rely on."""
    problems = determinism.locked_environment_problems(prov.get("environment"))
    for key in ("device_record_sha256", "inputs_record_sha256"):
        if not _is_sha256(prov.get(key)):
            problems.append(f"{key} is not a SHA-256 hash: {prov.get(key)!r}")
    split = prov.get("split_sha256")
    if not isinstance(split, dict) or set(split) != set(splits.EXPECTED_SHA256):
        problems.append(f"split_sha256 must have exactly the keys {sorted(splits.EXPECTED_SHA256)}")
    else:
        problems += [f"split_sha256.{k} is not a SHA-256 hash: {v!r}" for k, v in split.items() if not _is_sha256(v)]
    code = prov.get("code")
    if not isinstance(code, dict):
        problems.append("code identity is missing")
    else:
        if code.get("protected_paths") != list(runs.PROTECTED_PATHS):
            problems.append(f"code.protected_paths must be exactly {list(runs.PROTECTED_PATHS)}, "
                            f"got {code.get('protected_paths')!r}")
        if not _is_sha256(code.get("protected_tree_sha256")):
            problems.append("code.protected_tree_sha256 is not a SHA-256 hash")
    return problems


def check_provenance(recorded: Optional[dict], expected: dict, repo_root: Path, *, same_commit: bool) -> None:
    """Compare a recorded provenance (a run or the selection record) with the expected, current one.

    ``same_commit=True`` (within Stage A and at selection): the implementation commit must be the
    same. ``same_commit=False`` (C2, C3, Stage B): the recorded commit must be an ancestor of the
    current one and the protected tree must be identical, so later commits may add records but not
    change protected code/configuration. Nothing recorded is ever rewritten.
    """
    if not isinstance(recorded, dict):
        raise ProvenanceError("recorded provenance is missing")
    diffs = [k for k in LOCKED_PROVENANCE if recorded.get(k) != expected.get(k)]
    renv, eenv = recorded.get("environment") or {}, expected.get("environment") or {}
    diffs += [f"environment.{k}" for k in determinism.LOCKED_FIELDS if renv.get(k) != eenv.get(k)]
    rcode, ecode = recorded.get("code") or {}, expected["code"]
    canonical = list(runs.PROTECTED_PATHS)
    if ecode.get("protected_paths") != canonical:
        raise ProvenanceError(f"expected provenance must use the canonical protected paths {canonical}")
    if rcode.get("protected_paths") != canonical:
        diffs.append("code.protected_paths")
    if rcode.get("protected_tree_sha256") != ecode.get("protected_tree_sha256"):
        diffs.append("code.protected_tree_sha256")
    rcommit = rcode.get("commit")
    if not rcommit:
        diffs.append("code.commit missing")
    elif same_commit:
        if rcommit != ecode["commit"]:
            diffs.append("code.commit")
    else:
        if not runs.is_ancestor(repo_root, rcommit, ecode["commit"]):
            diffs.append("code.commit is not an ancestor of the current commit")
        else:
            try:
                tree = runs.protected_tree_sha256(repo_root, rcommit)          # canonical paths
            except runs.CodeIdentityError as exc:
                diffs.append(f"code.commit: {exc}")
            else:
                if tree != rcode.get("protected_tree_sha256"):
                    diffs.append("code.protected_tree_sha256 does not match the recorded commit")
    if diffs:
        raise ProvenanceError(f"provenance differs from the expected provenance: {diffs}")


def prepare_device(device_record_path: Path, repo_root: Path) -> str:
    """D-36: load the committed qualification record, configure its device and check that the
    current environment is the qualified one. Returns the device."""
    runs.require_committed(device_record_path, repo_root)
    record = runs.read_json(device_record_path)
    device = record["qualified_device"]
    determinism.configure(device)
    determinism.assert_same_environment(record["environment"], determinism.environment_record(device))
    return device


# --------------------------------------------------------------------------------------------
# 1. Device qualification


def qualification_inputs(gt: Inputs):
    """The qualification uses ground-truth fitting-subset inputs only (D-36)."""
    if gt.source != "gt":
        raise ValueError("the qualification uses ground-truth inputs")
    X_fit, y_fit, _ = gt.part("fitting")
    return X_fit, y_fit


def qualification_repetition(X_fit: np.ndarray, y_fit: np.ndarray, device: str,
                             config: TrainConfig = QUALIFICATION_CONFIG, seed: int = QUALIFICATION_SEED,
                             epochs: int = QUALIFICATION_EPOCHS) -> dict:
    """One repetition: training-only metrics and state hashes. No validation or test data."""
    if epochs < 5:
        raise ValueError("the qualification run needs at least five epochs (D-36)")
    if determinism.is_cuda(device):
        import torch
        if not torch.cuda.is_available():
            raise determinism.DeterminismError("CUDA is not available")
    determinism.configure(device)
    result = train_model(X_fit, y_fit, config, seed, epochs, device,
                         checkpoint_epochs=tuple(range(5, epochs + 1, 5)), evaluate=None)
    return {
        "device": device, "config": config.as_dict(), "seed": seed, "epochs": epochs,
        "initial_state_sha256": result.initial_state_sha256,
        "final_state_sha256": result.final_state_sha256,
        "checkpoint_state_sha256": {str(e): c["state_sha256"] for e, c in result.checkpoints.items()},
        "training_metrics": result.epochs,
        "environment": determinism.environment_record(device),
        "process_id": os.getpid(),                       # evidence of a separate process per repetition
    }


def compare_repetitions(a: dict, b: dict) -> List[str]:
    """Exact comparison of two repetitions (hashes, training-only metrics, locked environment)."""
    diffs = [k for k in ("device", "config", "seed", "epochs", "initial_state_sha256", "final_state_sha256",
                         "checkpoint_state_sha256", "training_metrics") if a.get(k) != b.get(k)]
    try:
        determinism.assert_same_environment(a["environment"], b["environment"])
    except determinism.DeterminismError as exc:
        diffs.append(f"environment: {exc}")
    return diffs


def qualify(run_repetition: Callable[[str], dict], candidates: Sequence[str] = DEVICE_CANDIDATES) -> dict:
    """Try each device in order; the first whose two repetitions are bit-identical is qualified.
    If none qualifies, raise ``QualificationError`` (work stops; D-36)."""
    attempts = []
    for device in candidates:
        try:
            first, second = run_repetition(device), run_repetition(device)
        except Exception as exc:                      # the attempt fails; record it and fall back
            attempts.append({"device": device, "qualified": False, "error": f"{type(exc).__name__}: {exc}"})
            continue
        diffs = compare_repetitions(first, second)
        attempts.append({"device": device, "qualified": not diffs, "differences": diffs,
                         "repetitions": [first, second]})
        if not diffs:
            return {"qualified_device": device, "environment": first["environment"], "attempts": attempts}
    raise QualificationError(f"no device passed the deterministic qualification: "
                             f"{[{k: a[k] for k in a if k != 'repetitions'} for a in attempts]}")


def device_record_path(records_dir: Path = RECORDS_DIR) -> Path:
    return Path(records_dir) / "release1-device.json"


def compact_device_record(result: dict, manifest_sha: str) -> dict:
    """The committed device record: what is needed to reproduce and verify the device, and nothing
    else. Error messages (which can contain local paths) stay in the external qualification run."""
    qualified = next(a for a in result["attempts"] if a.get("qualified"))
    rep = qualified["repetitions"][0]
    return {
        "record": "Release 1.0 deterministic device qualification (protocol D-36)",
        "protocol": PROTOCOL_VERSION,
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "qualified_device": result["qualified_device"],
        "environment": result["environment"],
        "qualification": {"config": rep["config"], "seed": rep["seed"], "epochs": rep["epochs"],
                          "inputs": "ground-truth fitting subset; training-only metrics",
                          "initial_state_sha256": rep["initial_state_sha256"],
                          "checkpoint_state_sha256": rep["checkpoint_state_sha256"],
                          "final_state_sha256": rep["final_state_sha256"]},
        "attempts": [{"device": a["device"], "qualified": bool(a.get("qualified")),
                      "failure": (a["error"].split(":", 1)[0] if "error" in a
                                  else ("non-identical repetitions" if a.get("differences") else None)),
                      "differing_fields": [d.split(":", 1)[0] for d in a.get("differences", [])]}
                     for a in result["attempts"]],
        "run_manifest_sha256": manifest_sha,
    }


def record_qualification(result: dict, out_root: Path, records_dir: Path) -> str:
    """Write the qualification run (outside Git) and its compact repository record."""
    record_path = device_record_path(records_dir)
    out_root = check_external(out_root)
    _ensure_absent(record_path)
    w = runs.RunWriter(Path(out_root) / "qualification" / "device")
    w.write_json("qualification.json", result)
    manifest_sha = w.finalise()
    return runs.write_record(record_path, compact_device_record(result, manifest_sha))


# --------------------------------------------------------------------------------------------
# 2. Stage A, C1


def run_name(config: TrainConfig, seed: int) -> str:
    return f"{config.name}_seed{seed}"


def _checkpoint_summary(result) -> List[dict]:
    return [{k: v for k, v in c.items() if k != "y_pred"} for _, c in sorted(result.checkpoints.items())]


def stage_a_c1(gt: Inputs, device: str, out_root: Path, plan: Plan = RELEASE1, *,
               prov: dict, repo_root: Path) -> List[str]:
    """Train every (configuration, seed); resumable: finished runs are verified (manifest and
    provenance) and skipped. A finished run from another commit, device record or environment
    stops the stage before anything new is trained; it is never rewritten."""
    plan.validate()
    stage_dir = check_external(out_root) / "stage_a_c1"
    validate_provenance(prov, repo_root)
    done = runs.stage_runs(stage_dir)
    for name in sorted(done):                                         # resumed runs: current provenance
        check_provenance(runs.read_json(stage_dir / name / "run.json").get("provenance"), prov,
                         repo_root, same_commit=True)
    X_fit, y_fit, _ = gt.part("fitting")
    X_val, y_val, ids_val = gt.part("validation")
    evaluator = make_evaluator(X_val, y_val, device)
    names = []
    for config in plan.grid:
        for seed in plan.seeds:
            name = run_name(config, seed)
            names.append(name)
            if name in done:
                continue
            result = train_model(X_fit, y_fit, config, seed, plan.max_epochs, device,
                                 checkpoint_epochs=plan.eval_epochs, evaluate=evaluator)
            validate_provenance(prov, repo_root)                      # nothing changed during training
            w = runs.RunWriter(stage_dir / name)
            w.write_json("run.json", {
                "stage": "A", "condition": "C1", "train_inputs": "gt fitting", "eval_inputs": "gt validation",
                "config": config.as_dict(), "seed": seed, "plan": plan.as_dict(),
                "initial_state_sha256": result.initial_state_sha256,
                "final_state_sha256": result.final_state_sha256,
                "training_metrics": result.epochs, "checkpoints": _checkpoint_summary(result),
                "provenance": prov,
            })
            w.write_npz("predictions.npz", ids=ids_val, y_true=y_val, epochs=np.array(plan.eval_epochs),
                        y_pred=np.stack([result.checkpoints[e]["y_pred"] for e in plan.eval_epochs]))
            w.finalise()
    return names


def collect_scores(stage_dir: Path, plan: Plan, expected_prov: dict, repo_root: Path):
    """Scores and state hashes from the finished C1 runs. All must be present, verified, and match
    the expected (current) provenance with the same implementation commit."""
    done = runs.stage_runs(stage_dir)
    expected = {run_name(c, s): (c, s) for c in plan.grid for s in plan.seeds}
    if set(done) != set(expected):
        raise ValueError(f"Stage A C1 runs incomplete or unexpected: missing {sorted(set(expected) - set(done))}, "
                         f"extra {sorted(set(done) - set(expected))}")
    scores, hashes, origins = [], {}, set()
    for name, (config, seed) in expected.items():
        rec = runs.read_json(Path(stage_dir) / name / "run.json")
        if rec["config"] != config.as_dict() or rec["seed"] != seed or rec["plan"] != plan.as_dict():
            raise ValueError(f"{name}: run record does not match the plan")
        check_provenance(rec.get("provenance"), expected_prov, repo_root, same_commit=True)
        prov = rec.get("provenance") or {}
        env = prov.get("environment") or {}
        origins.add((json.dumps({k: env.get(k) for k in determinism.LOCKED_FIELDS}, sort_keys=True),
                     (prov.get("git") or {}).get("commit"), prov.get("device_record_sha256")))
        if len(origins) > 1:
            raise ValueError("Stage A C1 runs come from different devices, software, code commits or device records")
        for c in rec["checkpoints"]:
            scores.append(Score(config, seed, int(c["epoch"]), float(c["macro_f1"])))
            hashes[(config, seed, int(c["epoch"]))] = c["state_sha256"]
    return scores, hashes, done


# --------------------------------------------------------------------------------------------
# 3. Selection and freeze record


def selection_record(selection: Selection, hashes: dict, run_manifests: dict, stage_manifest_sha: str,
                     y_all: np.ndarray, plan: Plan, prov: Optional[dict]) -> dict:
    s = selection.selected
    seeds = list(plan.seeds)
    majority = metrics.majority_class(y_all)
    return {
        "record": "Release 1.0 Stage A selection and Stage B freeze record (protocol Sections 9.5.4, 9.5.7; D-41)",
        "protocol": PROTOCOL_VERSION,
        "selection_inputs": "C1 ground-truth validation macro-F1 only (D-13)",
        "selected": {
            "config": s.config.as_dict(), "epoch": s.epoch, "mean_macro_f1": s.mean_macro_f1,
            "per_seed_macro_f1": {str(k): v for k, v in zip(seeds, s.per_seed)},
            "per_seed_state_sha256": {str(k): hashes[(s.config, k, s.epoch)] for k in seeds},
        },
        "ties": [{"config": c.config.as_dict(), "epoch": c.epoch} for c in selection.tied],
        "candidates": [{"config": c.config.as_dict(), "epoch": c.epoch, "mean_macro_f1": c.mean_macro_f1,
                        "per_seed_macro_f1": list(c.per_seed)} for c in selection.candidates],
        "stage_a_c1": {"stage_manifest_sha256": stage_manifest_sha, "run_manifest_sha256": run_manifests},
        "stage_b": {
            "config": s.config.as_dict(), "epochs": s.epoch, "seeds": seeds, "batch_size": BATCH_SIZE,
            "training_videos": int(len(y_all)), "class_weights": "from all training videos (D-24)",
            "majority_class": {"index": majority, "name": dataset.CLASS_NAMES[majority]},
        },
        "plan": plan.as_dict(),
        "stage_a_implementation": dict(prov["code"]) if prov else None,
        "provenance": prov,
    }


def write_selection_table(selection: Selection, path: Path) -> str:
    """Aggregate table (mean C1 validation macro-F1 per configuration and epoch); never overwrites."""
    buf = io.StringIO()
    wr = csv.writer(buf, lineterminator="\n")
    wr.writerow(["learning_rate", "weight_decay", "dropout", "epoch", "mean_macro_f1", "selected"])
    for c in selection.candidates:
        wr.writerow([c.config.learning_rate, c.config.weight_decay, c.config.dropout, c.epoch,
                     repr(c.mean_macro_f1), int(c == selection.selected)])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8", newline="\n") as fh:
        fh.write(buf.getvalue())
    return integrity.sha256_file(path)


def _ensure_absent(*paths: Optional[Path]) -> None:
    """Refuse before writing anything if any final output already exists."""
    existing = [str(p) for p in paths if p is not None and Path(p).exists()]
    if existing:
        raise runs.OutputExistsError(f"Refusing to overwrite: {existing}")


def select_and_record(out_root: Path, y_all: np.ndarray, record_path: Path, table_path: Optional[Path],
                      plan: Plan = RELEASE1, *, prov: dict, repo_root: Path) -> dict:
    stage_dir = check_external(out_root) / "stage_a_c1"
    validate_provenance(prov, repo_root)
    _ensure_absent(record_path, table_path, stage_dir / "stage-sha256.txt")
    scores, hashes, done = collect_scores(stage_dir, plan, prov, repo_root)
    selection = select(scores, plan)
    stage_sha = runs.write_stage_manifest(stage_dir, list(done))
    record = selection_record(selection, hashes, done, stage_sha, y_all, plan, prov)
    runs.write_record(record_path, record)
    if table_path is not None:
        write_selection_table(selection, table_path)
    return record


def _selected(record: dict):
    sel = record["selected"]
    c = sel["config"]
    return TrainConfig(c["learning_rate"], c["weight_decay"], c["dropout"]), int(sel["epoch"])


def require_only_epoch(checkpoints: dict, E: int) -> None:
    """D-43: C3 Stage A is evaluated only at the selected epoch E (an explicit check, independent of
    Python's ``assert``)."""
    if sorted(checkpoints) != [E]:
        raise RuntimeError(f"C3 Stage A must be evaluated only at E = {E} (D-43); got epochs {sorted(checkpoints)}")


def _empty_stage(stage_dir: Path) -> None:
    if Path(stage_dir).exists() and any(Path(stage_dir).iterdir()):
        raise runs.OutputExistsError(f"{stage_dir} already contains outputs; refusing to overwrite")


def stage_record_path(records_dir: Path, name: str) -> Path:
    return Path(records_dir) / f"release1-{name}.json"


def _stage_record(records_dir: Path, name: str, obj: dict) -> str:
    return runs.write_record(stage_record_path(records_dir, name), obj)


# --------------------------------------------------------------------------------------------
# 4. Stage A, C2 (deterministic reruns, then descriptive evaluation)


def stage_a_c2(gt: Inputs, est: Inputs, device: str, out_root: Path, record_path: Path, repo_root: Path,
               records_dir: Path, plan: Plan = RELEASE1, *, prov: dict) -> dict:
    commit = runs.require_committed(record_path, repo_root)          # D-41 commit order
    check_paired(gt, est)
    validate_provenance(prov, repo_root)
    record = runs.read_json(record_path)
    check_provenance(record.get("provenance"), prov, repo_root, same_commit=False)   # code unchanged since Stage A
    config, E = _selected(record)
    stage_dir = check_external(out_root) / "stage_a_c2"
    _empty_stage(stage_dir)
    _ensure_absent(stage_record_path(records_dir, "stage-a-c2"))
    X_fit, y_fit, _ = gt.part("fitting")
    Xg_val, y_val, ids_val = gt.part("validation")
    Xe_val, ye_val, _ = est.part("validation")
    gt_eval = make_evaluator(Xg_val, y_val, device)

    reruns = {}
    for seed in plan.seeds:                                           # verify every rerun first
        result = train_model(X_fit, y_fit, config, seed, E, device, checkpoint_epochs=(E,), evaluate=gt_eval)
        got_hash = result.checkpoints[E]["state_sha256"]
        got_f1 = result.checkpoints[E]["macro_f1"]
        want_hash = record["selected"]["per_seed_state_sha256"][str(seed)]
        want_f1 = record["selected"]["per_seed_macro_f1"][str(seed)]
        if got_hash != want_hash or got_f1 != want_f1:
            raise ReproductionError(f"seed {seed}: rerun to epoch {E} did not reproduce Stage A "
                                    f"(state {got_hash} vs {want_hash}; macro-F1 {got_f1!r} vs {want_f1!r})")
        reruns[seed] = result

    summary = {}
    for seed, result in reruns.items():                               # only now: estimated validation
        y_pred = predict(result.model, Xe_val, device)
        desc = metrics.summary(ye_val, y_pred)
        validate_provenance(prov, repo_root)
        w = runs.RunWriter(stage_dir / f"{config.name}_E{E}_seed{seed}")
        w.write_json("run.json", {
            "stage": "A", "condition": "C2 (descriptive only)", "model": "C1 Stage A rerun",
            "train_inputs": "gt fitting", "eval_inputs": "est validation",
            "config": config.as_dict(), "epoch": E, "seed": seed,
            "rerun_check": {"state_sha256": result.checkpoints[E]["state_sha256"],
                            "gt_validation_macro_f1": result.checkpoints[E]["macro_f1"], "reproduced": True},
            "c2_validation": desc, "selection_record_commit": commit, "provenance": prov,
        })
        w.write_npz("predictions.npz", ids=ids_val, y_true=ye_val, y_pred=y_pred,
                    y_pred_gt_rerun=result.checkpoints[E]["y_pred"])
        w.finalise()
        summary[str(seed)] = {"macro_f1": desc["macro_f1"], "accuracy": desc["accuracy"]}
    stage_sha = runs.write_stage_manifest(stage_dir, [f"{config.name}_E{E}_seed{s}" for s in plan.seeds])
    out = {"stage": "A", "condition": "C2", "descriptive_only": True, "config": config.as_dict(), "epoch": E,
           "per_seed": summary, "stage_manifest_sha256": stage_sha, "selection_record_commit": commit}
    _stage_record(records_dir, "stage-a-c2", out)
    return out


# --------------------------------------------------------------------------------------------
# 5. Stage A, C3 (evaluated only at E)


def stage_a_c3(est: Inputs, device: str, out_root: Path, record_path: Path, repo_root: Path,
               records_dir: Path, plan: Plan = RELEASE1, *, prov: dict) -> dict:
    commit = runs.require_committed(record_path, repo_root)
    validate_provenance(prov, repo_root)
    record = runs.read_json(record_path)
    check_provenance(record.get("provenance"), prov, repo_root, same_commit=False)   # code unchanged since Stage A
    config, E = _selected(record)
    stage_dir = check_external(out_root) / "stage_a_c3"
    _empty_stage(stage_dir)
    _ensure_absent(stage_record_path(records_dir, "stage-a-c3"))
    X_fit, y_fit, _ = est.part("fitting")
    X_val, y_val, ids_val = est.part("validation")
    evaluator = make_evaluator(X_val, y_val, device)
    summary = {}
    for seed in plan.seeds:
        result = train_model(X_fit, y_fit, config, seed, E, device, checkpoint_epochs=(E,), evaluate=evaluator)
        require_only_epoch(result.checkpoints, E)                      # D-43: only at E
        y_pred = result.checkpoints[E]["y_pred"]
        desc = metrics.summary(y_val, y_pred)
        validate_provenance(prov, repo_root)
        w = runs.RunWriter(stage_dir / f"{config.name}_E{E}_seed{seed}")
        w.write_json("run.json", {
            "stage": "A", "condition": "C3 (descriptive only)", "train_inputs": "est fitting",
            "eval_inputs": "est validation", "config": config.as_dict(), "epoch": E, "seed": seed,
            "initial_state_sha256": result.initial_state_sha256,
            "state_sha256": result.checkpoints[E]["state_sha256"], "training_metrics": result.epochs,
            "c3_validation": desc, "selection_record_commit": commit, "provenance": prov,
        })
        w.write_npz("predictions.npz", ids=ids_val, y_true=y_val, y_pred=y_pred)
        w.finalise()
        summary[str(seed)] = {"macro_f1": desc["macro_f1"], "accuracy": desc["accuracy"]}
    stage_sha = runs.write_stage_manifest(stage_dir, [f"{config.name}_E{E}_seed{s}" for s in plan.seeds])
    out = {"stage": "A", "condition": "C3", "descriptive_only": True, "config": config.as_dict(), "epoch": E,
           "per_seed": summary, "stage_manifest_sha256": stage_sha, "selection_record_commit": commit}
    _stage_record(records_dir, "stage-a-c3", out)
    return out


# --------------------------------------------------------------------------------------------
# 6. Stage B (all 1,258 videos; no evaluation)


def stage_b(gt: Inputs, est: Inputs, device: str, out_root: Path, record_path: Path, repo_root: Path,
            records_dir: Path, plan: Plan = RELEASE1, *, prov: dict) -> dict:
    commit = runs.require_committed(record_path, repo_root)
    check_paired(gt, est)
    validate_provenance(prov, repo_root)
    record = runs.read_json(record_path)
    check_provenance(record.get("provenance"), prov, repo_root, same_commit=False)   # code unchanged since Stage A
    config, E = _selected(record)
    out = {"stage": "B", "config": config.as_dict(), "epochs": E, "selection_record_commit": commit,
           "majority_class": record["stage_b"]["majority_class"], "conditions": {}}
    out_root = check_external(out_root)
    for condition in ("C1", "C3"):                                   # check everything before training
        _empty_stage(Path(out_root) / f"stage_b_{condition.lower()}")
    _ensure_absent(stage_record_path(records_dir, "stage-b"))
    for condition, inputs in (("C1", gt), ("C3", est)):
        stage_dir = Path(out_root) / f"stage_b_{condition.lower()}"
        X_all, y_all, _ = inputs.part("all")
        per_seed = {}
        for seed in plan.seeds:
            result = train_model(X_all, y_all, config, seed, E, device, checkpoint_epochs=(E,), evaluate=None)
            validate_provenance(prov, repo_root)
            w = runs.RunWriter(stage_dir / f"{config.name}_E{E}_seed{seed}")
            w.write_json("run.json", {
                "stage": "B", "condition": condition, "train_inputs": f"{inputs.source} all training videos",
                "training_videos": int(len(y_all)), "config": config.as_dict(), "epochs": E, "seed": seed,
                "early_stopping": False, "evaluation": None,
                "initial_state_sha256": result.initial_state_sha256,
                "final_state_sha256": result.final_state_sha256, "training_metrics": result.epochs,
                "selection_record_commit": commit, "provenance": prov,
            })
            w.write_state_dict("model.pt", result.model.state_dict())
            w.finalise()
            per_seed[str(seed)] = result.final_state_sha256
        out["conditions"][condition] = {
            "final_state_sha256": per_seed,
            "stage_manifest_sha256": runs.write_stage_manifest(
                stage_dir, [f"{config.name}_E{E}_seed{s}" for s in plan.seeds]),
        }
    _stage_record(records_dir, "stage-b", out)
    return out


# --------------------------------------------------------------------------------------------
# Loading the frozen Stage B models (for the separate, later final evaluation; nothing here
# reads or evaluates test data)


def load_stage_b_models(out_root: Path, stage_b_record: dict, device: str) -> Dict[str, Dict[int, "object"]]:
    """Return {"C1": {seed: model}, "C3": {seed: model}} in inference mode, after checking every run
    manifest, each stage manifest against the Stage B record, and each model's state hash."""
    import torch
    from .model import TemporalCNN

    c = stage_b_record["config"]
    config = TrainConfig(c["learning_rate"], c["weight_decay"], c["dropout"])
    E = int(stage_b_record["epochs"])
    models: Dict[str, Dict[int, object]] = {}
    for condition, rec in stage_b_record["conditions"].items():
        stage_dir = Path(out_root) / f"stage_b_{condition.lower()}"
        if integrity.sha256_file(stage_dir / "stage-sha256.txt") != rec["stage_manifest_sha256"]:
            raise ValueError(f"{stage_dir}: stage manifest differs from the Stage B record")
        done = runs.stage_runs(stage_dir)
        listed = integrity.read_manifest(stage_dir / "stage-sha256.txt")
        if {k.split("/")[0]: v for k, v in listed.items()} != done:
            raise ValueError(f"{stage_dir}: run manifests differ from the stage manifest")
        models[condition] = {}
        for seed_text, want in rec["final_state_sha256"].items():
            run_dir = stage_dir / f"{config.name}_E{E}_seed{seed_text}"
            state = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
            model = TemporalCNN(config.dropout)
            model.load_state_dict(state)
            if determinism.state_hash(model) != want:
                raise ValueError(f"{run_dir}: model state differs from the Stage B record")
            models[condition][int(seed_text)] = model.to(device).eval()
    return models
