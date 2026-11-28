"""Batch E (final evaluation) provenance gate and test-input safeguards (protocol v0.15,
Section 9.5.9). Nothing here reads test-partition data, runs inference or evaluates a model.

Stage A and Stage B ran under protocol v0.14 at one protected implementation (protected tree
``TRAINING_PROTECTED_TREE`` at ``TRAINING_COMMIT``). Recording D-45 to D-47 and implementing the
final evaluation must change protected files, so the Stage A/B code-identity gates (unchanged in
``experiment``) cannot apply to the final evaluation. ``check_code`` replaces them for Batch E:

1. the protected scope is committed and clean (``runs.code_identity``), and its identity is recorded;
2. ``STAGE_B_COMMIT`` and ``TRAINING_COMMIT`` are ancestors of HEAD, and the protected tree at
   ``TRAINING_COMMIT`` equals the recorded Stage A tree;
3. the frozen records (device, selection, C2, C3, Stage B) are byte-identical to ``STAGE_B_COMMIT``;
4. every protected file is identical to ``TRAINING_COMMIT`` except ``ALLOWED_PATHS``;
5. in ``EXPERIMENT_PATH`` only the top-level definitions in ``EXPERIMENT_MUTABLE`` may change: every
   other top-level statement must be unchanged (compared as Python syntax trees), and no new
   top-level statement may be added;
6. the ``CONTROLLED_DOCUMENTS`` (unprotected documentation that states the bootstrap settings) are
   committed and unmodified, and their SHA-256 is recorded.

``ALLOWED_PATHS`` names the files that may differ. Test-input and final-evaluation modules are added
to it only by a reviewed, committed change, which itself changes the recorded code identity.

``verify_stage_b`` checks all six Stage B runs through ``experiment.load_stage_b_models``.
``check_test_pairing`` and the ``checked_*`` wrappers enforce sorted, identical test identifiers and
identical labels before any metric or bootstrap.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from . import bootstrap, experiment, metrics, runs

#: The approved training implementation: Stage A C1, selection, C2, C3 and Stage B all ran with this
#: protected tree (Stage A at this commit; later steps at descendants with an identical tree).
TRAINING_COMMIT = "5f2c9daa58ab6705cd654d2f9d663db1bbb159f8"
TRAINING_PROTECTED_TREE = "5af6d7dbf493ba4f5e8462b37be9d66f35f23c7fc159699201733241fa37af02"
#: The commit that recorded the Stage B models.
STAGE_B_COMMIT = "0f69d2112141dc09733214f37249781386877471"

#: Committed records that must stay byte-identical to their state at ``STAGE_B_COMMIT``.
FROZEN_RECORDS = (
    "configs/recognition/release1-device.json",
    "configs/recognition/release1-selection.json",
    "configs/recognition/release1-stage-a-c2.json",
    "configs/recognition/release1-stage-a-c3.json",
    "configs/recognition/release1-stage-b.json",
)

#: The only files that may differ from ``TRAINING_COMMIT`` (protocol, configuration, bootstrap, the
#: experiment module within ``EXPERIMENT_MUTABLE``, this module, and the recogniser documentation).
#: ``docs/RECOGNITION.md`` is outside ``runs.PROTECTED_PATHS``, so the change rule never inspects it;
#: it is listed for completeness and controlled through ``CONTROLLED_DOCUMENTS`` below.
ALLOWED_PATHS = (
    "docs/RESEARCH_PROTOCOL.md",
    "configs/recognition/release1.json",
    "src/penn_pose/bootstrap.py",
    "src/penn_pose/experiment.py",
    "src/penn_pose/batch_e.py",
    "docs/RECOGNITION.md",
)

#: Unprotected documents that must be committed and unmodified when the gate runs; their SHA-256 is
#: recorded in the Batch E identity. They do not change the protected code identity.
CONTROLLED_DOCUMENTS = ("docs/RECOGNITION.md",)

EXPERIMENT_PATH = "src/penn_pose/experiment.py"
#: Top-level definitions of ``EXPERIMENT_PATH`` that may change: the module docstring, the protocol
#: version, the configuration definition and the Stage B model loader. Nothing may be added.
EXPERIMENT_MUTABLE = ("__doc__", "PROTOCOL_VERSION", "config_dict", "load_stage_b_models")


class BatchEGateError(RuntimeError):
    """The repository does not satisfy the Batch E provenance rule."""


# --------------------------------------------------------------------------------------------
# Git helpers (read-only)


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=Path(repo_root), capture_output=True)


def _blob(repo_root: Path, commit: str, rel: str) -> Optional[bytes]:
    """File content at ``commit``, or None if the path does not exist there."""
    proc = _git(repo_root, "show", f"{commit}:{rel}")
    return proc.stdout if proc.returncode == 0 else None


def _tree_files(repo_root: Path, commit: str, paths: Sequence[str]) -> Dict[str, str]:
    """{path: blob id} for every file under ``paths`` at ``commit``."""
    proc = _git(repo_root, "ls-tree", "-r", "--full-tree", commit, "--", *paths)
    if proc.returncode != 0:
        raise BatchEGateError(f"git ls-tree failed for {commit}: {proc.stderr.decode(errors='replace').strip()}")
    out = {}
    for line in proc.stdout.decode("utf-8").splitlines():
        meta, path = line.split("\t", 1)
        out[path] = meta.split()[2]
    return out


def _worktree_files(repo_root: Path, paths: Sequence[str]) -> Dict[str, str]:
    """{path: blob id} for tracked and untracked (not ignored) files under ``paths`` in the working
    tree, hashed as Git would store them. Read-only: nothing is written to the object database."""
    tracked = _git(repo_root, "ls-files", "--cached", "--others", "--exclude-standard", "--", *paths)
    files = sorted(set(tracked.stdout.decode("utf-8").splitlines()))
    present = [f for f in files if (Path(repo_root) / f).is_file()]
    if not present:
        return {}
    proc = subprocess.run(["git", "hash-object", "--stdin-paths"], cwd=Path(repo_root), capture_output=True,
                          input="\n".join(present).encode("utf-8"))
    if proc.returncode != 0:
        raise BatchEGateError(f"git hash-object failed: {proc.stderr.decode(errors='replace').strip()}")
    return dict(zip(present, proc.stdout.decode("utf-8").split()))


# --------------------------------------------------------------------------------------------
# Experiment-module comparison


def _top_level(source: bytes) -> Dict[str, str]:
    """{key: syntax-tree dump} for every top-level statement. Definitions and assignments are keyed
    by name; the module docstring by ``__doc__``; anything else by its own dump."""
    tree = ast.parse(source)
    out: Dict[str, str] = {}
    for i, node in enumerate(tree.body):
        if i == 0 and isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant) \
                and isinstance(node.value.value, str):
            key = "__doc__"
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            key = node.name
        elif isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets):
            key = ",".join(t.id for t in node.targets)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            key = node.target.id
        else:
            key = "stmt:" + ast.dump(node)
        if key in out:
            raise BatchEGateError(f"duplicate top-level definition {key!r}")
        out[key] = ast.dump(node)
    return out


def experiment_module_problems(old: bytes, new: bytes, mutable: Iterable[str] = EXPERIMENT_MUTABLE) -> List[str]:
    """Top-level statements that changed, disappeared or were added outside ``mutable``."""
    mutable = set(mutable)
    a, b = _top_level(old), _top_level(new)
    problems = [f"changed or removed: {k}" for k in a if k not in mutable and b.get(k) != a[k]]
    problems += [f"added: {k}" for k in b if k not in a and k not in mutable]
    return problems


# --------------------------------------------------------------------------------------------
# The gate


def changed_protected_paths(repo_root: Path, base: str, target: Optional[str]) -> List[str]:
    """Protected files that differ between ``base`` and ``target`` (a commit, or the working tree
    when ``target`` is None): changed, added or removed."""
    old = _tree_files(repo_root, base, runs.PROTECTED_PATHS)
    new = _tree_files(repo_root, target, runs.PROTECTED_PATHS) if target else _worktree_files(repo_root, runs.PROTECTED_PATHS)
    return sorted(p for p in set(old) | set(new) if old.get(p) != new.get(p))


def code_change_problems(repo_root: Path, target: Optional[str], *, training_commit: str = TRAINING_COMMIT,
                         allowed_paths: Sequence[str] = ALLOWED_PATHS, experiment_path: str = EXPERIMENT_PATH,
                         experiment_mutable: Iterable[str] = EXPERIMENT_MUTABLE) -> List[str]:
    """Rule 4 and 5 violations for ``target`` (a commit, or the working tree when None)."""
    problems = [f"not allowed to differ from the training implementation: {p}"
                for p in changed_protected_paths(repo_root, training_commit, target) if p not in allowed_paths]
    old = _blob(repo_root, training_commit, experiment_path)
    if target:
        new = _blob(repo_root, target, experiment_path)
    else:
        f = Path(repo_root) / experiment_path
        new = f.read_bytes() if f.is_file() else None
    if old is not None:
        if new is None:
            problems.append(f"{experiment_path} was removed")
        else:
            problems += [f"{experiment_path}: {p}" for p in experiment_module_problems(old, new, experiment_mutable)]
    return problems


def check_code(repo_root: Path = experiment.REPO_ROOT, *, training_commit: str = TRAINING_COMMIT,
               training_tree: str = TRAINING_PROTECTED_TREE, stage_b_commit: str = STAGE_B_COMMIT,
               frozen_records: Sequence[str] = FROZEN_RECORDS, allowed_paths: Sequence[str] = ALLOWED_PATHS,
               experiment_path: str = EXPERIMENT_PATH,
               experiment_mutable: Iterable[str] = EXPERIMENT_MUTABLE,
               selection_record: Optional[str] = "configs/recognition/release1-selection.json",
               controlled_documents: Sequence[str] = CONTROLLED_DOCUMENTS) -> dict:
    """Enforce the Batch E provenance rule (module docstring) and return the Batch E code identity.
    Raises ``BatchEGateError`` (or ``runs.CodeIdentityError``) on any violation."""
    repo_root = Path(repo_root)
    identity = runs.code_identity(repo_root)                          # committed, clean scope
    head = identity["commit"]
    problems = []
    for name, commit in (("training", training_commit), ("Stage B", stage_b_commit)):
        if not runs.is_ancestor(repo_root, commit, head):
            problems.append(f"the {name} commit {commit} is not an ancestor of HEAD {head}")
    if problems:
        raise BatchEGateError("; ".join(problems))
    if runs.protected_tree_sha256(repo_root, training_commit) != training_tree:
        raise BatchEGateError("the protected tree at the training commit differs from the recorded Stage A tree")
    if selection_record is not None:
        sel = _blob(repo_root, head, selection_record)
        impl = json.loads(sel).get("stage_a_implementation") if sel is not None else None
        if not impl or impl.get("commit") != training_commit or impl.get("protected_tree_sha256") != training_tree:
            raise BatchEGateError("the selection record's Stage A implementation differs from the training commit")
    for rel in frozen_records:
        now, then = _blob(repo_root, head, rel), _blob(repo_root, stage_b_commit, rel)
        if then is None or now != then:
            problems.append(f"frozen record changed or missing since {stage_b_commit[:7]}: {rel}")
        else:
            runs.require_committed(repo_root / rel, repo_root)         # working copy equals HEAD
    problems += code_change_problems(repo_root, head, training_commit=training_commit, allowed_paths=allowed_paths,
                                     experiment_path=experiment_path, experiment_mutable=experiment_mutable)
    for rel in controlled_documents:
        if rel not in allowed_paths:
            problems.append(f"controlled document is not in the allowed paths: {rel}")
        runs.require_committed(repo_root / rel, repo_root)             # committed and unmodified
    if problems:
        raise BatchEGateError("Batch E provenance rule violated: " + "; ".join(problems))
    return {
        "rule": "protocol v0.15 Section 9.5.9",
        "batch_e_code": identity,
        "training_commit": training_commit,
        "training_protected_tree_sha256": training_tree,
        "stage_b_commit": stage_b_commit,
        "changed_protected_paths": changed_protected_paths(repo_root, training_commit, head),
        "frozen_records_sha256": {rel: hashlib.sha256(_blob(repo_root, head, rel)).hexdigest()
                                  for rel in frozen_records},
        "controlled_documents_sha256": {rel: hashlib.sha256(_blob(repo_root, head, rel)).hexdigest()
                                        for rel in controlled_documents},
        "protocol": experiment.PROTOCOL_VERSION,
    }


def verify_stage_b(out_root: Path = runs.EXPERIMENT_ROOT, repo_root: Path = experiment.REPO_ROOT,
                   device: str = "cpu", *, records_dir: Path = experiment.RECORDS_DIR):
    """Load and verify all six Stage B models against the committed Stage B and selection records
    (``experiment.load_stage_b_models``). Reads only Stage B outputs and committed records."""
    stage_b = runs.read_json(Path(records_dir) / "release1-stage-b.json")
    selection = runs.read_json(Path(records_dir) / "release1-selection.json")
    models = experiment.load_stage_b_models(out_root, stage_b, device, selection_record=selection,
                                            repo_root=Path(repo_root))
    if {k: sorted(v) for k, v in models.items()} != {"C1": [0, 1, 2], "C3": [0, 1, 2]}:
        raise BatchEGateError(f"expected C1 and C3 models for seeds 0-2, got {sorted(models)}")
    return models


# --------------------------------------------------------------------------------------------
# Test-input safeguards (no data is read here)


def _string_ids(ids, name: str) -> List[str]:
    """The identifiers as a list of Python strings, refusing anything that is not genuinely a string.
    Accepted: a non-empty 1-D NumPy Unicode array (dtype kind ``U``), or a non-empty 1-D object array
    whose every element is a ``str`` (``numpy.str_`` is a ``str`` subclass). Nothing is converted:
    integers, floats, bytes, None and other objects are rejected, as are byte-string (``S``) arrays."""
    arr = np.asarray(ids)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} test identifiers must be a non-empty 1-D array")
    if arr.dtype.kind == "U":
        return [str(v) for v in arr.tolist()]
    if arr.dtype.kind == "O":
        values = arr.tolist()
        bad = [i for i, v in enumerate(values) if not isinstance(v, str)]
        if bad:
            raise ValueError(f"{name} test identifiers must all be strings; element {bad[0]} is "
                             f"{type(values[bad[0]]).__name__}")
        return [str(v) for v in values]
    raise ValueError(f"{name} test identifiers must be strings, got dtype {arr.dtype}")


def check_test_pairing(gt_ids: np.ndarray, gt_y: np.ndarray, est_ids: np.ndarray, est_y: np.ndarray,
                       expected_ids: Optional[Sequence[str]] = None):
    """Ground-truth and estimated test inputs must have identical, strictly increasing identifiers
    (sorted order) that are genuine strings, and identical labels in 0..14. ``expected_ids``, if
    given, must also be strings and equal the identifiers as a set. Returns (ids, y)."""
    listed = {}
    for name, ids in (("ground-truth", gt_ids), ("estimated", est_ids)):
        listed[name] = _string_ids(ids, name)
        values = listed[name]
        if any(b <= a for a, b in zip(values, values[1:])):
            raise ValueError(f"{name} test identifiers must be unique and in strictly increasing (sorted) order")
    if listed["ground-truth"] != listed["estimated"]:
        raise ValueError("ground-truth and estimated test identifiers differ")
    y = metrics.check_labels(gt_y, "ground-truth labels")
    y_est = metrics.check_labels(est_y, "estimated-input labels")
    if y.shape != (len(listed["ground-truth"]),) or not np.array_equal(y, y_est):
        raise ValueError("ground-truth and estimated test labels differ or do not match the identifiers")
    if expected_ids is not None:
        expected = list(expected_ids)
        if not all(isinstance(v, str) for v in expected):
            raise ValueError("expected test identifiers must all be strings")
        if listed["ground-truth"] != sorted(expected):
            raise ValueError("test identifiers differ from the expected test partition")
    return np.array(listed["ground-truth"]), y.copy()


def bootstrap_seed(config: Optional[dict] = None) -> int:
    """The D-45 seed from the committed configuration (checked against the code)."""
    cfg = config if config is not None else experiment.load_config()
    b = cfg["bootstrap"]
    if b.get("sd_ddof") != bootstrap.SD_DDOF or b.get("percentile_method") != bootstrap.PERCENTILE_METHOD \
            or b.get("n_resamples") != bootstrap.N_RESAMPLES:
        raise ValueError("configuration bootstrap conventions differ from the bootstrap module")
    seed = b.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("the bootstrap seed (D-45) is not fixed in the configuration")
    return seed


def checked_paired_comparison(gt_ids, gt_y, est_ids, est_y, preds_a: Mapping[int, np.ndarray],
                              preds_b: Mapping[int, np.ndarray], seed: int,
                              n_resamples: int = bootstrap.N_RESAMPLES) -> dict:
    """``bootstrap.paired_comparison`` after ``check_test_pairing`` (the seed is still explicit)."""
    _, y = check_test_pairing(gt_ids, gt_y, est_ids, est_y)
    return bootstrap.paired_comparison(y, preds_a, preds_b, seed=seed, n_resamples=n_resamples)


def checked_condition_summary(gt_ids, gt_y, est_ids, est_y, preds: Mapping[int, np.ndarray], seed: int,
                              n_resamples: int = bootstrap.N_RESAMPLES) -> dict:
    """``bootstrap.condition_summary`` after ``check_test_pairing`` (the seed is still explicit)."""
    _, y = check_test_pairing(gt_ids, gt_y, est_ids, est_y)
    return bootstrap.condition_summary(y, preds, seed=seed, n_resamples=n_resamples)
