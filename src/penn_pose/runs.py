"""Experiment output management (protocol D-42, Section 9.5.8; D-41 commit order).

- Outputs live outside Git, under ``C:\\datasets\\penn-action\\experiments\\release1\\``.
- Each run is written to ``<run>.incomplete`` and renamed to ``<run>`` only after its SHA-256
  manifest (``run-sha256.txt``) has been written. Existing runs, stage manifests and repository
  records are never overwritten; a leftover ``.incomplete`` folder blocks the run until removed.
- ``require_committed`` enforces that a repository record (for example the Stage A selection
  record) is tracked, committed and unmodified before later steps may run.
- ``code_identity`` identifies the committed implementation (HEAD plus a hash of the protected
  code/configuration tree) and refuses if the protected scope is not committed and clean.
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch

from . import integrity
from .labels import DEFAULT_DATA_ROOT

PathLike = Union[str, Path]

EXPERIMENT_ROOT = DEFAULT_DATA_ROOT / "experiments" / "release1"
RUN_MANIFEST = "run-sha256.txt"
INCOMPLETE = ".incomplete"


class OutputExistsError(FileExistsError):
    pass


class CommitGateError(RuntimeError):
    pass


def _dump_json(obj) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=False, allow_nan=False) + "\n").encode("utf-8")


class RunWriter:
    """Write one run's files into a staging folder, then publish it atomically by renaming."""

    def __init__(self, final_dir: PathLike):
        self.final_dir = Path(final_dir)
        self.staging = self.final_dir.with_name(self.final_dir.name + INCOMPLETE)
        for p in (self.final_dir, self.staging):
            if p.exists():
                raise OutputExistsError(f"Refusing to overwrite: {p} already exists")
        self.final_dir.parent.mkdir(parents=True, exist_ok=True)
        self.staging.mkdir()
        self.finalised = False

    def _path(self, name: str) -> Path:
        if self.finalised:
            raise RuntimeError("run already finalised")
        if "/" in name or "\\" in name or name in ("", RUN_MANIFEST) or name.endswith(INCOMPLETE):
            raise ValueError(f"invalid file name {name!r}")
        return self.staging / name

    def write_bytes(self, name: str, data: bytes) -> None:
        with open(self._path(name), "xb") as fh:
            fh.write(data)

    def write_json(self, name: str, obj) -> None:
        self.write_bytes(name, _dump_json(obj))

    def write_npz(self, name: str, **arrays) -> None:
        buf = io.BytesIO()
        np.savez(buf, **{k: np.asarray(v) for k, v in arrays.items()})
        self.write_bytes(name, buf.getvalue())

    def write_state_dict(self, name: str, state: dict) -> None:
        buf = io.BytesIO()
        torch.save({k: v.detach().to("cpu") for k, v in state.items()}, buf)
        self.write_bytes(name, buf.getvalue())

    def finalise(self) -> str:
        """Write the run manifest, rename the folder into place, and return the manifest SHA-256."""
        files = sorted(p.name for p in self.staging.iterdir())
        if not files:
            raise RuntimeError("refusing to finalise an empty run")
        lines = "".join(f"{integrity.sha256_file(self.staging / n)} *{n}\n" for n in files)
        with open(self.staging / RUN_MANIFEST, "x", encoding="ascii", newline="\n") as fh:
            fh.write(lines)
        if self.final_dir.exists():
            raise OutputExistsError(f"Refusing to overwrite: {self.final_dir} appeared during the run")
        self.staging.rename(self.final_dir)
        self.finalised = True
        return integrity.sha256_file(self.final_dir / RUN_MANIFEST)


def verify_run(run_dir: PathLike) -> str:
    """Check a finished run against its manifest (exact file set). Returns the manifest SHA-256."""
    run_dir = Path(run_dir)
    manifest = integrity.read_manifest(run_dir / RUN_MANIFEST)
    present = {p.name for p in run_dir.iterdir()} - {RUN_MANIFEST}
    if present != set(manifest):
        raise ValueError(f"{run_dir}: files {sorted(present)} differ from the manifest {sorted(manifest)}")
    bad = [n for n, h in manifest.items() if integrity.sha256_file(run_dir / n) != h]
    if bad:
        raise ValueError(f"{run_dir}: hash mismatch for {bad}")
    return integrity.sha256_file(run_dir / RUN_MANIFEST)


def read_json(path: PathLike):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stage_runs(stage_dir: PathLike) -> Dict[str, str]:
    """{run name: manifest SHA-256} for every finished run; refuses leftover incomplete runs."""
    stage_dir = Path(stage_dir)
    if not stage_dir.is_dir():
        return {}
    leftovers = sorted(p.name for p in stage_dir.iterdir() if p.name.endswith(INCOMPLETE))
    if leftovers:
        raise OutputExistsError(f"{stage_dir}: incomplete runs must be inspected and removed first: {leftovers}")
    return {p.name: verify_run(p) for p in sorted(stage_dir.iterdir()) if p.is_dir()}


def write_stage_manifest(stage_dir: PathLike, expected_runs: List[str]) -> str:
    """Write ``stage-sha256.txt`` (one line per run manifest) once every expected run exists."""
    stage_dir = Path(stage_dir)
    runs = stage_runs(stage_dir)
    if sorted(runs) != sorted(expected_runs):
        raise ValueError(f"{stage_dir}: runs {sorted(runs)} differ from the expected {sorted(expected_runs)}")
    path = stage_dir / "stage-sha256.txt"
    with open(path, "x", encoding="ascii", newline="\n") as fh:
        fh.writelines(f"{runs[n]} *{n}/{RUN_MANIFEST}\n" for n in sorted(runs))
    return integrity.sha256_file(path)


def write_record(path: PathLike, obj) -> str:
    """Write a compact repository record (JSON) without overwriting. Returns its SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as fh:
        fh.write(_dump_json(obj))
    return integrity.sha256_file(path)


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True)


def git_state(repo_root: PathLike) -> dict:
    repo_root = Path(repo_root)
    head = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
    status = _git(repo_root, "status", "--porcelain", "--untracked-files=all").stdout.splitlines()
    return {"commit": head, "dirty": bool(status), "changes": status}


#: Files and folders whose content determines what an experiment computes or which methodology it
#: follows (D-41 code identity). This canonical list is the only one ever used for checks.
#: Records, results, explanatory documentation and tests are deliberately outside this scope, so
#: committing a selection, device or stage record never changes the protected code identity.
PROTECTED_PATHS = (
    "src",                                   # all implementation code (D1 preprocessing and D2)
    "scripts",                               # command-line entry points
    "configs/recognition/release1.json",     # the experiment configuration (D-42)
    "configs/inputs/release1-cache.json",    # the recorded D1 input-cache hashes
    "configs/splits",                        # the D-02 fitting/validation lists
    "requirements.txt",                      # the main-environment dependency specification
    "docs/RESEARCH_PROTOCOL.md",             # the authoritative research protocol (v0.14)
)


class CodeIdentityError(RuntimeError):
    """The protected experimental code/configuration is uncommitted, modified or has changed."""


def protected_scope_problems(repo_root: PathLike, paths=PROTECTED_PATHS) -> List[str]:
    """Every tracked modification, staged change, untracked file or ignored file inside the
    protected scope (``git status`` lines). Python bytecode caches (``__pycache__``) are allowed:
    Python only uses them when the matching committed source file exists."""
    proc = _git(Path(repo_root), "status", "--porcelain=v1", "--untracked-files=all", "--ignored", "--", *paths)
    if proc.returncode != 0:
        raise CodeIdentityError(f"git status failed: {proc.stderr.strip()}")
    problems = []
    for line in proc.stdout.splitlines():
        if line.startswith("!! ") and "__pycache__/" in line[3:].replace("\\", "/"):
            continue
        problems.append(line)
    return problems


def protected_tree_sha256(repo_root: PathLike, commit: str, paths=PROTECTED_PATHS) -> str:
    """SHA-256 of ``git ls-tree -r`` for the protected paths at ``commit`` (modes, blob ids, paths).
    Every protected path must exist in that commit."""
    proc = _git(Path(repo_root), "ls-tree", "-r", "--full-tree", commit, "--", *paths)
    if proc.returncode != 0:
        raise CodeIdentityError(f"git ls-tree failed for {commit}: {proc.stderr.strip()}")
    listed = [line.split("\t", 1)[1] for line in proc.stdout.splitlines()]
    missing = [p for p in paths if not any(f == p or f.startswith(p + "/") for f in listed)]
    if missing:
        raise CodeIdentityError(f"protected paths missing from commit {commit}: {missing}")
    return hashlib.sha256(proc.stdout.encode("utf-8")).hexdigest()


def code_identity(repo_root: PathLike, paths=PROTECTED_PATHS) -> dict:
    """The committed implementation in use: HEAD and the hash of its protected tree. Refuses if
    anything in the protected scope is uncommitted, modified, staged, untracked or ignored."""
    problems = protected_scope_problems(repo_root, paths)
    if problems:
        raise CodeIdentityError("protected experimental code/configuration must be committed and clean "
                                f"(including no untracked files): {problems[:10]}")
    head = _git(Path(repo_root), "rev-parse", "HEAD").stdout.strip()
    return {"commit": head, "protected_paths": list(paths),
            "protected_tree_sha256": protected_tree_sha256(repo_root, head, paths)}


def is_ancestor(repo_root: PathLike, ancestor: str, descendant: str) -> bool:
    """True if ``ancestor`` is ``descendant`` or one of its ancestors."""
    return _git(Path(repo_root), "merge-base", "--is-ancestor", ancestor, descendant).returncode == 0


def require_committed(path: PathLike, repo_root: PathLike) -> str:
    """D-41: ``path`` must be tracked, committed in HEAD and unmodified. Returns the commit hash."""
    repo_root = Path(repo_root).resolve()
    path = Path(path).resolve()
    rel = path.relative_to(repo_root).as_posix()
    if not path.is_file():
        raise CommitGateError(f"{rel} does not exist")
    if _git(repo_root, "ls-files", "--error-unmatch", "--", rel).returncode != 0:
        raise CommitGateError(f"{rel} is not tracked by Git; commit it first")
    if _git(repo_root, "diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        raise CommitGateError(f"{rel} differs from the committed version; commit it first")
    committed = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=repo_root, capture_output=True).stdout
    on_disk = path.read_bytes()
    if committed.replace(b"\r\n", b"\n") != on_disk.replace(b"\r\n", b"\n"):
        raise CommitGateError(f"{rel} on disk differs from HEAD")
    return _git(repo_root, "rev-parse", "HEAD").stdout.strip()
