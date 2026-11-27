"""Synthetic fixtures for the D2 tests (no Penn Action data; CPU only)."""

import subprocess
from pathlib import Path

import numpy as np

import _common  # noqa: F401  (puts src on the path)
from penn_pose import determinism, experiment
from penn_pose.selection import Plan
from penn_pose.training import TrainConfig

DEVICE = "cpu"
TINY_PLAN = Plan(grid=(TrainConfig(0.003, 0.0001, 0.1), TrainConfig(0.003, 0.001, 0.3)),
                 seeds=(0, 1), max_epochs=4, eval_every=2)


def configure_cpu() -> None:
    determinism.configure(DEVICE)


def data(n: int = 45, seed: int = 0, shift: float = 0.0):
    """Float32 inputs (n, 24, 64) with a weak class signal; every class present when n >= 15."""
    rng = np.random.default_rng(seed)
    y = (np.arange(n) % 15).astype(np.int64)
    X = rng.normal(size=(n, 24, 64)).astype(np.float32) * 0.5
    X[:, 0, :] += (y[:, None] * 0.2 + shift).astype(np.float32)
    return X, y


def inputs(source: str, n_fit: int = 45, n_val: int = 30, shift: float = 0.0) -> experiment.Inputs:
    """Synthetic ``experiment.Inputs`` (ids ascending; fitting then validation interleaved)."""
    X_fit, y_fit = data(n_fit, seed=1, shift=shift)
    X_val, y_val = data(n_val, seed=2, shift=shift)
    n = n_fit + n_val
    order = np.argsort(np.r_[np.arange(0, 2 * n_fit, 2), np.arange(1, 2 * n_val, 2)], kind="stable")
    X = np.concatenate([X_fit, X_val])[order]
    y = np.concatenate([y_fit, y_val])[order]
    subset = np.array(["fitting"] * n_fit + ["validation"] * n_val, dtype="<U10")[order]
    ids = np.array([f"{i + 1:04d}" for i in range(n)], dtype="<U4")
    return experiment.Inputs(source=source, X=np.ascontiguousarray(X), y=y, ids=ids, subset=subset)


def git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid",
                           "-c", "core.autocrlf=false", *args], cwd=repo, capture_output=True, text=True,
                          check=True)


def init_repo(repo: Path) -> None:
    """A temporary repository whose first commit contains every protected path (runs.PROTECTED_PATHS)."""
    git(repo, "init", "-q")
    (repo / "README").write_text("synthetic test repository\n", encoding="utf-8")
    files = {"src/penn_pose/code.py": "VALUE = 1\n", "scripts/run.py": "print('run')\n",
             "configs/recognition/release1.json": "{}\n", "configs/inputs/release1-cache.json": "{}\n",
             "configs/splits/fitting.txt": "0001\n", "requirements.txt": "numpy\n",
             "docs/RESEARCH_PROTOCOL.md": "# Research Protocol (synthetic)\n"}
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text, encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "init")


def commit(repo: Path, path: Path, message: str = "record") -> None:
    git(repo, "add", str(path.relative_to(repo)))
    git(repo, "commit", "-q", "-m", message)


def change_code(repo: Path, text: str = "VALUE = 2\n") -> None:
    """Commit a change to protected implementation code."""
    path = repo / "src" / "penn_pose" / "code.py"
    path.write_text(text, encoding="utf-8")
    commit(repo, path, "code change")


def prov(repo: Path, device_record_sha256: str = "d" * 64, **overrides) -> dict:
    """Complete production-style provenance for the repository as it is now."""
    p = experiment.provenance(DEVICE, None, repo)
    p["device_record_sha256"] = device_record_sha256
    p.update(overrides)
    return p
