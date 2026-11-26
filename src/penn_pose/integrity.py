"""Integrity checks for the local Penn Action annotation files.

The SHA-256 manifest is kept outside the repository, next to the dataset
(``<root>/manifests/labels-sha256.txt``, lines of the form ``<sha256> *<file name>``).
Checks only read file bytes; they never modify or rewrite any dataset file.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union

from .labels import DEFAULT_DATA_ROOT, labels_dir

PathLike = Union[str, Path]

ARCHIVE_SHA256 = "e3e41bd99deb7b3beb9785f78b209de272bb0fa60ff4432ffa88118907a797de"
EXPECTED_LABEL_FILES = 2326


@dataclass
class IntegrityResult:
    checked: int = 0
    mismatched: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    unexpected: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.mismatched or self.missing or self.unexpected)


def sha256_file(path: PathLike, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_manifest(root: PathLike = DEFAULT_DATA_ROOT) -> Path:
    return Path(root) / "manifests" / "labels-sha256.txt"


def read_manifest(path: PathLike) -> Dict[str, str]:
    """Parse ``<sha256> *<name>`` (or ``<sha256>  <name>``) lines into {name: sha256}."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    entries: Dict[str, str] = {}
    for n, line in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"{path.name}: malformed line {n}: {line!r}")
        name = parts[1].lstrip("*").strip()
        if name in entries:
            raise ValueError(f"{path.name}: duplicate entry {name}")
        entries[name] = parts[0].lower()
    return entries


def verify_labels(root: PathLike = DEFAULT_DATA_ROOT, manifest: PathLike = None) -> IntegrityResult:
    """Compare every ``labels/*.mat`` file with the manifest."""
    expected = read_manifest(manifest if manifest is not None else default_manifest(root))
    directory = labels_dir(root)
    present = {p.name for p in directory.glob("*.mat")}
    result = IntegrityResult()
    for name, digest in sorted(expected.items()):
        path = directory / name
        if name not in present:
            result.missing.append(name)
            continue
        result.checked += 1
        if sha256_file(path) != digest:
            result.mismatched.append(name)
    result.unexpected = sorted(present - set(expected))
    return result


def verify_archive(root: PathLike = DEFAULT_DATA_ROOT) -> bool:
    """SHA-256 of ``Penn_Action.tar.gz`` against the recorded value (reads about 3.2 GB)."""
    return sha256_file(Path(root) / "Penn_Action.tar.gz") == ARCHIVE_SHA256
