"""The deterministic fitting/validation split of the official training partition (protocol D-02).

Procedure (protocol Section 4.1.1):
1. Take the training-partition videos (``train = 1``).
2. Process the action classes in alphabetical order.
3. Within each class, sort identifiers ascending; n_val = floor(0.15 * n_class + 0.5).
4. One ``numpy.random.default_rng(20261126)`` generator shared across classes in that order;
   for each class draw ``perm = rng.permutation(n_class)`` and take ``perm[:n_val]`` as validation.
5. All other training videos form the fitting subset.

Lists are stored as sorted four-digit identifiers, one per line, LF-separated, with a final LF,
ASCII. The recorded SHA-256 values are computed on exactly that content. Because Git may convert
line endings on checkout, stored files are verified by parsing them and re-serialising in the
canonical form before hashing, and the parse itself is strict.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple, Union

import numpy as np

PathLike = Union[str, Path]

SEED = 20261126
VALIDATION_FRACTION = 0.15
EXPECTED_COUNTS = {"fitting": 1070, "validation": 188}
EXPECTED_SHA256 = {
    "validation": "9863e3a89640cdc76861e310a11527d3748e379f15910a1ce166d0468b0d5828",
    "fitting": "b84e4fdab0772dd1430df33a1c62bd6f217d6661ca7ab444beadd3e0bcdbed55",
}
DEFAULT_SPLIT_DIR = Path(__file__).resolve().parents[2] / "configs" / "splits"


class SplitError(ValueError):
    """A split list is malformed or does not match the recorded hash."""


def n_validation(n_class: int) -> int:
    """Half-up rounding of 15% of a class size."""
    return int(math.floor(VALIDATION_FRACTION * n_class + 0.5))


def generate_split(train_actions: Mapping[str, str], seed: int = SEED) -> Tuple[List[str], List[str]]:
    """Return (fitting, validation) identifier lists, each sorted ascending."""
    rng = np.random.default_rng(seed)
    validation: List[str] = []
    for action in sorted(set(train_actions.values())):
        ids = sorted(v for v, a in train_actions.items() if a == action)
        perm = rng.permutation(len(ids))
        validation.extend(ids[i] for i in perm[: n_validation(len(ids))])
    val_set = set(validation)
    fitting = sorted(v for v in train_actions if v not in val_set)
    return fitting, sorted(validation)


def per_class_counts(train_actions: Mapping[str, str], validation: Iterable[str]) -> Dict[str, Tuple[int, int]]:
    """{action: (n_training, n_validation)} for reporting."""
    val = set(validation)
    out = {}
    for action in sorted(set(train_actions.values())):
        ids = [v for v, a in train_actions.items() if a == action]
        out[action] = (len(ids), sum(v in val for v in ids))
    return out


def canonical_text(ids: Iterable[str]) -> str:
    ids = list(ids)
    return "".join(f"{i}\n" for i in ids)


def list_sha256(ids: Iterable[str]) -> str:
    return hashlib.sha256(canonical_text(ids).encode("ascii")).hexdigest()


def parse_split_text(text: str, name: str = "split") -> List[str]:
    """Strictly parse a stored list: four-digit ids, ascending, unique, no blank lines."""
    lines = text.replace("\r\n", "\n").split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    else:
        raise SplitError(f"{name}: the file must end with a line feed")
    for n, line in enumerate(lines, 1):
        if not (len(line) == 4 and line.isdigit()):
            raise SplitError(f"{name}: line {n} is not a four-digit identifier: {line!r}")
    if lines != sorted(set(lines)):
        raise SplitError(f"{name}: identifiers must be unique and in ascending order")
    return lines


def write_split_files(fitting: List[str], validation: List[str], directory: PathLike = DEFAULT_SPLIT_DIR) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, ids in (("fitting", fitting), ("validation", validation)):
        if list_sha256(ids) != EXPECTED_SHA256[name]:
            raise SplitError(f"Refusing to write {name}: its hash does not match the protocol")
        with open(directory / f"{name}.txt", "w", encoding="ascii", newline="\n") as fh:
            fh.write(canonical_text(ids))


def load_split(name: str, directory: PathLike = DEFAULT_SPLIT_DIR, verify: bool = True) -> List[str]:
    """Load 'fitting' or 'validation'; by default verify count and recorded SHA-256."""
    if name not in EXPECTED_SHA256:
        raise ValueError("name must be 'fitting' or 'validation'")
    path = Path(directory) / f"{name}.txt"
    if not path.is_file():
        raise SplitError(f"Split file not found: {path}")
    ids = parse_split_text(path.read_text(encoding="ascii"), name)
    if verify:
        if len(ids) != EXPECTED_COUNTS[name]:
            raise SplitError(f"{name}: expected {EXPECTED_COUNTS[name]} ids, found {len(ids)}")
        digest = list_sha256(ids)
        if digest != EXPECTED_SHA256[name]:
            raise SplitError(f"{name}: SHA-256 {digest} does not match the protocol value "
                             f"{EXPECTED_SHA256[name]}")
    return ids
