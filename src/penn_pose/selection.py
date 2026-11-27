"""Stage A selection on C1 validation (protocol Section 9.5.4; D-22, D-25, D-44).

Grid: learning rate {0.001, 0.003} x weight decay {0.0001, 0.001} x dropout {0.1, 0.3}; seeds
0, 1, 2; up to 150 epochs; ground-truth validation macro-F1 recorded at epochs 5, 10, ..., 150.

Rule: choose the (configuration, E) with the highest mean validation macro-F1 over the three seeds.
Ties (D-44: exact equality of the unrounded float64 means) are broken by the smaller E, then the
smaller learning rate, weight decay and dropout. Means are computed with ``math.fsum`` (correctly
rounded, independent of summation order) divided by the number of seeds.

Only C1 ground-truth validation scores enter this module (D-13).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from .training import TrainConfig

LEARNING_RATES = (0.001, 0.003)
WEIGHT_DECAYS = (0.0001, 0.001)
DROPOUTS = (0.1, 0.3)
SEEDS = (0, 1, 2)
MAX_EPOCHS = 150
EVAL_EVERY = 5


@dataclass(frozen=True)
class Plan:
    """The selection design. ``RELEASE1`` is the protocol plan; reduced plans are for tests only."""
    grid: Tuple[TrainConfig, ...]
    seeds: Tuple[int, ...]
    max_epochs: int
    eval_every: int

    @property
    def eval_epochs(self) -> Tuple[int, ...]:
        return tuple(range(self.eval_every, self.max_epochs + 1, self.eval_every))

    def validate(self) -> None:
        if not self.grid or len(set(self.grid)) != len(self.grid):
            raise ValueError("grid must be non-empty with distinct configurations")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and distinct")
        if self.eval_every < 1 or self.max_epochs < self.eval_every or self.max_epochs % self.eval_every:
            raise ValueError("max_epochs must be a positive multiple of eval_every")

    def as_dict(self) -> dict:
        return {"grid": [c.as_dict() for c in self.grid], "seeds": list(self.seeds),
                "max_epochs": self.max_epochs, "eval_every": self.eval_every,
                "eval_epochs": list(self.eval_epochs)}


RELEASE1 = Plan(
    grid=tuple(TrainConfig(lr, wd, p) for lr in LEARNING_RATES for wd in WEIGHT_DECAYS for p in DROPOUTS),
    seeds=SEEDS, max_epochs=MAX_EPOCHS, eval_every=EVAL_EVERY,
)


@dataclass(frozen=True)
class Score:
    config: TrainConfig
    seed: int
    epoch: int
    macro_f1: float


@dataclass(frozen=True)
class Candidate:
    config: TrainConfig
    epoch: int
    mean_macro_f1: float
    per_seed: Tuple[float, ...]

    def key(self):
        c = self.config
        return (-self.mean_macro_f1, self.epoch, c.learning_rate, c.weight_decay, c.dropout)


@dataclass(frozen=True)
class Selection:
    selected: Candidate
    tied: Tuple[Candidate, ...]          # all candidates whose mean equals the best mean exactly
    candidates: Tuple[Candidate, ...]    # every (configuration, E), in grid then epoch order


def candidates(scores: Iterable[Score], plan: Plan) -> List[Candidate]:
    """Validate completeness and compute the seed means for every (configuration, E)."""
    plan.validate()
    table: Dict[Tuple[TrainConfig, int, int], float] = {}
    for s in scores:
        key = (s.config, s.seed, s.epoch)
        if s.config not in plan.grid or s.seed not in plan.seeds or s.epoch not in plan.eval_epochs:
            raise ValueError(f"score outside the plan: {key}")
        if key in table:
            raise ValueError(f"duplicate score: {key}")
        if not isinstance(s.macro_f1, float) or not math.isfinite(s.macro_f1) or not 0.0 <= s.macro_f1 <= 1.0:
            raise ValueError(f"invalid macro-F1 {s.macro_f1!r} for {key}")
        table[key] = s.macro_f1
    expected = len(plan.grid) * len(plan.seeds) * len(plan.eval_epochs)
    if len(table) != expected:
        raise ValueError(f"incomplete Stage A scores: {len(table)} of {expected}")
    out = []
    for c in plan.grid:
        for e in plan.eval_epochs:
            per_seed = tuple(table[(c, s, e)] for s in plan.seeds)
            out.append(Candidate(c, e, math.fsum(per_seed) / len(per_seed), per_seed))
    return out


def select(scores: Iterable[Score], plan: Plan = RELEASE1) -> Selection:
    cands = candidates(scores, plan)
    best = min(cands, key=Candidate.key)
    tied = tuple(c for c in cands if c.mean_macro_f1 == best.mean_macro_f1)
    return Selection(selected=best, tied=tied, candidates=tuple(cands))
