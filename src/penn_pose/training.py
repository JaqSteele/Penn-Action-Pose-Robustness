"""Training and evaluation of the temporal CNN (protocol Sections 9.5.3, 9.5.4; D-24, D-25, D-36,
D-38, D-39, D-40, D-44).

- Class-weighted cross-entropy, w_c = N / (K * N_c), from the run's own training labels (D-24),
  averaged per batch by PyTorch's standard ``reduction="mean"``: weighted sum / sum of weights (D-40).
- AdamW with its defaults apart from the learning rate and weight decay (D-22, D-39); weight decay
  on all parameters; constant learning rate.
- Batch size 32; videos shuffled every epoch; the last partial batch is kept (D-38).
- No augmentation, schedule, warm-up, gradient clipping, label smoothing or mixed precision.
- Random-number streams (D-44): weights from the "init" stream (``model.build_model``); batch order
  from a dedicated CPU generator seeded from the "order" stream; dropout from the global generator
  seeded from the "dropout" stream. Evaluation runs in inference mode, without gradients, inside
  ``torch.random.fork_rng``, so it can never change the training random-number sequence.
- The model-state SHA-256 is recorded at every requested checkpoint epoch (D-41).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn

from . import determinism, metrics
from .model import N_CHANNELS, N_CLASSES, N_FRAMES, TemporalCNN, build_model

BATCH_SIZE = 32
EVAL_BATCH_SIZE = 256


@dataclass(frozen=True)
class TrainConfig:
    """One point of the D-22 grid."""
    learning_rate: float
    weight_decay: float
    dropout: float

    def as_dict(self) -> dict:
        return {"learning_rate": self.learning_rate, "weight_decay": self.weight_decay, "dropout": self.dropout}

    @property
    def name(self) -> str:
        return f"lr{self.learning_rate:g}_wd{self.weight_decay:g}_p{self.dropout:g}"


def check_inputs(X: np.ndarray, y: np.ndarray) -> None:
    if not isinstance(X, np.ndarray) or X.dtype != np.float32 or X.ndim != 3 or X.shape[1:] != (N_CHANNELS, N_FRAMES):
        raise ValueError(f"X must be float32 (n, {N_CHANNELS}, {N_FRAMES}), got "
                         f"{getattr(X, 'dtype', type(X))} {getattr(X, 'shape', None)}")
    metrics.check_labels(y, "y")
    if y.dtype != np.int64 or y.shape != (X.shape[0],):
        raise ValueError(f"y must be int64 of shape ({X.shape[0]},), got {y.dtype} {y.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values")


def class_weights(y_train: np.ndarray) -> torch.Tensor:
    """D-24 weights w_c = N / (K * N_c), computed in float64 and returned as float32."""
    y = metrics.check_labels(y_train, "y_train")
    counts = np.bincount(y, minlength=N_CLASSES)
    if np.any(counts == 0):
        raise ValueError(f"every class needs at least one training video; missing {np.flatnonzero(counts == 0).tolist()}")
    w = y.size / (N_CLASSES * counts.astype(np.float64))
    return torch.tensor(w, dtype=torch.float32)


def make_loss(weights: torch.Tensor) -> nn.CrossEntropyLoss:
    return nn.CrossEntropyLoss(weight=weights, reduction="mean")


def make_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.AdamW:
    return torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)


def batch_slices(n: int, batch_size: int = BATCH_SIZE) -> List[slice]:
    """Consecutive batches over a permutation of length n; the last partial batch is kept."""
    if n < 1 or batch_size < 1:
        raise ValueError("n and batch_size must be positive")
    return [slice(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]


def order_generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(determinism.stream_seed(seed, "order"))


def epoch_orders(n: int, seed: int, epochs: int) -> List[np.ndarray]:
    """The batch order of every epoch for a run seed (identical for C1 and C3 runs)."""
    g = order_generator(seed)
    return [torch.randperm(n, generator=g).numpy() for _ in range(epochs)]


def _sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


@torch.no_grad()
def predict_logits(model: nn.Module, X: np.ndarray, device: str) -> np.ndarray:
    """Logits in inference mode, in fixed-size batches, without touching any random stream."""
    if X.dtype != np.float32 or X.ndim != 3 or X.shape[1:] != (N_CHANNELS, N_FRAMES):
        raise ValueError(f"X must be float32 (n, {N_CHANNELS}, {N_FRAMES})")
    was_training = model.training
    out = []
    with torch.random.fork_rng(devices=determinism.rng_devices(device)):
        model.eval()
        for s in batch_slices(X.shape[0], EVAL_BATCH_SIZE):
            out.append(model(torch.from_numpy(X[s]).to(device)).float().cpu().numpy())
    model.train(was_training)
    return np.concatenate(out).astype(np.float32)


def predict(model: nn.Module, X: np.ndarray, device: str) -> np.ndarray:
    """Class predictions (argmax; ties resolve to the smallest class index)."""
    return predict_logits(model, X, device).argmax(axis=1).astype(np.int64)


def make_evaluator(X: np.ndarray, y: np.ndarray, device: str) -> Callable[[nn.Module], dict]:
    """Evaluator returning macro-F1, accuracy and per-video predictions for one evaluation set."""
    check_inputs(X, y)

    def evaluate(model: nn.Module) -> dict:
        y_pred = predict(model, X, device)
        return {"macro_f1": metrics.macro_f1(y, y_pred), "accuracy": metrics.accuracy(y, y_pred), "y_pred": y_pred}
    return evaluate


@dataclass
class TrainResult:
    model: TemporalCNN
    initial_state_sha256: str
    final_state_sha256: str
    epochs: List[dict]                         # per epoch: training loss, training accuracy, order hash
    checkpoints: Dict[int, dict] = field(default_factory=dict)   # epoch -> state hash (+ evaluation)


def train_model(X: np.ndarray, y: np.ndarray, config: TrainConfig, seed: int, epochs: int, device: str,
                checkpoint_epochs: Sequence[int] = (),
                evaluate: Optional[Callable[[nn.Module], dict]] = None) -> TrainResult:
    """Train one model from scratch. ``evaluate`` (if given) runs only at ``checkpoint_epochs``."""
    check_inputs(X, y)
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError(f"epochs must be a positive integer, got {epochs!r}")
    checkpoints = sorted(set(int(e) for e in checkpoint_epochs))
    if checkpoints and (checkpoints[0] < 1 or checkpoints[-1] > epochs):
        raise ValueError(f"checkpoint epochs must lie in 1..{epochs}, got {checkpoints}")
    determinism.check_configured(device)

    model = build_model(config.dropout, seed)
    initial_hash = determinism.state_hash(model)
    model.to(device)
    order_gen = order_generator(seed)
    torch.manual_seed(determinism.stream_seed(seed, "dropout"))   # dropout stream (CPU and CUDA)

    X_dev = torch.from_numpy(X).to(device)
    y_dev = torch.from_numpy(y).to(device)
    weights = class_weights(y).to(device)
    loss_fn = make_loss(weights)
    optimizer = make_optimizer(model, config)
    slices = batch_slices(X.shape[0])

    epoch_records: List[dict] = []
    checkpoint_records: Dict[int, dict] = {}
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(X.shape[0], generator=order_gen)
        weighted_loss_sum = 0.0                              # sum over batches of (batch loss x batch weight)
        weight_sum = 0.0
        correct = 0
        for s in slices:
            idx = order[s].to(device)
            logits = model(X_dev[idx])
            loss = loss_fn(logits, y_dev[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_weight = float(weights[y_dev[idx]].sum().item())
            weighted_loss_sum += float(loss.item()) * batch_weight
            weight_sum += batch_weight
            correct += int((logits.argmax(dim=1) == y_dev[idx]).sum().item())
        epoch_records.append({
            "epoch": epoch,
            # Epoch-level class-weighted mean of the per-video losses, sum(w_i l_i) / sum(w_i), the
            # D-40 definition applied to the whole epoch (the smaller last batch counts by its weight).
            "train_loss": weighted_loss_sum / weight_sum,
            # Proportion of training videos whose training-mode prediction (dropout active, batch
            # statistics) was correct when it was computed during the epoch.
            "train_accuracy": correct / X.shape[0],
            "order_sha256": _sha256_array(order.numpy().astype(np.int64)),
        })
        if epoch in checkpoints:
            record = {"epoch": epoch, "state_sha256": determinism.state_hash(model)}
            if evaluate is not None:
                record.update(evaluate(model))
                if determinism.state_hash(model) != record["state_sha256"]:
                    raise AssertionError("evaluation changed the model state")
            checkpoint_records[epoch] = record
    return TrainResult(model=model, initial_state_sha256=initial_hash,
                       final_state_sha256=determinism.state_hash(model),
                       epochs=epoch_records, checkpoints=checkpoint_records)
