"""The Release 1.0 temporal CNN (protocol D-21, Section 9.5.2; defaults D-39).

    Input        24 channels x 64 frames
    Block 1      Conv1d(24 -> 64, k5, pad 2, no bias) -> BatchNorm -> ReLU -> Dropout(p) -> MaxPool(2)
    Block 2      Conv1d(64 -> 128, k5, pad 2, no bias) -> BatchNorm -> ReLU -> Dropout(p) -> MaxPool(2)
    Block 3      Conv1d(128 -> 128, k5, pad 2, no bias) -> BatchNorm -> ReLU -> Dropout(p)
    Pooling      global average over time (a plain mean, deterministic on every device)
    Output       Dropout(p) -> Linear(128 -> 15)

133,135 trainable parameters. PyTorch default initialisation and BatchNorm settings (D-39).
Weights are initialised from the run seed's "init" stream (D-44), on the CPU, so the initial state
is the same on every device and for matched C1/C3 runs.
"""

from __future__ import annotations

import torch
from torch import nn

from .determinism import stream_seed

N_CHANNELS = 24
N_FRAMES = 64
N_CLASSES = 15
EXPECTED_PARAMETERS = 133_135


def _block(c_in: int, c_out: int, dropout: float, pool: bool) -> list:
    layers = [nn.Conv1d(c_in, c_out, kernel_size=5, padding=2, bias=False), nn.BatchNorm1d(c_out),
              nn.ReLU(), nn.Dropout(dropout)]
    if pool:
        layers.append(nn.MaxPool1d(2))
    return layers


class TemporalCNN(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        if not (isinstance(dropout, float) and 0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be a float in [0, 1), got {dropout!r}")
        self.dropout_p = dropout
        self.features = nn.Sequential(*_block(N_CHANNELS, 64, dropout, True),
                                      *_block(64, 128, dropout, True),
                                      *_block(128, 128, dropout, False))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(128, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or tuple(x.shape[1:]) != (N_CHANNELS, N_FRAMES):
            raise ValueError(f"expected input (batch, {N_CHANNELS}, {N_FRAMES}), got {tuple(x.shape)}")
        h = self.features(x)            # (batch, 128, 16)
        h = h.mean(dim=2)               # global average over time
        return self.classifier(self.dropout(h))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(dropout: float, seed: int) -> TemporalCNN:
    """A new model initialised from the run seed's "init" stream, without touching any other
    random-number state (the global CPU generator is restored afterwards)."""
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(stream_seed(seed, "init"))   # CPU generator only
        model = TemporalCNN(dropout)
    if count_parameters(model) != EXPECTED_PARAMETERS:
        raise AssertionError(f"parameter count {count_parameters(model)} != {EXPECTED_PARAMETERS}")
    return model
