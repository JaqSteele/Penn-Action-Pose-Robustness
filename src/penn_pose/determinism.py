"""Deterministic execution, random-number streams, state hashing and provenance (protocol D-36, D-44).

D-36 (Section 9.5.8): FP32; ``torch.use_deterministic_algorithms(True)``; cuDNN deterministic on and
benchmark off; TF32 off for cuDNN and matrix multiplication; ``CUBLAS_WORKSPACE_CONFIG=:4096:8``
set before CUDA is initialised; no data-loading worker processes; on the CPU fallback, one thread.

D-44: separate seeded streams for weight initialisation, data ordering and other stochastic
operations (dropout). Each stream's seed is derived from the D-25 run seed (0, 1 or 2) and the
stream name only, so it does not depend on the condition (C1 or C3) or the hyperparameters:
matched seeds give identical initial weights, batch order and dropout sequence.

Importing this module sets the cuBLAS environment variable before ``torch`` is imported here.
"""

from __future__ import annotations

import hashlib
import os
import platform
import sys

CUBLAS_ENV = "CUBLAS_WORKSPACE_CONFIG"
CUBLAS_VALUE = ":4096:8"


class DeterminismError(RuntimeError):
    """The deterministic configuration required by D-36 is missing or cannot be guaranteed."""


def set_cuda_environment() -> None:
    """Set ``CUBLAS_WORKSPACE_CONFIG`` (must happen before CUDA is initialised)."""
    current = os.environ.get(CUBLAS_ENV)
    if current not in (None, CUBLAS_VALUE):
        raise DeterminismError(f"{CUBLAS_ENV} is set to {current!r}; D-36 requires {CUBLAS_VALUE!r}")
    os.environ[CUBLAS_ENV] = CUBLAS_VALUE


set_cuda_environment()

import numpy as np  # noqa: E402
import torch  # noqa: E402

STREAMS = ("init", "order", "dropout")


def stream_seed(seed: int, stream: str) -> int:
    """Seed of one named stream, derived from the run seed only (63-bit, from SHA-256)."""
    if stream not in STREAMS:
        raise ValueError(f"unknown stream {stream!r}; expected one of {STREAMS}")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    digest = hashlib.sha256(f"penn_pose/release1/{stream}/{int(seed)}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def is_cuda(device: str) -> bool:
    return torch.device(device).type == "cuda"


def rng_devices(device: str) -> list:
    """Device list for ``torch.random.fork_rng`` (CUDA device index, or none for the CPU)."""
    d = torch.device(device)
    return [d.index if d.index is not None else 0] if d.type == "cuda" else []


def configure(device: str) -> dict:
    """Apply the D-36 settings for ``device`` ('cpu' or 'cuda[:k]'). Returns the settings applied."""
    d = torch.device(device)
    if d.type not in ("cpu", "cuda"):
        raise DeterminismError(f"unsupported device {device!r}")
    set_cuda_environment()
    if d.type == "cuda" and torch.cuda.is_initialized():
        raise DeterminismError("CUDA was initialised before the deterministic configuration was applied")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if d.type == "cpu":
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads() != 1:
                raise DeterminismError("could not set one inter-op thread for the CPU device")
    check_configured(device)
    return settings_snapshot()


def settings_snapshot() -> dict:
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        CUBLAS_ENV: os.environ.get(CUBLAS_ENV),
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
        "data_loader_workers": 0,
        "dtype": "float32",
    }


def check_configured(device: str) -> None:
    """Raise ``DeterminismError`` unless every D-36 setting is in force for ``device``."""
    s = settings_snapshot()
    problems = []
    if not s["deterministic_algorithms"] or s["deterministic_warn_only"]:
        problems.append("deterministic algorithms not enforced")
    if not s["cudnn_deterministic"] or s["cudnn_benchmark"]:
        problems.append("cuDNN not deterministic or benchmark enabled")
    if s["cudnn_allow_tf32"] or s["matmul_allow_tf32"] or s["float32_matmul_precision"] != "highest":
        problems.append("TF32 enabled")
    if s[CUBLAS_ENV] != CUBLAS_VALUE:
        problems.append(f"{CUBLAS_ENV} not set to {CUBLAS_VALUE}")
    if torch.device(device).type == "cpu" and (s["num_threads"] != 1 or s["num_interop_threads"] != 1):
        problems.append("CPU device must use one thread")
    if problems:
        raise DeterminismError("; ".join(problems))


def state_hash(module_or_state: object) -> str:
    """SHA-256 of a model state (parameters and buffers, in state-dict order, with names,
    dtypes and shapes)."""
    state = module_or_state.state_dict() if hasattr(module_or_state, "state_dict") else module_or_state
    digest = hashlib.sha256()
    for name, value in state.items():
        t = value.detach().to("cpu").contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(t.dtype).encode("ascii") + b"\0")
        digest.update(repr(tuple(t.shape)).encode("ascii") + b"\0")
        digest.update(t.numpy().tobytes())
    return digest.hexdigest()


def environment_record(device: str) -> dict:
    """Device and software versions recorded for every run (D-36)."""
    import scipy
    import sklearn
    d = torch.device(device)
    record = {
        "device": str(d),
        "device_type": d.type,
        "device_name": None,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "settings": settings_snapshot(),
    }
    if d.type == "cuda":
        record["device_name"] = torch.cuda.get_device_name(d.index if d.index is not None else 0)
    else:
        record["device_name"] = platform.processor() or "cpu"
    return record


#: Fields that must be identical between the qualification record and every later run.
LOCKED_FIELDS = ("device", "device_type", "device_name", "torch", "cuda", "cudnn")


def locked_environment_problems(env) -> list:
    """Every locked field must be present, non-null and well formed: a CPU or CUDA ``device`` whose
    type equals ``device_type``; non-empty ``device_name``, ``torch`` and ``cuda`` strings; and a
    positive integer ``cudnn`` version (as recorded by ``environment_record``)."""
    if not isinstance(env, dict):
        return ["environment is missing or not a mapping"]
    problems = [f"environment.{k} is missing" for k in LOCKED_FIELDS if k not in env]
    problems += [f"environment.{k} is null" for k in LOCKED_FIELDS if k in env and env[k] is None]
    if problems:
        return problems
    device = env["device"]
    try:
        parsed = torch.device(device) if isinstance(device, str) else None
    except (RuntimeError, ValueError):
        parsed = None
    if parsed is None or parsed.type not in ("cpu", "cuda"):
        problems.append(f"environment.device is not a CPU or CUDA device: {device!r}")
    elif env["device_type"] != parsed.type:
        problems.append(f"environment.device_type {env['device_type']!r} does not match device {device!r}")
    for k in ("device_name", "torch", "cuda"):
        if not isinstance(env[k], str) or not env[k].strip():
            problems.append(f"environment.{k} must be a non-empty string, got {env[k]!r}")
    cudnn = env["cudnn"]
    if isinstance(cudnn, bool) or not isinstance(cudnn, int) or cudnn <= 0:
        problems.append(f"environment.cudnn must be a positive integer, got {cudnn!r}")
    return problems


def assert_same_environment(qualified: dict, current: dict) -> None:
    """D-36: the device (and the software that determines its numerics) never changes."""
    diffs = {k: (qualified.get(k), current.get(k)) for k in LOCKED_FIELDS if qualified.get(k) != current.get(k)}
    if diffs:
        raise DeterminismError(f"environment differs from the qualified device: {diffs}")
