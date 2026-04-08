from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def replace_nan_with_zero(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=0.0)


def timestamp_dir(base_dir: str | Path, experiment_name: str) -> Path:
    out = ensure_dir(base_dir)
    exp = out / experiment_name
    exp.mkdir(parents=True, exist_ok=True)
    return exp
