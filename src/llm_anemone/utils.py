from __future__ import annotations

import random

import numpy as np
import torch
from scipy.stats import rankdata


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def empirical_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) <= 1 or np.all(values == values[0]):
        return np.zeros(len(values), dtype=np.float32)
    ranks = rankdata(values, method="average")
    return ((ranks - 1.0) / (len(values) - 1.0)).astype(np.float32)


def selected_positive_rank(values: np.ndarray, active: np.ndarray) -> np.ndarray:
    result = np.zeros(len(values), dtype=np.float32)
    index = np.flatnonzero(active & (np.asarray(values) > 0))
    if len(index) == 1:
        result[index] = 1.0
    elif len(index) > 1:
        result[index] = empirical_rank(np.asarray(values)[index])
    return result
