from __future__ import annotations
import os, random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def get_device(prefer: str = "cuda"):
    return "cuda" if (prefer == "cuda" and torch.cuda.is_available()) else "cpu"