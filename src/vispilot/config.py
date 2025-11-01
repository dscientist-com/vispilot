from __future__ import annotations
import yaml
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class Config:
    dataset: str
    data_root: str = "./data"
    batch_size: int = 128
    num_workers: int = 4
    model: str = "resnet18"
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    image_size: int = 224
    checkpoint_dir: str = "checkpoints"
    seed: int = 42
    device: str = "cuda"  # "cuda" | "cpu"

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        data: Dict[str, Any] = yaml.safe_load(f)
    return Config(**data)