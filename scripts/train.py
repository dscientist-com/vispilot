import argparse
import os
import torch

from vispilot.config import load_config
from vispilot.utils import set_seed, ensure_dir, get_device
from vispilot.data import mnist, cifar10, fashion_mnist, stl10
from vispilot.models.classifiers import create_classifier
from vispilot.data.base import num_classes_from_dataset
from vispilot.engine.train_eval import fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    print(f"Using device: {device}")

    # --- Unified dataset loader block ---
    ds = cfg.dataset.lower()
    if ds == "mnist":
        dm = mnist.build(cfg.data_root, cfg.batch_size, cfg.num_workers, cfg.image_size)
    elif ds in ("cifar10", "cifar-10"):
        dm = cifar10.build(cfg.data_root, cfg.batch_size, cfg.num_workers, cfg.image_size)
    elif ds in ("fashion-mnist", "fashion_mnist", "fashionmnist"):
        dm = fashion_mnist.build(cfg.data_root, cfg.batch_size, cfg.num_workers, cfg.image_size)
    elif ds == "stl10":
        dm = stl10.build(cfg.data_root, cfg.batch_size, cfg.num_workers, cfg.image_size)
    else:
        raise ValueError(f"Unsupported dataset in quickstart CLI: {cfg.dataset}")

    # --- Model and training ---
    num_classes = num_classes_from_dataset(dm.train_ds)
    model = create_classifier(cfg.model, num_classes=num_classes, pretrained=True).to(device)

    ensure_dir(cfg.checkpoint_dir)
    ckpt = os.path.join(cfg.checkpoint_dir, f"{cfg.dataset.lower()}_{cfg.model}.pth")

    fit(
        model,
        dm.train_loader(),
        dm.val_loader(),
        cfg.epochs,
        cfg.lr,
        cfg.weight_decay,
        device=device,
        ckpt_path=ckpt,
    )


if __name__ == "__main__":
    main()
