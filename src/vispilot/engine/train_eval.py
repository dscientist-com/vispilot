from __future__ import annotations

import os
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from vispilot.utils import ensure_dir


def accuracy(logits, targets):
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def run_epoch(model, loader: DataLoader, criterion, optimizer=None, device: str = "cpu") -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for imgs, labels in tqdm(loader, disable=False):
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(imgs)
            loss = criterion(logits, labels)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy(logits, labels) * bs
        n += bs

    return total_loss / n, total_acc / n


def fit(model, train_loader, val_loader, epochs, lr, weight_decay, device: str = "cpu", ckpt_path: str | None = None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = -1.0

    # --- Hardened saver: ensure checkpoint directory always exists ---
    ckpt_root = os.path.dirname(ckpt_path) if ckpt_path else "checkpoints"
    ensure_dir(ckpt_root)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)

        print(
            f"Epoch {epoch}: "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if val_acc > best_val and ckpt_path:
            best_val = val_acc
            # Re-ensure in case folder was removed mid-run (AV/sync/etc.)
            ensure_dir(os.path.dirname(ckpt_path) or "checkpoints")
            torch.save({"model": model.state_dict(), "val_acc": val_acc}, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path} (val_acc={val_acc:.4f})")

    return model


def evaluate(model, loader, device: str = "cpu") -> float:
    criterion = nn.CrossEntropyLoss()
    loss, acc = run_epoch(model, loader, criterion, None, device)
    print(f"Test: loss {loss:.4f} acc {acc:.4f}")
    return acc
