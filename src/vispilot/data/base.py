from __future__ import annotations
from torch.utils.data import DataLoader
from typing import Tuple

class DataModule:
   
    def __init__(self, train_ds, val_ds, test_ds, batch_size: int = 128, num_workers: int = 4):
        self.train_ds, self.val_ds, self.test_ds = train_ds, val_ds, test_ds
        self.batch_size, self.num_workers = batch_size, num_workers

    def train_loader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_loader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_loader(self) -> DataLoader:
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

def num_classes_from_dataset(dataset) -> int:
    try:
        return len(dataset.classes)
    except Exception:
        
        targets = getattr(dataset, "targets", None)
        return int(max(targets)) + 1 if targets is not None and len(targets) > 0 else 10