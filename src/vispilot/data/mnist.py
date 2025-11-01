from __future__ import annotations
import torch
from torchvision import datasets, transforms
from .base import DataModule

def build(root: str, batch_size: int = 128, num_workers: int = 4, img_size: int = 224) -> DataModule:
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])
    train = datasets.MNIST(root, train=True, transform=tfm, download=True)
    test  = datasets.MNIST(root, train=False, transform=tfm, download=True)
    # simple split for val
    val = torch.utils.data.Subset(train, range(len(train)//10))
    train = torch.utils.data.Subset(train, range(len(train)//10, len(train)))
    return DataModule(train, val, test, batch_size=batch_size, num_workers=num_workers)