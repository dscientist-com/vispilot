from __future__ import annotations
from torchvision import datasets, transforms
from .base import DataModule

def build(root: str, batch_size: int = 128, num_workers: int = 4, img_size: int = 224) -> DataModule:
    train_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train = datasets.CIFAR10(root, train=True, transform=train_tfm, download=True)
    test  = datasets.CIFAR10(root, train=False, transform=test_tfm, download=True)
    val = test  
    return DataModule(train, val, test, batch_size=batch_size, num_workers=num_workers)