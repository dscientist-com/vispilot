from __future__ import annotations
import torch
import torch.nn as nn
from torchvision import models

def create_classifier(name: str, num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    elif name in ("efficientnet_v2_s", "effnetv2s", "efficientnetv2s"):
        m = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    else:
        raise ValueError(f"Unknown model: {name}")