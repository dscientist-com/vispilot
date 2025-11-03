# 🧠 VisPilot — Multi-Dataset Image Classification Framework

**VisPilot** is a modular deep learning framework designed for **image classification and computer vision tasks**.  
Built with **Python** and **PyTorch**, it showcases professional-grade development practices in data science and machine learning engineering.

---

## 🔍 Overview

VisPilot supports multiple benchmark datasets and focuses on:

- 🧩 Reproducible and configurable workflows  
- 🧱 Modular architecture for easy extension  
- ⚙️ Clean data handling and preprocessing pipelines  
- 🚀 Scalable training and evaluation across datasets  

This repository is structured and documented to reflect real-world, production-ready machine learning engineering standards.

---

## 🧩 Supported Datasets

| Dataset | Domain | Description |
|----------|---------|-------------|
| **MNIST** | Handwritten digits | Classic benchmark for simple classification |
| **CIFAR-10** | Natural images | 10 classes of everyday objects |
| **Fashion-MNIST** | Apparel images | Visually complex replacement for MNIST |
| **STL-10** | High-resolution natural images | Larger dataset for transfer and semi-supervised learning |

Each dataset module handles automatic download, preprocessing, and DataLoader configuration.

---

## ⚙️ Core Features

- 🧮 **ResNet-based classification** (configurable architectures)  
- 🔄 **Unified training and evaluation** pipelines  
- 📁 **Modular structure** separating data, models, and training logic  
- 🧰 **YAML-based configuration** for reproducible experiments  
- 💾 **Automatic checkpointing** for model saving/loading  
- 💻 Works seamlessly on **CPU** or **CUDA** (GPU)  

---

## 🚀 Quick Start

### 1️⃣ Create and activate environment
```bash
conda create -n vispilot python=3.11
conda activate vispilot

---

## 2️⃣ Install dependencies

pip install -r requirements.txt

## 3️⃣ Set the project path

set PYTHONPATH=%CD%\src

## 4️⃣ Train a model

python scripts/train.py --config configs\mnist.yaml

## 5️⃣ Evaluate a model

python scripts/eval.py --config configs\mnist.yaml --checkpoint models\mnist_resnet18.pth


