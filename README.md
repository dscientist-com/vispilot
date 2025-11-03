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
## 📁 Project Structure

VisPilot/
│
├── configs/                # YAML configuration files for datasets
├── data/                   # Auto-downloaded datasets (ignored in .git)
├── models/                 # Saved model checkpoints
├── scripts/                # CLI tools: train, eval, serve, predict
├── src/vispilot/           # Core library (data, models, engine, utils)
├── requirements.txt        # Dependencies
├── LICENSE                 # MIT License
└── README.md               # Project documentation

## 📊 Example Results

| Dataset       | Accuracy |
| ------------- | -------- |
| MNIST         | 99.1%    |
| CIFAR-10      | 87.1%    |
| Fashion-MNIST | 91.5%    |
| STL-10        | 88.1%    |

Results may vary slightly depending on hardware and random initialization.

## 🧠 Model Architecture

All current experiments use ResNet-18, chosen for its strong performance-to-speed ratio.
The modular design allows you to quickly replace it with EfficientNet, ResNet-50, or Vision Transformers (ViT) by editing one line in the configuration.

## 🧩 Extending the Framework

To add a new dataset or model:

Create a new data loader in src/vispilot/data/ with a build() function.

Register your model in src/vispilot/models/classifiers.py.

Add a new configuration file under configs/.

Train and evaluate using the existing CLI commands.

This modular structure allows VisPilot to scale from academic research to enterprise-grade ML pipelines.


## 🌐 Using VisPilot as an API

Once trained, models can be served to the public via REST APIs or interactive web apps.

Recommended stack:

Backend: FastAPI or Flask (Python)

Frontend: React, Next.js, or simple HTML/JS form

Deployment: Docker, Render, Railway, or AWS/GCP/Azure


## Example output (JSON):

{
  "dataset": "cifar10",
  "topk": [
    {"label": "cat", "prob": 0.82},
    {"label": "dog", "prob": 0.12},
    {"label": "deer", "prob": 0.03}
  ],
  "inference_ms": 24
}


## 🧾 License

This project is released under the MIT License.
You are free to use, modify, and distribute it for both commercial and non-commercial purposes.

## 👨‍💻 Author

# Atif Majeed

📊 Data Scientist | Machine Learning Enthusiast | Transforming Data into Insights | SQL • Tableau • Excel

🌐 [GitHub Profile](https://github.com/dscientist-com)

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
