# 🩺 Mediscan-HybridViT

> **A Hybrid CNN–Vision Transformer Architecture for Data-Imbalanced Medical Image Classification**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![PyTorch Lightning](https://img.shields.io/badge/PyTorch%20Lightning-792EE5?style=flat&logo=pytorchlightning&logoColor=white)
![MONAI](https://img.shields.io/badge/MONAI-61D3B5?style=flat)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📖 Overview

**Mediscan-HybridViT** is a hybrid deep learning framework that combines the strengths of **Convolutional Neural Networks (CNNs)** and **Vision Transformers (ViTs)** for robust medical image classification.

Traditional CNNs effectively learn local spatial features but struggle to capture long-range dependencies. Vision Transformers model global contextual relationships through self-attention but often require massive datasets for effective training. This project bridges both approaches by using a lightweight **MONAI ResNet-10** backbone to extract discriminative local features before passing them to a **Transformer Encoder** for global feature modeling.

The proposed architecture is specifically designed for **highly imbalanced medical datasets**, achieving excellent classification performance while providing interpretable attention visualizations for clinical analysis.

---

## ✨ Features

- 🧠 Hybrid CNN + Vision Transformer architecture
- 🏥 Optimized for medical image classification
- ⚡ Lightweight MONAI ResNet-10 backbone
- 🔍 Multi-head self-attention for global context learning
- 📊 Robust against severe class imbalance
- 📈 Native transformer attention visualization
- 🚀 Implemented with PyTorch Lightning
- 🔬 Easy to extend for research purposes

---

## 🏗️ Model Architecture

```
Input Image (3 × 224 × 224)
           │
           ▼
   MONAI ResNet-10
 Feature Extraction
           │
           ▼
Feature Map (256 × 28 × 28)
           │
           ▼
 1×1 Convolution Projection
           │
           ▼
 Flatten into 784 Tokens
           │
           ▼
 Position Embeddings
           │
           ▼
 Transformer Encoder ×2
 Multi-Head Self-Attention
           │
           ▼
 Global Average Pooling
           │
           ▼
  MLP Classification Head
           │
           ▼
   Softmax (4 Classes)
```

---

## 📊 Experimental Results

The proposed model was evaluated on a multi-class medical imaging dataset containing four diagnostic categories:

- 🦴 Bone
- 🧠 Brain
- ❤️ Heart
- 🫁 Lungs

### Performance

| Metric | Score |
|--------|-------|
| Validation Accuracy | **95.8%** |
| Macro F1-Score | **0.95** |
| Weighted F1-Score | **0.96** |
| Precision | **0.95** |
| Recall | **0.95** |

Despite an extreme class imbalance (greater than **40:1**), the model maintains excellent performance across all classes.

---

## 🔬 Explainability

Unlike conventional CNN-based classifiers, **Mediscan-HybridViT** provides native interpretability by analyzing transformer token activations.

The generated attention maps highlight clinically relevant anatomical regions including:

- Brain lesion boundaries
- Pulmonary structures
- Bone morphology
- Cardiac regions

These visualizations help verify that the network focuses on meaningful pathological features rather than irrelevant background artifacts.

---

## 📂 Project Structure

```
Mediscan-HybridViT/
│
├── datasets/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── models/
│   ├── hybrid_vit.py
│   └── modules.py
│
├── notebooks/
│
├── checkpoints/
│
├── utils/
│
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/prityd825/Mediscan-HybridViT.git

cd Mediscan-HybridViT
```

Create a virtual environment (recommended):

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch torchvision monai pytorch-lightning matplotlib numpy pandas scikit-learn opencv-python
```

---

## 🚀 Training

Train the model using PyTorch Lightning.

```python
from pytorch_lightning import Trainer

trainer = Trainer(
    max_epochs=10,
    accelerator="auto",
    devices=1
)

trainer.fit(model, datamodule=dm)
```

Or simply execute:

```bash
python train.py
```

---

## 📈 Evaluation

Evaluate a trained model checkpoint:

```bash
python evaluate.py --checkpoint checkpoints/model.ckpt
```

---

## 🛠️ Technologies Used

- Python
- PyTorch
- PyTorch Lightning
- MONAI
- Torchvision
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

---

## 🚧 Future Improvements

- Support additional medical imaging modalities
- Multi-label disease classification
- Grad-CAM integration
- Attention rollout visualization
- ONNX model export
- TensorRT optimization
- Distributed multi-GPU training

---


## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "Add new feature"
```

4. Push to your branch

```bash
git push origin feature-name
```

5. Open a Pull Request

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 👩‍💻 Author

**Prity Rani Das**

Software Engineer

- GitHub: https://github.com/prityd825

---

## ⭐ Support

If you found this repository helpful, please consider giving it a **⭐ Star**. It helps others discover the project and supports future development.
