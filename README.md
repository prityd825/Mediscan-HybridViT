# 🩺 Mediscan-HybridViT

> **A Dual-Architecture Framework for Data-Imbalanced Medical Image Classification Using a Hybrid CNN–Transformer and DenseNet-121 Baseline**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat\&logo=pytorch\&logoColor=white)
![PyTorch Lightning](https://img.shields.io/badge/PyTorch%20Lightning-792EE5?style=flat\&logo=pytorchlightning\&logoColor=white)
![MONAI](https://img.shields.io/badge/MONAI-Medical%20Imaging-61D3B5?style=flat)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📖 Overview

**Mediscan-HybridViT** is a deep learning project for multi-class medical image classification. The repository implements **two different model architectures** to explore and compare CNN-based and hybrid CNN–Transformer approaches.

### 🧠 Model 1 — Mediscan-HybridViT

A custom hybrid architecture that combines:

* **MONAI ResNet-10** for local feature extraction
* **1×1 convolutional projection**
* **Transformer Encoder** for global spatial feature modeling
* **MLP classification head** for four-class prediction

### 🏥 Model 2 — DenseNet-121 Baseline

A MONAI **DenseNet-121** classifier used as a baseline, enhanced with **class-weighted Cross-Entropy Loss** to address the imbalanced class distribution.

The two pipelines provide different approaches to the same medical image classification task and can be used for comparative experimentation.

---

## 📊 Dataset

The project uses the **MediScan Dataset** from Kaggle.

**Dataset Link:**
https://www.kaggle.com/datasets/pritydas825/mediscan-dataset

The classification task contains four categories:

* 🦴 **Bone**
* 🧠 **Brain**
* ❤️ **Heart**
* 🫁 **Lungs**

The implementation collects supported image formats including:

* `.png`
* `.jpg`
* `.jpeg`
* `.webp`

The collected data is split into:

* **80% Training**
* **20% Validation**

using `train_test_split` with `random_state=42`.

> **Note:** The current notebook constructs its own training and validation split from the configured dataset directory.

---

## ✨ Features

* 🧠 Two medical image classification architectures
* 🔬 Hybrid CNN–Transformer model
* 🏥 DenseNet-121 baseline model
* 🎨 Automatic image channel normalization
* 📏 Image resizing to `224 × 224`
* 🔄 MONAI-based data augmentation
* ⚖️ Class-weighted loss for the DenseNet-121 baseline
* 🔍 Transformer-based global context modeling
* 👁️ Token activation visualization for the HybridViT model
* 📊 Classification report and confusion matrix generation
* 📈 PyTorch Lightning training and validation
* 💾 Model checkpointing support

---

# 🏗️ Model Architectures

## 🧠 Model 1: Mediscan-HybridViT

The primary model combines a convolutional backbone with a Transformer Encoder.

```text
Input Image
(3 × 224 × 224)
        │
        ▼
┌──────────────────────┐
│    MONAI ResNet-10   │
│ Local Feature Learning│
└──────────────────────┘
        │
        ▼
Feature Map
(256 × 28 × 28)
        │
        ▼
1×1 Convolution Projection
        │
        ▼
Flatten into 784 Spatial Tokens
        │
        ▼
Learnable Positional Embeddings
        │
        ▼
Transformer Encoder ×2
Multi-Head Self-Attention
        │
        ▼
Global Mean Pooling
        │
        ▼
LayerNorm + Linear Layer
        │
        ▼
Output: 4 Classes
```

### Architecture Details

The model is configured with:

| Component              | Configuration      |
| ---------------------- | ------------------ |
| CNN Backbone           | MONAI ResNet-10    |
| Input Channels         | 3                  |
| Feature Map            | `256 × 28 × 28`    |
| Projection             | `1×1 Conv2D`       |
| Number of Tokens       | `28 × 28 = 784`    |
| Hidden Dimension       | 256                |
| Attention Heads        | 4                  |
| Transformer Layers     | 2                  |
| Transformer Activation | GELU               |
| Dropout                | 0.1                |
| Classifier             | LayerNorm + Linear |
| Output Classes         | 4                  |

### How It Works

1. The **ResNet-10 backbone** extracts local spatial features from the input image.
2. The resulting feature map is projected using a **1×1 convolution**.
3. The `28 × 28` spatial feature map is flattened into **784 tokens**.
4. Learnable positional embeddings are added to preserve spatial information.
5. A two-layer Transformer Encoder processes the token sequence using multi-head self-attention.
6. Global mean pooling aggregates the token representations.
7. A classification head produces logits for the four diagnostic categories.

---

## 🏥 Model 2: DenseNet-121 Baseline

The second model uses MONAI's **DenseNet-121** architecture as a convolutional baseline.

```text
Input Image
(3 × 224 × 224)
        │
        ▼
┌──────────────────────┐
│   MONAI DenseNet-121 │
│ Dense Feature Learning│
└──────────────────────┘
        │
        ▼
Weighted Cross-Entropy Loss
        │
        ▼
Output: 4 Classes
```

### Class-Weighted Loss

The DenseNet-121 model uses the following class weights:

```python
[4.0, 20.0, 1.0, 0.5]
```

The weights are registered as a PyTorch buffer so they automatically move with the model between CPU and GPU.

```python
weights = torch.tensor(
    [4.0, 20.0, 1.0, 0.5],
    dtype=torch.float
)

self.register_buffer(
    "class_weights",
    weights
)

self.loss_fn = nn.CrossEntropyLoss(
    weight=self.class_weights
)
```

This strategy increases the training penalty for errors involving classes assigned higher weights.

---

# 🎨 Image Preprocessing

Medical images can contain different numbers of channels. The project includes custom preprocessing functions to ensure that all images have a consistent **three-channel input format**.

### Channel Handling

```text
Grayscale Image (1 Channel)
          │
          ▼
Repeat Channel → 3 Channels

RGB Image (3 Channels)
          │
          ▼
Keep Original Format

RGBA / Multi-Channel Image (>3 Channels)
          │
          ▼
Keep First 3 Channels
```

Two channel-processing functions are used:

* `ensure_rgb()` for the HybridViT pipeline
* `force_three_channels()` for the DenseNet-121 pipeline

All images are resized to:

```text
224 × 224
```

and intensity scaling is applied before model training.

---

# 🔄 Data Augmentation

## HybridViT Pipeline

Training images use:

* `RandRotate90d` with probability `0.5`
* `RandFlipd` with probability `0.5`

Validation images are resized and normalized without random augmentation.

## DenseNet-121 Pipeline

Training images use:

* `RandFlipd` with probability `0.5`
* `RandRotated` with probability `0.5`

The DenseNet pipeline also uses MONAI's `list_data_collate` function for DataLoader batching.

---

# ⚙️ Training Configuration

## 🧠 HybridViT Training

The HybridViT model is trained using PyTorch Lightning with the following configuration:

| Parameter          | Value     |
| ------------------ | --------- |
| Batch Size         | 16        |
| Learning Rate      | `1e-4`    |
| Optimizer          | AdamW     |
| Weight Decay       | `1e-4`    |
| Maximum Epochs     | 10        |
| Accelerator        | Auto      |
| Checkpoint Monitor | `val_acc` |

The best checkpoint is selected based on validation accuracy.

```python
checkpoint_callback = ModelCheckpoint(
    monitor="val_acc",
    mode="max",
    filename="best-hybrid-mediscan",
    save_top_k=1
)
```

---

## 🏥 DenseNet-121 Training

The DenseNet-121 baseline uses:

| Parameter      | Value  |
| -------------- | ------ |
| Batch Size     | 16     |
| Learning Rate  | `2e-4` |
| Optimizer      | AdamW  |
| Weight Decay   | `1e-5` |
| Maximum Epochs | 15     |
| Accelerator    | Auto   |
| Devices        | 1      |

---

# 📈 Evaluation

## HybridViT Evaluation

The HybridViT pipeline performs explicit evaluation on the validation DataLoader and generates:

* Classification predictions
* Scikit-learn classification report
* Confusion matrix

The confusion matrix is visualized using Seaborn.

```python
cm = confusion_matrix(
    all_labels,
    all_preds
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names
)
```

---

## DenseNet-121 Evaluation

The DenseNet-121 model is evaluated using the PyTorch Lightning validation pipeline:

```python
val_results = trainer.validate(
    model=model,
    dataloaders=val_loader,
    ckpt_path="best"
)
```

The current implementation logs:

* Training Loss
* Training Accuracy
* Validation Loss
* Validation Accuracy

The code also initializes macro Precision, Recall, and F1-score metric objects for future extension.

---

# 👁️ HybridViT Visualization

The HybridViT model includes a custom visualization pipeline that selects samples from the validation data and displays predictions alongside token activation overlays.

### Visualization Workflow

```text
Validation Image
       │
       ▼
ResNet-10 Feature Extraction
       │
       ▼
Feature Projection
       │
       ▼
Transformer Encoder Output
       │
       ▼
Variance Across Token Features
       │
       ▼
28 × 28 Activation Map
       │
       ▼
Resize to 224 × 224
       │
       ▼
OpenCV JET Heatmap
       │
       ▼
Overlay on Original Image
```

The visualization displays:

* Original image
* True class
* Predicted class
* Softmax confidence score
* Match or mismatch status
* Token activation overlay

The visualization function selects samples representing the available classes and produces a qualitative prediction summary.

> **Important:** The current heatmap is calculated using the variance of Transformer token representations. It is an activation-based visualization and is **not a direct visualization of Transformer attention weights** or a clinically validated saliency method.

---

# 🛠️ Technologies Used

| Category             | Technologies               |
| -------------------- | -------------------------- |
| Programming Language | Python                     |
| Deep Learning        | PyTorch                    |
| Training Framework   | PyTorch Lightning          |
| Medical Imaging      | MONAI                      |
| Metrics              | TorchMetrics, Scikit-learn |
| Computer Vision      | OpenCV                     |
| Visualization        | Matplotlib, Seaborn        |
| Data Processing      | NumPy, Pandas              |

---

# 📦 Installation

## Clone the Repository

```bash
git clone https://github.com/prityd825/Mediscan-HybridViT.git
cd Mediscan-HybridViT
```

## Create a Virtual Environment

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

## Install Dependencies

```bash
pip install torch torchvision monai pytorch-lightning torchmetrics matplotlib numpy pandas scikit-learn opencv-python seaborn
```

---

# 🚀 Usage

The current implementation is provided as a Jupyter Notebook containing the complete workflow for:

1. Dataset preparation
2. Image preprocessing
3. HybridViT training
4. HybridViT evaluation
5. Confusion matrix generation
6. Activation visualization
7. DenseNet-121 baseline training
8. DenseNet-121 validation

Open and run the notebook in Jupyter or Kaggle:

```bash
jupyter notebook
```

Then execute the notebook cells in order.

---

# 🔍 Model Comparison

| Feature                  | Mediscan-HybridViT                       | DenseNet-121 Baseline     |
| ------------------------ | ---------------------------------------- | ------------------------- |
| Architecture             | ResNet-10 + Transformer                  | DenseNet-121              |
| Local Feature Learning   | ResNet-10                                | Dense Connections         |
| Global Context Modeling  | Multi-Head Self-Attention                | CNN Feature Propagation   |
| Class Imbalance Handling | Data Augmentation                        | Weighted Cross-Entropy    |
| Input Size               | 224 × 224                                | 224 × 224                 |
| Training Framework       | PyTorch Lightning                        | PyTorch Lightning         |
| Evaluation               | Classification Report + Confusion Matrix | Lightning Validation      |
| Visualization            | Token Activation Overlay                 | Not Currently Implemented |

---

# 🚧 Future Improvements

* [ ] Add Grad-CAM for DenseNet-121
* [ ] Log Precision, Recall, and F1-score during DenseNet validation
* [ ] Add direct Transformer attention extraction or attention rollout
* [ ] Compare both models using a common evaluation protocol
* [ ] Add per-class performance comparison
* [ ] Add ROC/AUC analysis where appropriate
* [ ] Export trained models to ONNX
* [ ] Support distributed multi-GPU training
* [ ] Extend the framework to 3D medical images such as CT and MRI volumes

---


# 👩‍💻 Author

**Prity Rani Das**

* **GitHub:** https://github.com/prityd825

---

⭐ **If you find this project useful, please consider giving the repository a star!**
