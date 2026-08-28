# 🩺 Mediscan-HybridViT & DenseNet-121 Baseline

> **A Dual-Architecture Framework for Data-Imbalanced Medical Image Classification Using Hybrid CNN–Transformer and Weighted DenseNet-121 Models**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat\&logo=pytorch\&logoColor=white)
![PyTorch Lightning](https://img.shields.io/badge/PyTorch%20Lightning-792EE5?style=flat\&logo=pytorchlightning\&logoColor=white)
![MONAI](https://img.shields.io/badge/MONAI-Medical%20Imaging-61D3B5?style=flat)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📊 Dataset

This project uses the **MediScan Dataset**, containing four medical imaging categories:

* 🦴 **Bone**
* 🧠 **Brain**
* ❤️ **Heart**
* 🫁 **Lungs**

**Dataset Link:**
https://www.kaggle.com/datasets/pritydas825/mediscan-dataset

---

## 📖 Overview

**Mediscan-HybridViT** is a dual-model deep learning framework for multi-class medical image classification.

The project implements and compares two different architectures:

### 🧠 1. Mediscan-HybridViT — Hybrid CNN–Transformer

A hybrid architecture that combines **MONAI ResNet-10** for local feature extraction with a **Transformer Encoder** for global context modeling.

### 🏥 2. MedicalClassifier — DenseNet-121 Baseline

A CNN-based baseline built with **MONAI DenseNet-121**, enhanced with **class-weighted Cross-Entropy Loss** to address severe class imbalance.

The framework is designed to explore the advantages of combining local convolutional features with global Transformer-based representations while comparing them with a strong DenseNet-121 baseline.

---

## ✨ Key Features

* 🧠 Dual-model architecture for comparative experiments
* 🔬 Hybrid CNN + Vision Transformer pipeline
* 🏥 DenseNet-121 baseline model
* ⚡ MONAI-based medical image preprocessing
* 🎨 Automatic RGB channel normalization
* 📊 Class imbalance handling with weighted loss
* 🔍 Multi-head self-attention for global feature modeling
* 👁️ Token activation visualization for the HybridViT model
* 📈 Accuracy, Precision, Recall, and F1-score evaluation
* 🚀 PyTorch Lightning training pipelines
* 💾 Automatic checkpoint management

---

# 🏗️ Model Architectures

## 🧠 Model 1: Mediscan-HybridViT

The proposed HybridViT architecture combines a CNN backbone with Transformer-based global context modeling.

```text
Input Image (3 × 224 × 224)
            │
            ▼
   MONAI ResNet-10
  Local Feature Extraction
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
Learnable Position Embeddings
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
 Output (4 Classes)
```

### Architecture Components

* **Backbone:** MONAI ResNet-10
* **Feature Map:** `256 × 28 × 28`
* **Projection Layer:** `1×1` Convolution
* **Number of Tokens:** `28 × 28 = 784`
* **Hidden Dimension:** `256`
* **Transformer Layers:** `2`
* **Attention Heads:** `4`
* **Classifier:** LayerNorm + Linear Layer
* **Loss Function:** Cross-Entropy Loss
* **Optimizer:** AdamW

The CNN extracts local hierarchical features, while the Transformer learns relationships between different spatial regions of the medical image.

---

## 🏥 Model 2: MedicalClassifier — DenseNet-121 Baseline

The second pipeline uses MONAI's **DenseNet-121** as a baseline for comparison.

```text
Input Image (3 × 224 × 224)
            │
            ▼
    MONAI DenseNet-121
 Dense Feature Connections
            │
            ▼
 Weighted Cross-Entropy Loss
 [4.0, 20.0, 1.0, 0.5]
            │
            ▼
 TorchMetrics Evaluation
 Accuracy • Precision
 Recall • F1-Score
            │
            ▼
 Output (4 Classes)
```

### Class Imbalance Strategy

The DenseNet-121 model uses class-specific weights:

```python
weights = torch.tensor([4.0, 20.0, 1.0, 0.5])
loss_fn = nn.CrossEntropyLoss(weight=weights)
```

These weights increase the loss penalty for errors on underrepresented classes.

The class weights are registered as a buffer so they automatically move between CPU and GPU together with the model.

---

# 🎨 Image Preprocessing

Medical images may contain different numbers of channels. The project includes custom functions to ensure that all images are converted into a consistent **3-channel RGB format**.

### Channel Handling

```text
Grayscale Image (1 Channel) ──► Convert to 3 Channels
RGB Image (3 Channels) ───────► Keep Original
RGBA Image (4 Channels) ──────► Remove Alpha Channel
```

Two channel normalization functions are used across the pipelines:

* `ensure_rgb`
* `force_three_channels`

All images are resized to:

```text
224 × 224
```

and intensity values are scaled before training.

---

# 🔄 Data Augmentation

The project uses MONAI transformations to improve model robustness.

### HybridViT Pipeline

* `RandRotate90d`
* `RandFlipd`

### DenseNet-121 Pipeline

* `RandFlipd`
* `RandRotated`

These augmentations introduce controlled variation during training and help improve generalization.

---

# 📊 Dataset Preparation

The dataset contains images from the following four classes:

| Index | Class |
| ----: | ----- |
|     0 | Bone  |
|     1 | Brain |
|     2 | Heart |
|     3 | Lungs |

The available data is split into training and validation sets using:

```python
train_test_split(
    data_dicts,
    test_size=0.2,
    random_state=42
)
```

This creates an **80% training split** and a **20% validation split**.

---

# 📈 Evaluation & Metrics

The models are evaluated using classification metrics appropriate for multi-class classification.

### Metrics

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
* **Classification Report**
* **Confusion Matrix**

The HybridViT pipeline generates a Scikit-learn classification report and confusion matrix.

The DenseNet-121 pipeline uses TorchMetrics for training and validation accuracy and defines macro Precision, Recall, and F1-score metrics for evaluation.

---

# 👁️ Explainability & Visualization

The **Mediscan-HybridViT** model includes a visualization pipeline based on Transformer token representations.

The visualization process:

1. Extracts features from the ResNet-10 backbone.
2. Projects features using the `1×1` convolution layer.
3. Converts the feature map into spatial tokens.
4. Processes the tokens through the Transformer Encoder.
5. Calculates the variance across each token's feature dimension.
6. Reshapes the values into a `28 × 28` activation map.
7. Resizes the map to `224 × 224`.
8. Generates an OpenCV `COLORMAP_JET` heatmap.
9. Overlays the heatmap on the original image.

The visualization also displays:

* True class
* Predicted class
* Prediction confidence
* Match or mismatch status

> **Note:** The current visualization is based on Transformer token activation variance. It is not a direct extraction of Transformer attention weights or a clinically validated saliency method.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/prityd825/Mediscan-HybridViT.git
cd Mediscan-HybridViT
```

## 2. Create a Virtual Environment

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

## 3. Install Dependencies

If a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

Or install the required packages manually:

```bash
pip install torch torchvision monai pytorch-lightning torchmetrics matplotlib numpy pandas scikit-learn opencv-python seaborn
```

---

# 🚀 Training

## Train the Mediscan-HybridViT Model

```python
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

dm = MediscanDataModule(
    train_files,
    val_files,
    batch_size=16
)

model = HybridLightningModule(
    num_classes=4,
    lr=1e-4
)

checkpoint_callback = ModelCheckpoint(
    monitor="val_acc",
    mode="max",
    filename="best-hybrid-mediscan",
    save_top_k=1
)

trainer = pl.Trainer(
    max_epochs=10,
    accelerator="auto",
    callbacks=[checkpoint_callback],
    log_every_n_steps=5
)

trainer.fit(
    model,
    datamodule=dm
)
```

---

## Train the DenseNet-121 Baseline

```python
model = MedicalClassifier(
    num_classes=4,
    lr=2e-4
)

trainer = pl.Trainer(
    max_epochs=15,
    accelerator="auto",
    devices=1,
    enable_checkpointing=True,
    log_every_n_steps=10
)

trainer.fit(
    model,
    train_loader,
    val_loader
)
```

---

# 📈 Evaluation

## HybridViT Evaluation

The HybridViT model can be evaluated using the validation DataLoader to generate:

* Classification report
* Confusion matrix
* Per-class predictions

```python
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in dm.val_dataloader():
        images = batch["image"].to(device)
        labels = batch["label"]

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
```

## DenseNet-121 Evaluation

Evaluate the DenseNet-121 model using the best available checkpoint:

```python
val_results = trainer.validate(
    model=model,
    dataloaders=val_loader,
    ckpt_path="best"
)

print(val_results)
```

---

# 👁️ Generate HybridViT Activation Visualizations

Generate balanced samples from all four classes and visualize model predictions:

```python
visualize_predictions_with_attention(
    model,
    dm,
    num_samples=4
)
```

The visualization displays the original image alongside its activation heatmap and prediction confidence.

---

# 🛠️ Technologies Used

| Category                 | Technologies               |
| ------------------------ | -------------------------- |
| **Programming Language** | Python                     |
| **Deep Learning**        | PyTorch                    |
| **Training Framework**   | PyTorch Lightning          |
| **Medical Imaging**      | MONAI                      |
| **Evaluation**           | TorchMetrics, Scikit-learn |
| **Computer Vision**      | OpenCV                     |
| **Visualization**        | Matplotlib, Seaborn        |
| **Data Processing**      | NumPy, Pandas              |

---

# 🔍 Model Comparison

| Feature                  | Mediscan-HybridViT             | DenseNet-121 Baseline      |
| ------------------------ | ------------------------------ | -------------------------- |
| Architecture             | ResNet-10 + Transformer        | DenseNet-121               |
| Local Feature Extraction | ResNet-10                      | Dense Convolutional Blocks |
| Global Context Learning  | Transformer Self-Attention     | Dense Feature Connections  |
| Class Imbalance Strategy | Data Augmentation              | Weighted Cross-Entropy     |
| Explainability           | Token Activation Visualization | Not Yet Implemented        |
| Training Framework       | PyTorch Lightning              | PyTorch Lightning          |
| Input Size               | 224 × 224                      | 224 × 224                  |

---

# 🚧 Future Improvements

* [ ] Add Grad-CAM visualization for the DenseNet-121 baseline
* [ ] Compare Grad-CAM with HybridViT token activation visualizations
* [ ] Add direct Transformer attention visualization or attention rollout
* [ ] Perform detailed side-by-side performance comparison between both models
* [ ] Support additional medical imaging modalities
* [ ] Add multi-label disease classification
* [ ] Export trained models to ONNX
* [ ] Add TensorRT optimization
* [ ] Implement distributed multi-GPU training
* [ ] Extend the framework to support 3D CT and MRI volumes

---

# 🤝 Contributing

Contributions are welcome! 🎉

1. Fork the repository.
2. Create a feature branch:

```bash
git checkout -b feature-name
```

3. Commit your changes:

```bash
git commit -m "Add new feature"
```

4. Push your branch:

```bash
git push origin feature-name
```

5. Open a Pull Request.

---

# 👩‍💻 Author

**Prity Rani Das**

* **GitHub:** https://github.com/prityd825

---

⭐ **If you find this project useful, please consider giving the repository a star!**
