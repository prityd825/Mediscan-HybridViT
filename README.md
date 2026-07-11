# Mediscan-HybridViT

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/-Lightning-792EE5?style=flat&logo=pytorchlightning&logoColor=white)](https://lightning.ai/)
[![MONAI](https://img.shields.io/badge/MONAI-61D3B5?style=flat)](https://monai.io/)

An interpretable, hybrid deep learning architecture designed for robust medical image classification under severe class imbalance profiles. This framework couples a **MONAI ResNet-10** convolutional backbone with a multi-layer **Vision Transformer (ViT)** encoder to successfully merge localized inductive biases with global self-attention mechanisms.

---

## 🚀 Key Features
* **Hybrid CNN-ViT Topology:** Uses an optimized, shallow CNN backbone to generate a localized intermediate spatial representation ($28 \times 28 \times 256$), which is then flattened into $784$ sequence tokens for global attention mapping.
* **Resilient to Extreme Class Imbalance:** Reaches a stable **0.95 Macro F1-score** on highly skewed medical diagnostic scans (e.g., matching performance on a 42:1 majority-to-minority class skew layout).
* **Native Token Explanations:** Includes diagnostic explainability without relying on external packages by extracting and mapping token activation variances directly from the Transformer layers.
* **Production-Ready Implementation:** Fully implemented inside an organized PyTorch Lightning system tracking explicit sample sizes across distributed and interactive notebook workloads.

---

## 📐 Architecture Overview

```text
Input Image (3 x 224 x 224)
       │
 [MONAI ResNet-10] ──► Extracts spatial features
       │
 Feature Map (256 x 28 x 28)
       │
 [1x1 Conv Proj]  ──► Reshapes & Projects to hidden dimension
       │
 Token Sequence (784 x 256) + Learnable Position Embeddings
       │
 [Transformer Encoder Layer x2] ──► Global Self-Attention mapping
       │
 [Global Average Pooling]
       │
 [MLP Classification Head] ──► Softmax Output Logits (4 Classes)
