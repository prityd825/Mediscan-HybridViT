# 🩺 Mediscan-HybridViT

> **Multi-Class Medical Image Classification Using a Hybrid CNN–Transformer, DenseNet-121, and Ensemble Decision Strategies**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=flat\&logo=pytorch\&logoColor=white)](https://pytorch.org/)
[![PyTorch Lightning](https://img.shields.io/badge/PyTorch%20Lightning-Training-792EE5?style=flat\&logo=pytorchlightning\&logoColor=white)](https://lightning.ai/)
[![MONAI](https://img.shields.io/badge/MONAI-Medical%20Imaging-61D3B5?style=flat)](https://monai.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 Overview

**Mediscan-HybridViT** is a notebook-based deep learning research project for **multi-class medical image classification** across four image categories:

* 🦴 **Bone**
* 🧠 **Brain**
* ❤️ **Heart**
* 🫁 **Lungs**

The project investigates whether combining **local feature learning from convolutional neural networks** with **global contextual modeling from Transformer encoders** can provide complementary information for medical image classification.

Two standalone architectures are implemented and evaluated:

1. **Mediscan-HybridViT** — a custom hybrid CNN–Transformer architecture based on MONAI ResNet-10.
2. **DenseNet-121** — a convolutional baseline based on MONAI DenseNet-121.

Beyond standalone model comparison, the project also explores several **ensemble and decision-making strategies**, including:

* Equal Soft Voting
* Temperature-Calibrated Voting
* Specialist ViT Override
* 2-Tier Cascade Classification

The experiments are implemented through Jupyter Notebooks rather than a Python package or modular training framework.

---

# 🎯 Research Motivation

Medical image datasets can contain substantial **class imbalance**, while different neural network architectures may learn complementary visual representations.

This project investigates three main questions:

1. **Can a hybrid CNN–Transformer architecture effectively classify multiple medical image categories?**
2. **How does the HybridViT compare with a CNN-based DenseNet-121 baseline?**
3. **Can combining the predictions of both models improve classification performance compared with using a single model?**

The ensemble experiments further investigate whether **model specialization and hierarchical decision-making** can be more effective than simply averaging model probabilities.

---

# ✨ Key Highlights

* 🧠 Custom **Hybrid CNN–Transformer** architecture
* 🏥 **MONAI DenseNet-121** baseline
* 🔬 Multi-class classification across four medical image categories
* ⚖️ Class-weighted Cross-Entropy Loss for the DenseNet-121 pipeline
* 🔄 MONAI-based image augmentation
* 📏 Standardized `224 × 224` image input
* 📊 Classification reports and confusion matrices
* 👁️ Transformer token activation visualization
* 🤝 Multiple ensemble strategies
* 🌡️ Temperature-based probability calibration experiment
* 🎯 Specialist routing / model override experiment
* 🪜 2-Tier cascade classification experiment
* 🔍 Sample-level error analysis
* 📈 Comparative evaluation on a held-out test set

---

# 📊 Main Experimental Results

The models and ensemble strategies were evaluated on a test set containing **315 samples** across four classes.

### Test Set Distribution

| Class     | Number of Samples |
| --------- | ----------------: |
| Bone      |                64 |
| Brain     |               215 |
| Heart     |                 5 |
| Lungs     |                31 |
| **Total** |           **315** |

> ⚠️ The test set is highly imbalanced, particularly because the Brain class contains substantially more samples than the Heart class. Therefore, accuracy should be interpreted together with per-class precision, recall, and F1-score where available.

## Performance Comparison

| Model / Strategy              | Test Accuracy | Correct / Total | Bone Recall | Lungs Recall |
| ----------------------------- | ------------: | --------------: | ----------: | -----------: |
| **HybridViT Baseline**        |        98.10% |       309 / 315 |      95.31% |   **96.77%** |
| **DenseNet-121 Baseline**     |    **99.05%** |   **312 / 315** | **100.00%** |       90.32% |
| Equal Soft Voting             |        98.41% |       310 / 315 |      96.88% |       90.32% |
| Temperature-Calibrated Voting |        98.41% |       310 / 315 |      96.88% |       90.32% |
| Specialist ViT Override       |        98.73% |       311 / 315 |      95.31% |   **96.77%** |
| **2-Tier Cascade**            |    **99.05%** |   **312 / 315** | **100.00%** |       90.32% |

### Key Result

The **DenseNet-121 baseline** and the **2-Tier Cascade strategy** achieved the highest observed test accuracy:

> **99.05% — 312 correct predictions out of 315**

The HybridViT achieved:

> **98.10% — 309 correct predictions out of 315**

The experiments therefore show that the HybridViT and DenseNet-121 have **different strengths**, even though the final cascade strategy did not exceed the best standalone DenseNet-121 accuracy on this test set.

---

# 🏗️ Model Architectures

## 🧠 Model 1 — Mediscan-HybridViT

The primary model combines a **MONAI ResNet-10 convolutional backbone** with a **Transformer Encoder**.

### Architecture

```text
                    Input Image
                  3 × 224 × 224
                         │
                         ▼
              ┌────────────────────┐
              │   MONAI ResNet-10   │
              │ Local Feature       │
              │ Extraction          │
              └──────────┬─────────┘
                         │
                         ▼
                 Feature Map
                  256 × 28 × 28
                         │
                         ▼
                 1 × 1 Conv2D
                   Projection
                         │
                         ▼
                 784 Spatial Tokens
                    28 × 28
                         │
                         ▼
             Learnable Positional
                  Embeddings
                         │
                         ▼
              Transformer Encoder
                     × 2
                         │
                 Multi-Head
                Self-Attention
                         │
                         ▼
                Global Mean Pooling
                         │
                         ▼
                 LayerNorm + Linear
                         │
                         ▼
                    4 Classes
```

### Configuration

| Component          | Configuration      |
| ------------------ | ------------------ |
| CNN Backbone       | MONAI ResNet-10    |
| Input Channels     | 3                  |
| Input Resolution   | `224 × 224`        |
| Feature Map        | `256 × 28 × 28`    |
| Projection         | `1 × 1 Conv2D`     |
| Spatial Tokens     | `784`              |
| Hidden Dimension   | `256`              |
| Attention Heads    | `4`                |
| Transformer Layers | `2`                |
| Activation         | GELU               |
| Dropout            | `0.1`              |
| Classifier         | LayerNorm + Linear |
| Output Classes     | `4`                |

### Processing Flow

1. ResNet-10 extracts local visual features.
2. A `1 × 1` convolution projects the feature representation.
3. The `28 × 28` feature map is converted into **784 spatial tokens**.
4. Learnable positional embeddings provide spatial information.
5. Transformer Encoder layers model long-range relationships between image regions.
6. Global mean pooling aggregates the token representations.
7. A classification head produces logits for the four classes.

---

# 🏥 Model 2 — DenseNet-121

The second architecture uses **MONAI DenseNet-121** as a CNN-based baseline.

```text
                    Input Image
                  3 × 224 × 224
                         │
                         ▼
              ┌────────────────────┐
              │   MONAI DenseNet    │
              │       -121          │
              │ Dense Feature       │
              │ Learning            │
              └──────────┬─────────┘
                         │
                         ▼
                Classification
                      Logits
                         │
                         ▼
             Weighted Cross-Entropy
                         │
                         ▼
                    4 Classes
```

DenseNet provides extensive feature reuse through dense connections, allowing information and gradients to flow efficiently throughout the network.

---

# ⚖️ Class Imbalance Handling

The DenseNet-121 training pipeline uses **class-weighted Cross-Entropy Loss**.

The configured class weights are:

```python
[4.0, 20.0, 1.0, 0.5]
```

The weights are registered as a PyTorch buffer:

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

Higher weights increase the penalty associated with misclassifying classes assigned greater importance during training.

This is particularly relevant because the dataset contains a highly uneven number of samples among the four categories.

---

# 🎨 Image Preprocessing

Medical images may contain different channel configurations.

The notebooks therefore include custom preprocessing functions to standardize images into a **three-channel format**.

### Channel Handling

```text
Grayscale
1 Channel
    │
    ▼
Repeat Channel
    │
    ▼
3 Channels


RGB
3 Channels
    │
    ▼
Keep Original


RGBA / Multi-Channel
>3 Channels
    │
    ▼
Keep First 3 Channels
```

Two preprocessing functions are used:

* `ensure_rgb()` — HybridViT pipeline
* `force_three_channels()` — DenseNet-121 pipeline

Images are resized to:

```text
224 × 224
```

and intensity normalization/scaling is applied before model input.

---

# 🔄 Data Augmentation

## HybridViT

The HybridViT training pipeline applies:

* `RandRotate90d` with probability `0.5`
* `RandFlipd` with probability `0.5`

Validation data does not use random augmentation.

## DenseNet-121

The DenseNet-121 pipeline applies:

* `RandFlipd` with probability `0.5`
* `RandRotated` with probability `0.5`

The DenseNet pipeline also uses MONAI's:

```python
list_data_collate
```

for DataLoader batching.

---

# ⚙️ Training Configuration

## HybridViT

| Parameter          | Value     |
| ------------------ | --------- |
| Batch Size         | 16        |
| Learning Rate      | `1e-4`    |
| Optimizer          | AdamW     |
| Weight Decay       | `1e-4`    |
| Maximum Epochs     | 10        |
| Accelerator        | Auto      |
| Checkpoint Monitor | `val_acc` |

The best HybridViT checkpoint is selected according to validation accuracy.

```python
checkpoint_callback = ModelCheckpoint(
    monitor="val_acc",
    mode="max",
    filename="best-hybrid-mediscan",
    save_top_k=1
)
```

## DenseNet-121

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

# 🤝 Ensemble Experiments

One of the main extensions of the project is the investigation of **prediction-level ensemble strategies**.

The goal was not simply to average predictions, but to determine whether the two architectures could be used according to their respective strengths.

---

## 1️⃣ Equal Soft Voting

The first ensemble strategy combines the probability outputs of the two models.

```text
              Input Image
                   │
          ┌────────┴────────┐
          │                 │
          ▼                 ▼
     HybridViT          DenseNet-121
          │                 │
          ▼                 ▼
   Class Probabilities Class Probabilities
          │                 │
          └────────┬────────┘
                   │
                   ▼
             Average
           Probabilities
                   │
                   ▼
            Final Prediction
```

The equal-weight approach achieved:

> **98.41% accuracy (310/315)**

This did not outperform the standalone DenseNet-121 model.

### Observation

The experiment indicated that direct probability averaging can be affected by **model confidence differences**. DenseNet-121 produced highly confident predictions in several cases, which could dominate the averaged probability distribution.

---

# 🌡️ 2️⃣ Temperature-Calibrated Voting

To investigate the effect of model confidence, a temperature-scaling experiment was performed.

The temperature parameter used in the experiment was:

```text
T = 1.5
```

The purpose was to soften the probability distributions before combining predictions.

```text
Model Logits
     │
     ▼
Temperature Scaling
     │
     ▼
Softened Probabilities
     │
     ├──────────────┐
     │              │
     ▼              ▼
 HybridViT      DenseNet-121
     │              │
     └──────┬───────┘
            ▼
        Soft Voting
```

The calibrated voting strategy achieved:

> **98.41% accuracy (310/315)**

Therefore, temperature scaling reduced the effect of extreme confidence in the probability distributions but did not change the final classification accuracy compared with equal soft voting.

---

# 🎯 3️⃣ Specialist ViT Override

The next strategy was based on the observation that the two models showed different class-level strengths.

DenseNet-121 performed particularly well on:

* Bone
* Brain
* Heart

while HybridViT achieved stronger recall for:

* Lungs

The specialist strategy therefore used the HybridViT prediction as a targeted override in appropriate Lung-related decisions.

```text
                  Input
                    │
                    ▼
              DenseNet-121
                    │
          ┌─────────┴─────────┐
          │                   │
   Bone / Brain / Heart      Lung
          │                   │
          ▼                   ▼
    Keep DenseNet        HybridViT
      Prediction          Specialist
                              │
                              ▼
                       Final Prediction
```

The refined specialist strategy achieved:

> **98.73% accuracy (311/315)**

This improved over equal soft voting and demonstrated that **model specialization can be more useful than naive probability averaging**.

---

# 🪜 4️⃣ 2-Tier Cascade Classifier

The final decision strategy used a hierarchical approach.

Instead of averaging the two models' probabilities for every sample, the system uses DenseNet-121 as the primary classifier and introduces HybridViT selectively.

```text
                       Input Image
                            │
                            ▼
                     ┌────────────┐
                     │ DenseNet121│
                     └──────┬─────┘
                            │
                 ┌──────────┴──────────┐
                 │                     │
                 ▼                     ▼
          Bone / Brain / Heart       Lungs
                 │                     │
                 ▼                     ▼
        Keep DenseNet Prediction   HybridViT
                                      │
                                      ▼
                               Final Decision
```

The 2-Tier Cascade achieved:

> **99.05% accuracy (312/315)**

This matched the best standalone DenseNet-121 result.

### Important Interpretation

The cascade strategy **did not exceed DenseNet-121** on the current test set.

Its significance is therefore not that it produced a higher accuracy, but that the experiments demonstrate how **architecture-specific strengths can be incorporated into a structured decision process**.

For this dataset and test split, DenseNet-121 remained the strongest standalone model by accuracy.

---

# 📈 Evaluation

The project uses several evaluation approaches within the notebooks.

## Classification Metrics

The evaluation workflow includes:

* Accuracy
* Precision
* Recall
* F1-score
* Classification reports
* Confusion matrices

For the HybridViT evaluation, predictions are collected from the validation/test DataLoader and analyzed using Scikit-learn.

Example:

```python
cm = confusion_matrix(
    all_labels,
    all_preds
)
```

The confusion matrix is visualized using Seaborn.

---

# 🔬 Error Analysis

The project goes beyond reporting overall accuracy by examining individual misclassifications.

The final errors were investigated to understand whether they originated from:

* Model limitations
* Class similarity
* Dataset ambiguity
* Image artifacts
* Potential outlier samples

Several samples, including:

```text
#289
#292
#294
```

were examined during the error analysis.

A particular focus was placed on Lung samples that were predicted as Bone by DenseNet-121.

The analysis suggested that at least one sample exhibited characteristics of a potential dataset outlier or highly ambiguous image.

> These observations are dataset-specific and should not be interpreted as evidence of clinical mislabeling without expert medical review.

---

# 👁️ HybridViT Token Activation Visualization

The HybridViT notebook includes a custom qualitative visualization method.

The workflow is:

```text
Validation/Test Image
        │
        ▼
ResNet-10 Feature Extraction
        │
        ▼
1 × 1 Feature Projection
        │
        ▼
Transformer Tokens
        │
        ▼
Token Feature Variance
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
* Softmax confidence
* Correct/incorrect status
* Token activation overlay

### ⚠️ Visualization Disclaimer

The generated heatmap is based on the **variance of Transformer token representations**.

It is therefore an **activation-based qualitative visualization**.

It is **not**:

* Direct Transformer attention visualization
* Attention rollout
* Grad-CAM
* A clinically validated saliency map
* Evidence of medically relevant anatomical attention

---

# 🧪 Experimental Findings

The experiments produced several useful observations.

### 1. DenseNet-121 was the strongest standalone model

DenseNet-121 achieved:

> **99.05% test accuracy**

compared with:

> **98.10% for HybridViT**

This indicates that the CNN baseline was extremely competitive for this particular dataset.

---

### 2. HybridViT provided complementary information

Although HybridViT achieved lower overall accuracy, it showed stronger Lung recall:

> **96.77% Lung recall**

compared with:

> **90.32% for DenseNet-121**

This motivated the specialist routing experiment.

---

### 3. Naive soft voting did not improve performance

Equal soft voting achieved:

> **98.41%**

which was lower than DenseNet-121's:

> **99.05%**

The experiment suggests that combining model probabilities does not automatically produce better predictions.

---

### 4. Temperature scaling did not change the final accuracy

Using:

```text
T = 1.5
```

produced the same observed accuracy as equal soft voting:

> **98.41%**

This suggests that confidence calibration alone was insufficient to improve the decision boundaries on this test set.

---

### 5. Specialist routing improved over simple voting

The specialist ViT override achieved:

> **98.73%**

This was better than the two voting strategies and demonstrates the potential value of **architecture-specific specialization**.

---

### 6. The cascade matched the best standalone result

The 2-Tier Cascade achieved:

> **99.05% (312/315)**

matching DenseNet-121.

Therefore, the current experiments do not demonstrate that ensembling necessarily improves overall accuracy. Instead, they demonstrate that **different decision strategies can exploit complementary model behavior in different ways**.

---

# 🗂️ Dataset

The project uses the **MediScan Dataset** hosted on Kaggle.

**Dataset:** MediScan Dataset

https://www.kaggle.com/datasets/pritydas825/mediscan-dataset

The dataset contains four image categories:

```text
Bone
Brain
Heart
Lungs
```

Supported image formats include:

```text
.png
.jpg
.jpeg
.webp
```

The original notebook workflow creates the required dataset splits from the configured dataset directory.

---

# 📁 Repository Structure

The repository is intentionally kept notebook-based.

```text
Mediscan-HybridViT/
│
├── Mediscan-HybridViT Model/
│   └── Model-related files and resources
│
├── mediscan-hybridvit.ipynb
│   └── Main model implementation and training workflow
│
├── mediscan-hybridvit with predication summary.ipynb
│   └── Prediction summaries, evaluation,
│       comparison, and additional experiments
│
└── README.md
    └── Project documentation
```

> **Note:** The repository does not currently use separate `models/`, `ensemble/`, `utils/`, `train.py`, or `evaluate.py` directories/files. The implementation and experiments are contained in the available Jupyter Notebooks.

---

# 🚀 Getting Started

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

```bash
pip install torch torchvision monai pytorch-lightning torchmetrics matplotlib numpy pandas scikit-learn opencv-python seaborn
```

---

# 💻 Usage

The current implementation is provided through Jupyter Notebooks.

## Main Notebook

Open:

```text
mediscan-hybridvit.ipynb
```

The notebook contains the main workflow for:

1. Dataset preparation
2. Image preprocessing
3. Data augmentation
4. HybridViT construction
5. HybridViT training
6. HybridViT validation
7. DenseNet-121 construction
8. DenseNet-121 training
9. Model evaluation
10. Confusion matrix generation
11. Token activation visualization

---

## Prediction & Experimental Analysis Notebook

Open:

```text
mediscan-hybridvit with predication summary.ipynb
```

This notebook contains additional prediction summaries and experimental analysis, including model comparison and ensemble-related experiments.

These experiments include:

* Standalone model comparison
* Equal Soft Voting
* Temperature-Calibrated Voting
* Specialist ViT Override
* 2-Tier Cascade
* Sample-level prediction analysis
* Error analysis

---

## Running with Jupyter

Install Jupyter if necessary:

```bash
pip install notebook
```

Then run:

```bash
jupyter notebook
```

Open the desired notebook and execute the cells in sequence.

The notebooks can also be opened through environments such as **Google Colab** or **Kaggle**, provided the dataset paths and required dependencies are configured correctly.

---

# 🛠️ Technologies Used

| Category                | Technology                                   |
| ----------------------- | -------------------------------------------- |
| Programming Language    | Python                                       |
| Deep Learning           | PyTorch                                      |
| Training Framework      | PyTorch Lightning                            |
| Medical Imaging         | MONAI                                        |
| Neural Networks         | ResNet-10, DenseNet-121, Transformer Encoder |
| Metrics                 | TorchMetrics, Scikit-learn                   |
| Computer Vision         | OpenCV                                       |
| Visualization           | Matplotlib, Seaborn                          |
| Data Processing         | NumPy, Pandas                                |
| Development Environment | Jupyter Notebook / Kaggle                    |

---

# 🔍 Model Comparison

| Feature                  | Mediscan-HybridViT         | DenseNet-121            |
| ------------------------ | -------------------------- | ----------------------- |
| Architecture             | ResNet-10 + Transformer    | DenseNet-121            |
| Feature Learning         | CNN + Self-Attention       | Dense CNN Connections   |
| Local Features           | ✅                          | ✅                       |
| Global Context           | Transformer Self-Attention | CNN Feature Propagation |
| Input Size               | `224 × 224`                | `224 × 224`             |
| Class Imbalance Strategy | Augmentation               | Weighted Cross-Entropy  |
| Training                 | PyTorch Lightning          | PyTorch Lightning       |
| Token Visualization      | ✅                          | ❌                       |
| Overall Test Accuracy    | **98.10%**                 | **99.05%**              |
| Lung Recall              | **96.77%**                 | 90.32%                  |

---

# 🧩 Ensemble Comparison

| Strategy                      |   Accuracy |     Correct |
| ----------------------------- | ---------: | ----------: |
| HybridViT                     |     98.10% |     309/315 |
| DenseNet-121                  | **99.05%** | **312/315** |
| Equal Soft Voting             |     98.41% |     310/315 |
| Temperature-Calibrated Voting |     98.41% |     310/315 |
| Specialist ViT Override       |     98.73% |     311/315 |
| 2-Tier Cascade                | **99.05%** | **312/315** |

### Overall Observation

The experiments demonstrate that **ensemble complexity does not automatically translate into higher accuracy**.

The strongest observed result came from DenseNet-121 and the cascade strategy, both achieving **99.05%**.

The more important finding is that the two models exhibit **different class-level behavior**, particularly for Lung images, which provides motivation for future adaptive ensemble strategies.

---

# ⚠️ Limitations

This project is an experimental deep learning study and has several limitations.

### Dataset Limitations

* The test set contains only **315 samples**.
* The class distribution is highly imbalanced.
* The Heart class contains only **5 test samples**.
* The dataset comes from a single source.
* External dataset validation has not been performed.

### Model Limitations

* The HybridViT does not consistently outperform the DenseNet-121 baseline.
* Ensemble strategies did not improve overall accuracy beyond DenseNet-121.
* Temperature calibration did not improve the observed test accuracy.
* The current specialist routing rules are dataset-specific.

### Medical AI Limitations

This project is intended for **research and educational purposes**.

The reported results should not be interpreted as:

* Clinical validation
* Medical diagnosis
* A clinically deployable diagnostic system
* Evidence of real-world generalization

Additional validation using larger and independent clinical datasets would be required before any real-world medical application.

---

# 🚧 Future Work

Potential future improvements include:

* [ ] Evaluate on an independent external test dataset
* [ ] Perform stratified cross-validation
* [ ] Investigate additional class-balancing techniques
* [ ] Add Grad-CAM for DenseNet-121
* [ ] Implement Transformer attention rollout
* [ ] Compare direct attention maps with token activation maps
* [ ] Add detailed per-class F1-score comparison
* [ ] Add ROC/AUC analysis where statistically appropriate
* [ ] Investigate learned ensemble weights
* [ ] Replace rule-based routing with a learned gating network
* [ ] Explore confidence-aware dynamic model selection
* [ ] Perform statistical significance testing
* [ ] Investigate larger pretrained CNN–Transformer architectures
* [ ] Evaluate external-domain generalization
* [ ] Export models for deployment experiments
* [ ] Extend the framework to additional medical imaging modalities

---

# 📚 Research Takeaways

The project demonstrates several practical lessons for multi-class medical image classification:

> **1. A strong CNN baseline should always be established before introducing a more complex architecture.**

DenseNet-121 achieved **99.05%**, showing that architectural complexity alone does not guarantee better performance.

> **2. Hybrid architectures can provide complementary information even when their overall accuracy is lower.**

HybridViT achieved higher Lung recall than DenseNet-121 in the evaluated test set.

> **3. Simple probability averaging is not necessarily an effective ensemble strategy.**

Equal soft voting performed worse than the best individual model.

> **4. Calibration alone may not solve model disagreement.**

Temperature scaling reduced confidence sharpness but did not improve the final observed accuracy.

> **5. Model specialization can be useful.**

The specialist ViT strategy improved the ensemble result to **98.73%**.

> **6. Hierarchical decision-making can match a strong standalone model.**

The 2-Tier Cascade reached **99.05%**, matching DenseNet-121 on the current test set.

---

# 👩‍💻 Author

**Prity Rani Das**

GitHub:
https://github.com/prityd825

---

# 📄 License

This project is released under the **MIT License**.

---

⭐ **If you find this project useful, consider giving the repository a star!**

---

## ⚠️ Research Disclaimer

This repository is intended for **research, experimentation, and educational purposes only**. The models have not been clinically validated and should not be used to make medical decisions or diagnoses.
