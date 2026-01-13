# Alzheimer’s Disease Classification Using Functional Connectivity and Machine Learning

## Overview

This repository presents a reproducible machine learning and deep learning pipeline for **Alzheimer’s Disease (AD) classification** based on **functional connectivity (FC) networks** derived from fMRI data. The study integrates **classical machine learning** and **graph-based deep learning** approaches, enabling a systematic comparison between different modeling paradigms.

---

## Data Source

The functional connectivity data are derived from preprocessed fMRI time-series data. ROI-based brain signals are extracted and reduced from **410 to 400 regions** prior to connectivity computation.
The dataset consists of subjects belonging to two groups: Cognitively Normal (CN) and Alzheimer’s Disease (AD). To ensure a fair and unbiased classification setting, the dataset was manually balanced, resulting in 45 CN and 45 AD subjects.

---

## Methods

### Functional Connectivity Construction

* ROI-based time-series signals are extracted and reduced from **410 to 400 regions**.
* Functional connectivity matrices are computed using:

  * **Pearson correlation** (baseline connectivity)
  * **Spearman correlation**
  * **Higher-Order Network (HON) Spearman–Pearson correlation**
* Fisher Z-transformation is applied for numerical stability.
* Weak connections are removed using **percentile-based thresholding** (top 26% strongest connections).

### Feature Engineering

Two different data representations are constructed:

1. **Node-level feature vectors**

   * Mean absolute node strength is computed for each ROI.
   * Used as input for **SVM** and **MLP** models.

2. **Graph-level representations**

   * Full **400 × 400** functional connectivity matrices.
   * Used as input for **Graph Neural Network (GNN)** models.

---

## Models

### Support Vector Machine (SVM)

* Uses node-strength features.
* Features are extracted from:

  * Pearson correlation matrices
  * Spearman correlation matrices
  * C-HON Spearman–Pearson correlation matrices
* Feature selection is performed using Welch’s t-test.
* Evaluation is carried out using Leave-One-Out Cross-Validation (LOOCV).

### Multilayer Perceptron (MLP)

* Uses node-strength feature vectors.
* Features are extracted **only from Pearson functional connectivity matrices**.
* The MLP implementation is adapted from the following open-source repository:
  [https://github.com/gururgg/fNET-Analysis/blob/main/ABIDE/ABIDE-MLP.py](https://github.com/gururgg/fNET-Analysis/blob/main/ABIDE/ABIDE-MLP.py)

### Graph Neural Network (GNN)+GAT

* Implemented using a **Graph Attention Network (GAT)** architecture.
* Operates directly on full **400 × 400 Pearson functional connectivity matrices**.
* Preserves the complete graph structure and learns attention-based edge importance weights.
* The GNN (GAT) implementation is adapted from the following open-source repository:
  https://github.com/gururgg/fNET-Analysis/blob/main/ABIDE/ABIDE-GNN.py

---

## Repository Structure

```text
├── pearson.m              # Pearson FC computation and feature extraction
├── Spearman.m             # Spearman FC computation
├── HON_Pearson.m          # Higher-Order Network based Pearson correlation
├── HON_Spearman.m         # Higher-Order Network based Spearman correlation
├── SVM.m                  # SVM training and evaluation (LOOCV)
├── MLP.py                 # MLP model (adapted from fNET-Analysis)
├── GNN+GAT.py             # Graph Attention Network (GAT) model
├── README.md              # Project documentation
├── .gitignore             # Git ignore rules
└── LICENSE                # License file


---

## Reproducibility

* No manual intervention is required.
* Random seeds can be fixed to ensure deterministic experiments.


## Applications

* Alzheimer’s Disease classification
* Brain network analysis
* Comparison of classical machine learning and graph deep learning
* Neuroimaging-based biomarker research


