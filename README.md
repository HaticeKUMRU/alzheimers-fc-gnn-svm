# Alzheimer’s Disease Classification: Functional Connectivity & ML Pipeline

![Status](https://img.shields.io/badge/Status-Active-success)
![Language](https://img.shields.io/badge/Language-MATLAB-orange)
![License](https://img.shields.io/badge/License-MIT-blue)

## 📌 Overview

This repository contains a complete pipeline for classifying **Alzheimer’s Disease (AD)** vs. **Cognitively Normal (CN)** subjects using functional connectivity (FC) derived from fMRI data.

The project explores two major modeling approaches:
1.  **Classical Machine Learning:** SVM and MLP using graph-theoretic features (Node Strength).
2.  **Deep Learning:** Graph Neural Networks (GNN) and DeepSets using full connectivity matrices.

## 🚀 Key Features

* **Data Preprocessing:** ROI signal extraction and reduction (410 $\to$ 400 regions).
* **Connectivity Construction:**
    * Pearson Correlation.
    * Fisher Z-Transformation.
    * Percentile-based Thresholding (Top 26%).
* **Feature Engineering:**
    * **Vector Features:** Node Strength centrality for SVM/MLP.
    * **Graph Features:** $400 \times 400$ Adjacency Matrices for GNNs.
* **Classification:**
    * Linear SVM with Leave-One-Out Cross-Validation (LOOCV).
    * Statistical Feature Selection (Welch's t-test).

## 📂 Repository Structure

The project is organized into modular scripts for reproducibility:

```text
├── data_preparation/
│   ├── prepare_data.m           # Generates FC matrices and extracts features
│   └── ALZHEIMER_ML_VERISI/     # Output folder for processed .mat files
│
├── analysis/
│   ├── svm_analysis.m           # Runs SVM classification and reports metrics
│   └── comparison_plots.m       # Visualization of results
│
├── README.md                    # Project documentation
└── .gitignore
