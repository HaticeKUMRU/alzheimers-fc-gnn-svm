"""
Alzheimer Classification using Node Strength Features
Pipeline:
400x400 Connectivity Matrix → Node Strength → RF Feature Selection → MLP

Clinical Metrics:
Accuracy, F1-score, Sensitivity, Specificity

MLP implementation is adapted from the following open-source repository:
https://github.com/gururgg/fNET-Analysis/blob/main/ABIDE/ABIDE-MLP.py
"""

# ======================================================
# 1. IMPORTS
# ======================================================
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Optional: Nilearn anatomical labeling
try:
    from nilearn import datasets
    NILEARN_AVAILABLE = True
except ImportError:
    NILEARN_AVAILABLE = False
    print("Warning: Nilearn not installed → anatomical labels disabled")


# ======================================================
# 2. GLOBAL SETTINGS
# ======================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RANDOM_SEED = 42
N_SPLITS = 5
EPOCHS = 150
BATCH_SIZE = 16

ROI_TARGET = 60
HIDDEN_DIM = 64
REPORT_TOP_N = 8

MAT_PATH = "Alzheimer_400x400_Full_yenipearson.mat"

print("Model: Node Strength → MLP")
print("Metrics: Accuracy | F1 | Sensitivity | Specificity")
print(f"Device: {DEVICE}")


# ======================================================
# 3. REPRODUCIBILITY
# ======================================================
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================================================
# 4. DATA LOADING & FEATURE EXTRACTION
# ======================================================
def load_and_vectorize(mat_path):
    if not os.path.exists(mat_path):
        raise FileNotFoundError("MAT file not found")

    mat = scipy.io.loadmat(mat_path)
    X = mat["X_Tum"]       # (N, 400, 400)
    y = mat["y_Tum"].flatten()

    print(f"Raw data loaded: {X.shape}")

    # Node strength feature extraction
    X_vec = np.mean(np.abs(X), axis=2)  # (N, 400)
    print(f"Node strength features extracted: {X_vec.shape}")

    # Class balancing (same strategy as GNN experiment)
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]

    np.random.shuffle(idx_0)
    np.random.shuffle(idx_1)

    limit = 40
    selected = np.concatenate([idx_0[:limit], idx_1[:limit]])
    np.random.shuffle(selected)

    print(f"Balanced dataset: {len(selected)} samples")

    return X_vec[selected], y[selected]


# ======================================================
# 5. RF-BASED ROI SELECTION + REPORT
# ======================================================
def rf_feature_selection(X, y, n_features):
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    rf.fit(X, y)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:n_features]
    scores = importances[indices]

    return indices, scores


def print_roi_report(indices, scores):
    if not NILEARN_AVAILABLE:
        print("Selected ROI indices:", indices[:REPORT_TOP_N])
        return

    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7)
    labels = [l.decode() if isinstance(l, bytes) else l for l in atlas.labels]

    print("\n" + "=" * 95)
    print(f"Top {REPORT_TOP_N} Discriminative Brain Regions (Random Forest)")
    print("=" * 95)
    print(f"{'Rank':<5} | {'Score':<7} | {'ROI':<6} | {'Hemisphere':<10} | {'Network':<15} | Region")
    print("-" * 95)

    for r in range(REPORT_TOP_N):
        idx = indices[r]
        parts = labels[idx].split("_")

        hemi = "LH" if "LH" in parts[1] else "RH"
        network = parts[2]
        region = " ".join(parts[3:])

        print(f"{r+1:<5} | {scores[r]:.4f} | {idx+1:<6} | {hemi:<10} | {network:<15} | {region}")

    print("=" * 95)


# ======================================================
# 6. MLP MODEL
# ======================================================
class NodeStrengthMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(HIDDEN_DIM, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.net(x)


# ======================================================
# 7. TRAINING & EVALUATION
# ======================================================
def train_mlp(X, y):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    all_preds, all_labels = [], []

    for fold, (tr, va) in enumerate(skf.split(X, y)):
        X_tr, X_va = X[tr], X[va]
        y_tr, y_va = y[tr], y[va]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)

        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr)),
            batch_size=BATCH_SIZE, shuffle=True
        )

        model = NodeStrengthMLP(input_dim=X.shape[1]).to(DEVICE)
        optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
        criterion = nn.CrossEntropyLoss()

        best_f1, best_preds, best_truth = 0, None, None

        for _ in range(EPOCHS):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                out = model(torch.FloatTensor(X_va).to(DEVICE))
                preds = out.argmax(1).cpu().numpy()

            f1 = f1_score(y_va, preds, average="weighted", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_preds = preds
                best_truth = y_va

        tn, fp, fn, tp = confusion_matrix(best_truth, best_preds).ravel()
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)

        print(f"Fold {fold+1} | F1: {best_f1:.3f} | Sensitivity: {sens:.3f} | Specificity: {spec:.3f}")

        all_preds.extend(best_preds)
        all_labels.extend(best_truth)

    return np.array(all_labels), np.array(all_preds)


# ======================================================
# 8. MAIN
# ======================================================
if __name__ == "__main__":
    set_seed(RANDOM_SEED)

    X_vec, y_vec = load_and_vectorize(MAT_PATH)

    roi_idx, roi_scores = rf_feature_selection(X_vec, y_vec, ROI_TARGET)
    print_roi_report(roi_idx, roi_scores)

    X_sel = X_vec[:, roi_idx]
    labels, preds = train_mlp(X_sel, y_vec)

    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    print("\nFINAL MLP RESULTS")
    print(f"Accuracy    : {accuracy_score(labels, preds):.4f}")
    print(f"F1-score    : {f1_score(labels, preds, average='weighted'):.4f}")
    print(f"Sensitivity : {tp/(tp+fn):.4f}")
    print(f"Specificity : {tn/(tn+fp):.4f}")

    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=["Healthy", "AD"],
                yticklabels=["Healthy", "AD"])
    plt.title("MLP Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
