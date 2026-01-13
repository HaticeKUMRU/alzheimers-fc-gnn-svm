"""
Graph Neural Network (GNN) with Graph Attention Network (GAT) for
Alzheimer's Disease classification using functional connectivity matrices.

This implementation is adapted from the following open-source repository:
https://github.com/gururgg/fNET-Analysis/blob/main/ABIDE/ABIDE-GNN.py

The code has been modified for dataset-specific preprocessing,
ROI selection, and extended clinical performance evaluation.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns

# PyG Imports
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge

# -------------------------------
# NILEARN INTEGRATION
# -------------------------------
try:
    from nilearn import datasets
    NILEARN_AVAILABLE = True
except ImportError:
    NILEARN_AVAILABLE = False
    print("WARNING: 'nilearn' is not installed. ROI names will not be displayed. (pip install nilearn)")

# -------------------------------
# SETTINGS
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
K_TOP_EDGES = 8
HIDDEN_DIM = 16
DROPOUT = 0.3
DROP_EDGE_RATES = 0.05
EPOCHS = 150
ROI_HEDEF = 60
REPORT_TOP_N = 8

print(f"Strategy: Top {ROI_HEDEF} ROI | Sensitivity & Specificity Analysis Enabled")


def fix_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# -------------------------------
# REPORTING FUNCTION FOR THESIS / PAPER
# -------------------------------
def print_thesis_report(selected_indices, importances):
    if not NILEARN_AVAILABLE:
        print(f"Selected Indices: {selected_indices[:REPORT_TOP_N]}")
        return

    print("\n" + "=" * 95)
    print(f"PAPER RESULTS: TOP {REPORT_TOP_N} MOST DISCRIMINATIVE BRAIN REGIONS")
    print("=" * 95)

    try:
        atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7)
        labels = [l.decode() if isinstance(l, bytes) else l for l in atlas.labels]
    except Exception as e:
        print(f"Atlas could not be loaded: {e}")
        return

    print(f"{'Rank':<5} | {'Score':<8} | {'ROI ID':<8} | {'Hemisphere':<10} | {'Network':<15} | {'Region Detail'}")
    print("-" * 95)

    for rank in range(REPORT_TOP_N):
        idx = selected_indices[rank]
        score = importances[rank]
        roi_id = idx + 1

        if 0 <= idx < len(labels):
            full_label = labels[idx]
            parts = full_label.split('_')
            if len(parts) >= 4:
                hemi = "Left (LH)" if "LH" in parts[1] else "Right (RH)"
                network = parts[2]
                region_detail = " ".join(parts[3:])
            else:
                hemi = "-"
                network = "Unknown"
                region_detail = full_label
            print(f"{rank + 1:<5} | {score:.4f}   | {roi_id:<8} | {hemi:<10} | {network:<15} | {region_detail}")
        else:
            print(f"{rank + 1:<5} | {score:.4f}   | {roi_id:<8} | -          | -               | Invalid ID")
    print("=" * 95 + "\n")


# -------------------------------
# 1. DATA LOADING
# -------------------------------
def load_full_matrix(mat_path):
    try:
        mat = scipy.io.loadmat(mat_path)
    except:
        raise ValueError(f"File could not be read: {mat_path}")

    keys = [k for k in mat.keys() if not k.startswith("__")]
    X_key = keys[0]
    if 'X_Tum' in mat:
        X_key = 'X_Tum'
    X = mat[X_key]

    if 'y_Tum' in mat:
        y = mat['y_Tum'].flatten()
    else:
        print("'y_Tum' not found, generating labels automatically...")
        y = np.concatenate([np.zeros(X.shape[0] // 2), np.ones(X.shape[0] - X.shape[0] // 2)])

    return X, y


# -------------------------------
# 2. GRAPH CONVERSION
# -------------------------------
def matrix_to_graph(corr_matrix, label, k=10):
    x = torch.tensor(corr_matrix, dtype=torch.float)
    vals, indices = torch.topk(torch.abs(x), k=k, dim=1)
    source_nodes = []
    target_nodes = []
    for i in range(x.shape[0]):
        for j in range(k):
            target = indices[i, j]
            if i != target:
                source_nodes.append(i)
                target_nodes.append(target.item())
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    y = torch.tensor([label], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)


# -------------------------------
# 3. MODEL
# -------------------------------
class MatrixGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4, concat=True)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False)
        self.lin = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, edge_index, batch):
        if self.training:
            edge_index, _ = dropout_edge(edge_index, p=DROP_EDGE_RATES, force_undirected=True)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=DROPOUT, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        x = self.lin(x)
        return x
