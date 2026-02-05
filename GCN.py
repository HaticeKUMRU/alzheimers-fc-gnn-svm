# =========================================================
# Graph Convolutional Network (GCN) for fMRI Functional Connectivity
# =========================================================
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# =========================================================
# DEVICE CONFIGURATION & REPRODUCIBILITY
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int = 42):
    """Fix random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =========================================================
# ROI FILTERING (420 → 400)
# =========================================================
def filter_rois(matrix: np.ndarray) -> np.ndarray:
    """
    Select 400 ROIs from the original 420 by removing unwanted indices.
    
    Args:
        matrix (np.ndarray): Square functional connectivity matrix.
    
    Returns:
        np.ndarray: Filtered matrix with 400 ROIs.
    """
    indices = np.concatenate([np.arange(0, 200), np.arange(210, 410)])
    return matrix[np.ix_(indices, indices)]

# =========================================================
# EDGE SPARSIFICATION (%25)
# =========================================================
def sparsify_matrix(matrix: np.ndarray, keep_ratio: float = 0.25) -> np.ndarray:
    """
    Retain only the top edges by absolute value to enforce sparsity.
    
    Args:
        matrix (np.ndarray): Square functional connectivity matrix.
        keep_ratio (float): Fraction of edges to keep.
    
    Returns:
        np.ndarray: Sparse connectivity matrix.
    """
    n = matrix.shape[0]
    triu = np.triu_indices(n, k=1)
    vals = np.abs(matrix[triu])
    threshold = np.percentile(vals, 100 * (1 - keep_ratio))
    sparse_matrix = np.where(np.abs(matrix) >= threshold, matrix, 0)
    return sparse_matrix

# =========================================================
# MATRIX TO GRAPH CONVERSION
# =========================================================
def matrix_to_graph(matrix: np.ndarray, label: int) -> Data:
    """
    Convert functional connectivity matrix into a PyG graph.
    
    Args:
        matrix (np.ndarray): Functional connectivity matrix.
        label (int): Class label (0=Control, 1=AD).
    
    Returns:
        torch_geometric.data.Data: Graph object.
    """
    matrix = filter_rois(matrix)
    matrix = sparsify_matrix(matrix, keep_ratio=0.25)

    # Node features = adjacency matrix itself
    x = torch.tensor(matrix, dtype=torch.float)
    
    # Edge index & edge attributes
    edge_index = torch.tensor(np.array(np.nonzero(matrix)), dtype=torch.long)
    edge_attr = torch.tensor(matrix[edge_index[0], edge_index[1]], dtype=torch.float)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([label], dtype=torch.long)
    )

# =========================================================
# LOAD DATASET
# =========================================================
def load_graphs(data_dir: str) -> list:
    """
    Load all graphs from a directory structured by class folders.
    
    Args:
        data_dir (str): Root directory containing class subfolders with .txt matrices.
    
    Returns:
        list: List of PyG Data objects.
    """
    graphs = []
    class_folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
    print("📂 Class folders detected:", class_folders)

    for cls in class_folders:
        label = 1 if "AD" in cls.upper() else 0
        cls_path = os.path.join(data_dir, cls)

        for file in os.listdir(cls_path):
            if not file.endswith(".txt"):
                continue
            matrix = np.loadtxt(os.path.join(cls_path, file))
            graphs.append(matrix_to_graph(matrix, label))

    print(f"\n✅ Total number of graphs: {len(graphs)}")
    return graphs

# =========================================================
# FOCAL LOSS FOR CLASS IMBALANCE
# =========================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()

# =========================================================
# GRAPH CONVOLUTIONAL NETWORK (BASELINE)
# =========================================================
class GCNNet(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.gcn1 = GCNConv(in_dim, 64)
        self.gcn2 = GCNConv(64, 64)

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, x, edge_index, batch):
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.classifier(x)

# =========================================================
# LEAVE-ONE-OUT CROSS-VALIDATION
# =========================================================
def run_loocv(graphs: list):
    """
    Evaluate the model using Leave-One-Out Cross-Validation (LOOCV).
    
    Args:
        graphs (list): List of PyG Data objects.
    """
    loo = LeaveOneOut()
    y_true, y_pred, y_prob = [], [], []

    for i, (train_idx, test_idx) in enumerate(loo.split(graphs)):
        print(f"Subject {i+1}/{len(graphs)}")

        train_set = [graphs[j] for j in train_idx]
        test_set = [graphs[j] for j in test_idx]

        train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

        # Initialize model from scratch for each fold
        model = GCNNet(graphs[0].x.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
        criterion = FocalLoss()

        prev_loss = 1e9
        for epoch in range(100):
            model.train()
            total_loss = 0

            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            # Early stopping
            if epoch > 30 and abs(prev_loss - avg_loss) < 1e-4:
                break
            prev_loss = avg_loss

        # Evaluation on unseen test data
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                prob = F.softmax(out, dim=1)[0, 1].item()

                y_true.append(batch.y.item())
                y_pred.append(out.argmax(dim=1).item())
                y_prob.append(prob)

    print("\n==============================")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("F1-score:", f1_score(y_true, y_pred))
    print("AUC:", roc_auc_score(y_true, y_prob))
    print("==============================")

# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    set_seed(42)

    # Set a general path for the dataset (adjust according to your system)
    DATA_DIR = os.path.join("data", "Alzheimer_fnets")  # Example: ./data/Alzheimer_fnets
    graphs = load_graphs(DATA_DIR)

    print("\n🚀 Starting LOOCV evaluation...\n")
    run_loocv(graphs)
