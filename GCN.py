# =========================================================
# GNN for fMRI Functional Connectivity (TXT files)
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
# DEVICE & SEED
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fix_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =========================================================
# ROI FILTER (420 → 400)
# =========================================================
def filter_rois(mat):
    idx = np.concatenate([np.arange(0, 200), np.arange(210, 410)])
    return mat[np.ix_(idx, idx)]

# =========================================================
# EDGE SPARSITY (%25)
# =========================================================
def sparsify_matrix(mat, keep_ratio=0.25):
    n = mat.shape[0]
    triu = np.triu_indices(n, k=1)
    vals = np.abs(mat[triu])
    thresh = np.percentile(vals, 100 * (1 - keep_ratio))
    sparse = np.where(np.abs(mat) >= thresh, mat, 0)
    return sparse

# =========================================================
# MATRIX → GRAPH
# =========================================================
def matrix_to_graph(mat, label):
    mat = filter_rois(mat)
    mat = sparsify_matrix(mat, keep_ratio=0.25)

    x = torch.tensor(mat, dtype=torch.float)  # node features
    edge_index = torch.tensor(np.array(np.nonzero(mat)), dtype=torch.long)
    edge_weight = torch.tensor(mat[edge_index[0], edge_index[1]], dtype=torch.float)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_weight,
        y=torch.tensor([label], dtype=torch.long)
    )

# =========================================================
# LOAD DATASET (.txt)
# =========================================================
def load_graphs(data_dir):
    graphs = []
    class_folders = sorted(
        [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    )

    print("📂 Sınıf klasörleri:", class_folders)

    for cls in class_folders:
        label = 1 if "AD" in cls else 0
        cls_path = os.path.join(data_dir, cls)

        for file in os.listdir(cls_path):
            if not file.endswith(".txt"):
                continue

            mat = np.loadtxt(os.path.join(cls_path, file))
            graphs.append(matrix_to_graph(mat, label))

    print(f"\n✅ Toplam grafik sayısı: {len(graphs)}")
    return graphs

# =========================================================
# FOCAL LOSS
# =========================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()

# =========================================================
# GCN MODEL (BASELINE)
# =========================================================
class GCNNet(nn.Module):
    def __init__(self, in_dim):
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
# LOOCV
# =========================================================
def run_loocv(graphs):
    loo = LeaveOneOut()
    y_true, y_pred, y_prob = [], [], []

    for i, (train_idx, test_idx) in enumerate(loo.split(graphs)):
        print(f"Denek {i+1}/{len(graphs)}")

        train_set = [graphs[j] for j in train_idx]
        test_set = [graphs[j] for j in test_idx]

        train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

        # 🔁 MODEL SIFIRDAN
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
            # Erken durdurma
            if epoch > 30 and abs(prev_loss - avg_loss) < 1e-4:
                break
            prev_loss = avg_loss

        # 🧪 TEST (GÖRÜLMEMİŞ VERİ)
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
# MAIN
# =========================================================
if __name__ == "__main__":
    fix_seed(42)

    DATA_DIR = "/Users/haticekumru/Desktop/Alzheimer_fnets"
    graphs = load_graphs(DATA_DIR)

    print("\n🚀 LOOCV başlıyor...\n")
    run_loocv(graphs)
