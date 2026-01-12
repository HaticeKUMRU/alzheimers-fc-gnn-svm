import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, recall_score
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
# NILEARN ENTEGRASYONU
# -------------------------------
try:
    from nilearn import datasets

    NILEARN_AVAILABLE = True
except ImportError:
    NILEARN_AVAILABLE = False
    print("UYARI: 'nilearn' yüklü değil. Bölge isimleri yazılamayacak. (pip install nilearn)")

# -------------------------------
# AYARLAR
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
K_TOP_EDGES = 8
HIDDEN_DIM = 16
DROPOUT = 0.3
DROP_EDGE_RATES = 0.05
EPOCHS = 150
ROI_HEDEF = 60
REPORT_TOP_N = 8

print(f"Strateji: Top {ROI_HEDEF} ROI | Sensitivity & Specificity Analizi Dahil")


def fix_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# -------------------------------
# MAKALE İÇİN RAPORLAMA FONKSİYONU
# -------------------------------
def print_thesis_report(selected_indices, importances):
    if not NILEARN_AVAILABLE:
        print(f"Seçilen İndeksler: {selected_indices[:REPORT_TOP_N]}")
        return

    print("\n" + "=" * 95)
    print(f"MAKALE İÇİN BULGULAR: EN AYIRT EDİCİ {REPORT_TOP_N} BÖLGE")
    print("=" * 95)

    try:
        atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7)
        labels = [l.decode() if isinstance(l, bytes) else l for l in atlas.labels]
    except Exception as e:
        print(f"Atlas yüklenemedi: {e}")
        return

    print(f"{'Sıra':<5} | {'Skor':<8} | {'ROI ID':<8} | {'Yarımküre':<10} | {'AĞ (Network)':<15} | {'BÖLGE DETAYI'}")
    print("-" * 95)

    for rank in range(REPORT_TOP_N):
        idx = selected_indices[rank]
        score = importances[rank]
        roi_id = idx + 1

        if 0 <= idx < len(labels):
            full_label = labels[idx]
            parts = full_label.split('_')
            if len(parts) >= 4:
                hemi = "Sol (LH)" if "LH" in parts[1] else "Sağ (RH)"
                network = parts[2]
                region_detail = " ".join(parts[3:])
            else:
                hemi = "-"
                network = "Bilinmiyor"
                region_detail = full_label
            print(f"{rank + 1:<5} | {score:.4f}   | {roi_id:<8} | {hemi:<10} | {network:<15} | {region_detail}")
        else:
            print(f"{rank + 1:<5} | {score:.4f}   | {roi_id:<8} | -          | -               | Geçersiz ID")
    print("=" * 95 + "\n")


# -------------------------------
# 1. VERİ YÜKLEME
# -------------------------------
def load_full_matrix(mat_path):
    try:
        mat = scipy.io.loadmat(mat_path)
    except:
        raise ValueError(f"Dosya okunamadı: {mat_path}")

    keys = [k for k in mat.keys() if not k.startswith("__")]
    X_key = keys[0]
    if 'X_Tum' in mat: X_key = 'X_Tum'
    X = mat[X_key]

    if 'y_Tum' in mat:
        y = mat['y_Tum'].flatten()
    else:
        print("'y_Tum' bulunamadı, otomatik oluşturuluyor...")
        y = np.concatenate([np.zeros(X.shape[0] // 2), np.ones(X.shape[0] - X.shape[0] // 2)])

    return X, y


# -------------------------------
# 2. GRAF DÖNÜŞÜMÜ
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


# -------------------------------
# ANA DÖNGÜ
# -------------------------------
if __name__ == "__main__":
    fix_seed(42)
    mat_path = "Alzheimer_400x400_Full_yenipearson.mat"

    if os.path.exists(mat_path):
        # A. Yükle
        X_full, y_raw = load_full_matrix(mat_path)
        if X_full.shape[0] == 400 and X_full.ndim == 3:
            X_full = np.transpose(X_full, (2, 0, 1))

        # RF Seçimi
        X_flat = np.mean(np.abs(X_full), axis=2)
        print(f"En İyi {ROI_HEDEF} Bölge Seçiliyor (Random Forest)...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_flat, y_raw)

        importances = rf.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        top_indices = sorted_indices[:ROI_HEDEF]
        top_scores = importances[top_indices]
        print_thesis_report(top_indices, top_scores)

        X_selected = X_full[:, top_indices, :][:, :, top_indices]
        print(f"  Veri Hazır: {X_selected.shape}")

        # B. Graf Dönüşümü
        print(f"Graf dönüşümü yapılıyor...")
        graph_dataset = [matrix_to_graph(X_selected[i], y_raw[i], k=K_TOP_EDGES) for i in range(len(X_selected))]

        # C. Eğitim
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        labels = np.array([d.y.item() for d in graph_dataset])

        results_acc = []
        results_f1 = []
        results_sens = []  # Sensitivity listesi
        results_spec = []  # Specificity listesi

        global_preds = []
        global_labels = []

        print("\n EĞİTİM BAŞLIYOR (Sensitivity & Specificity Takibi)")
        print("=" * 75)

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
            train_loader = DataLoader([graph_dataset[i] for i in train_idx], batch_size=16, shuffle=True)
            val_loader = DataLoader([graph_dataset[i] for i in val_idx], batch_size=16, shuffle=False)

            model = MatrixGAT(input_dim=ROI_HEDEF, hidden_dim=HIDDEN_DIM, output_dim=2).to(device)
            optimizer = Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
            scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

            y_train = labels[train_idx]
            counts = np.bincount(y_train)
            weights = torch.tensor([sum(counts) / (2 * c) for c in counts]).float().to(device)
            criterion = nn.CrossEntropyLoss(weight=weights)

            best_fold_f1 = 0
            best_fold_acc = 0
            best_preds = []
            best_truths = []

            for epoch in range(1, EPOCHS + 1):
                model.train()
                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    out = model(batch.x, batch.edge_index, batch.batch)
                    loss = criterion(out, batch.y)
                    loss.backward()
                    optimizer.step()
                scheduler.step()

                model.eval()
                temp_preds, temp_lbls = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        out = model(batch.x, batch.edge_index, batch.batch)
                        temp_preds.extend(out.argmax(1).cpu().numpy())
                        temp_lbls.extend(batch.y.cpu().numpy())

                val_f1 = f1_score(temp_lbls, temp_preds, average='weighted', zero_division=0)
                val_acc = accuracy_score(temp_lbls, temp_preds)

                if val_f1 > best_fold_f1:
                    best_fold_f1 = val_f1
                    best_fold_acc = val_acc
                    best_preds = temp_preds
                    best_truths = temp_lbls

            # --- BU FOLD İÇİN SENS/SPEC HESAPLA ---
            tn_f, fp_f, fn_f, tp_f = confusion_matrix(best_truths, best_preds).ravel()
            fold_sens = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else 0
            fold_spec = tn_f / (tn_f + fp_f) if (tn_f + fp_f) > 0 else 0

            results_acc.append(best_fold_acc)
            results_f1.append(best_fold_f1)
            results_sens.append(fold_sens)
            results_spec.append(fold_spec)

            global_preds.extend(best_preds)
            global_labels.extend(best_truths)

            print(f" Fold {fold + 1}: Acc: %{best_fold_acc * 100:.1f} | F1: {best_fold_f1:.3f} | "
                  f"Sens: {fold_sens:.3f} | Spec: {fold_spec:.3f}")

        # ----------------------------------------
        # FİNAL METRİKLER 
        # ----------------------------------------
        print("\n" + "=" * 60)
        print(" FİNAL KLİNİK PERFORMANS DEĞERLERİ")
        print("=" * 60)

        # Global Confusion Matrix
        cm = confusion_matrix(global_labels, global_preds)
        tn, fp, fn, tp = cm.ravel()

        # Final Metrics Calculation
        sensitivity = tp / (tp + fn)  # Sensitivity = Recall (Positive Class)
        specificity = tn / (tn + fp)  # Specificity = Recall (Negative Class)
        f1_final = f1_score(global_labels, global_preds, average='weighted')
        acc_final = accuracy_score(global_labels, global_preds)

        print(f" Ortalama Accuracy : %{acc_final * 100:.2f}")
        print(f" Ortalama F1-Score : {f1_final:.4f}")
        print(f" SENSITIVITY (Duyarlılık - Hasta Bulma) : {sensitivity:.4f}")
        print(f" SPECIFICITY (Özgüllük - Sağlıklı Bulma): {specificity:.4f}")
        print("-" * 60)
        print(f"Detaylar: TP: {tp} (Doğru Hasta) | TN: {tn} (Doğru Sağlıklı)")
        print(f"          FP: {fp} (Yanlış Hasta) | FN: {fn} (Kaçırılan Hasta)")

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Sağlıklı (0)", "Hasta (1)"],
                    yticklabels=["Sağlıklı (0)", "Hasta (1)"])
        plt.title(f"Confusion Matrix\nSens: {sensitivity:.2f} | Spec: {specificity:.2f}")
        plt.xlabel("Tahmin Edilen")
        plt.ylabel("Gerçek Durum")
        plt.show()

    else:
        print(f" Dosya bulunamadı! '{mat_path}'")
