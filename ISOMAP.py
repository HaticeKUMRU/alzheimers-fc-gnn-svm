import os
import numpy as np
import warnings

from sklearn.manifold import Isomap
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# =================================================
# CONFIG
# =================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

GROUPS = [
    {"name": "CN", "folder": "45_CN_yeni", "label": 0},
    {"name": "AD", "folder": "45_AD_yeni", "label": 1}
]

ROI_INDEX = np.r_[0:200, 210:410]  # 400 ROI
ROI_N = len(ROI_INDEX)
UT_IDX = np.triu_indices(ROI_N, k=1)

# =================================================
# UTILS
# =================================================
def fisher_z(x):
    x = np.clip(x, -0.999999, 0.999999)
    return np.arctanh(x)


def extract_hon_features(M):
    """
    HON-Pearson feature extraction
    """
    Mz = fisher_z(M)
    HON = np.corrcoef(Mz)
    HON[~np.isfinite(HON)] = 0
    np.fill_diagonal(HON, 0)
    return fisher_z(HON[UT_IDX])


# =================================================
# LOAD FC MATRICES
# =================================================
FC_ALL, y_ALL = [], []

for g in GROUPS:
    folder_path = os.path.join(DATA_DIR, g["folder"])
    if not os.path.exists(folder_path):
        continue
    files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    print(f"{g['name']} ({len(files)} subjects)")

    for fname in files:
        try:
            fc = np.loadtxt(os.path.join(folder_path, fname))
            fc = fc[np.ix_(ROI_INDEX, ROI_INDEX)]
            fc[~np.isfinite(fc)] = 0
            np.fill_diagonal(fc, 0)
            FC_ALL.append(fc)
            y_ALL.append(g["label"])
        except:
            continue

FC_ALL = np.asarray(FC_ALL)
y_ALL = np.asarray(y_ALL)

# =================================================
# LOOCV WITH ISOMAP
# =================================================
loo = LeaveOneOut()
pred_all = []
score_all = []

print("\nIsomap ile boyut indirgeme ve LOOCV başlatılıyor...")

for fold, (train_idx, test_idx) in enumerate(loo.split(FC_ALL), 1):

    X_train_raw = np.array([extract_hon_features(FC_ALL[i]) for i in train_idx])
    y_train = y_ALL[train_idx]
    X_test_raw = extract_hon_features(FC_ALL[test_idx[0]]).reshape(1, -1)

    # Pipeline: StandardScaler + Isomap + LogisticRegressionCV
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("isomap", Isomap(n_components=10, n_neighbors=5)),
        ("logreg", LogisticRegressionCV(
            solver="saga",
            penalty="l2",
            max_iter=5000,
            cv=3
        ))
    ])

    clf.fit(X_train_raw, y_train)
    prob = clf.predict_proba(X_test_raw)[0, 1]

    pred_all.append(int(prob >= 0.5))
    score_all.append(prob)

    if fold % 10 == 0:
        print(f" Fold {fold}/{len(FC_ALL)} tamamlandı.")

# =================================================
# PERFORMANCE METRICS
# =================================================
pred_all = np.array(pred_all)
score_all = np.array(score_all)

ACC = accuracy_score(y_ALL, pred_all)
AUC = roc_auc_score(y_ALL, score_all)

print("\n===== ISOMAP + LOOCV SONUÇLARI =====")
print(f"Accuracy : {ACC * 100:.2f}%")
print(f"AUC      : {AUC:.3f}")
