import os
import numpy as np
import warnings

from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# =================================================
# CONFIG
# =================================================
warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

GROUPS = [
    {"name": "CN", "folder": "CN", "label": 0},
    {"name": "AD", "folder": "AD", "label": 1}
]

ROI_INDEX = np.r_[0:200, 210:410]   # 410 → 400 ROI
ROI_N = len(ROI_INDEX)
UT_IDX = np.triu_indices(ROI_N, k=1)

# =================================================
# UTILS
# =================================================
def fisher_z(x):
    x = np.clip(x, -0.999999, 0.999999)
    return np.arctanh(x)


def extract_hon_pearson_features(fc_matrix):
    """
    HON-Pearson feature extraction from FC matrix
    """
    z_fc = fisher_z(fc_matrix)
    hon = np.corrcoef(z_fc)
    hon[~np.isfinite(hon)] = 0
    np.fill_diagonal(hon, 0)
    return fisher_z(hon[UT_IDX])


# =================================================
# LOAD FC MATRICES
# =================================================
FC_ALL, y_ALL = [], []

for g in GROUPS:
    folder_path = os.path.join(DATA_DIR, g["folder"])
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
        except Exception:
            continue

FC_ALL = np.asarray(FC_ALL)
y_ALL = np.asarray(y_ALL)

print(f"\nSubjects: {len(FC_ALL)} | ROI: {ROI_N}")

# =================================================
# TRUE LOOCV
# =================================================
loo = LeaveOneOut()

y_pred = []
y_score = []

for train_idx, test_idx in loo.split(FC_ALL):

    X_train = np.array([
        extract_hon_pearson_features(FC_ALL[i])
        for i in train_idx
    ])
    y_train = y_ALL[train_idx]

    X_test = extract_hon_pearson_features(
        FC_ALL[test_idx[0]]
    ).reshape(1, -1)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegressionCV(
            solver="saga",
            l1_ratios=[0.5],
            Cs=100,
            cv=5,
            scoring="roc_auc",
            max_iter=5000,
            tol=1e-3,
            n_jobs=1
        ))
    ])

    model.fit(X_train, y_train)

    prob = model.predict_proba(X_test)[0, 1]
    y_score.append(prob)
    y_pred.append(int(prob >= 0.5))

# =================================================
# PERFORMANCE METRICS
# =================================================
y_pred = np.array(y_pred)
y_score = np.array(y_score)

ACC = accuracy_score(y_ALL, y_pred)
AUC = roc_auc_score(y_ALL, y_score)

TN, FP, FN, TP = confusion_matrix(y_ALL, y_pred).ravel()

Sensitivity = TP / (TP + FN)
Specificity = TN / (TN + FP)
Precision   = TP / (TP + FP)
F1          = 2 * Precision * Sensitivity / (Precision + Sensitivity)

print("\n===== RESULTS (TRUE LOOCV | HON-PEARSON) =====")
print(f"Accuracy     : {ACC*100:.2f}%")
print(f"AUC          : {AUC:.3f}")
print(f"Sensitivity  : {Sensitivity*100:.2f}%")
print(f"Specificity  : {Specificity*100:.2f}%")
print(f"F1-score     : {F1:.3f}")
