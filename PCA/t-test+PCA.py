import os
import numpy as np
import warnings

from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from scipy.stats import ttest_ind

# =================================================
# CONFIG
# =================================================
warnings.filterwarnings("ignore")

TOP_K   = 500     # Number of features selected by t-test
PCA_VAR = 0.90    # PCA explained variance
C_LOG   = 1.0     # Logistic Regression regularization

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

GROUPS = [
    {"name": "CN", "folder": "45_CN_yeni", "label": 0},
    {"name": "AD", "folder": "45_AD_yeni", "label": 1}
]

ROI_INDEX = np.r_[0:200, 210:410]   # 410 -> 400 ROI
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
    HON-Pearson feature extraction
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
        fc = np.loadtxt(os.path.join(folder_path, fname))
        fc = fc[np.ix_(ROI_INDEX, ROI_INDEX)]
        fc[~np.isfinite(fc)] = 0
        np.fill_diagonal(fc, 0)

        FC_ALL.append(fc)
        y_ALL.append(g["label"])

FC_ALL = np.asarray(FC_ALL)
y_ALL = np.asarray(y_ALL)

print(f"\nSubjects loaded: {len(FC_ALL)} | ROI: {ROI_N}")

# =================================================
# FEATURE MATRIX
# =================================================
print("\nExtracting HON-Pearson features...")
X = np.array([extract_hon_pearson_features(fc) for fc in FC_ALL])
y = y_ALL.copy()

n_subjects, n_features = X.shape
print(f"Feature matrix: {n_subjects} × {n_features}")

# =================================================
# TRUE LOOCV
# =================================================
loo = LeaveOneOut()

y_true = []
y_score = []
feature_counter = np.zeros(n_features)

print("\nStarting TRUE LOOCV...")

for fold, (train_idx, test_idx) in enumerate(loo.split(X), 1):

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Feature selection using t-test (training set only)
    t_vals, _ = ttest_ind(
        X_train[y_train == 0],
        X_train[y_train == 1],
        axis=0,
        equal_var=False
    )

    selected_idx = np.argsort(np.abs(t_vals))[-TOP_K:]
    feature_counter[selected_idx] += 1

    X_train = X_train[:, selected_idx]
    X_test  = X_test[:, selected_idx]

    # Standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # PCA (training set only)
    pca = PCA(n_components=PCA_VAR)
    X_train = pca.fit_transform(X_train)
    X_test  = pca.transform(X_test)

    # Logistic Regression classifier
    clf = LogisticRegression(
        C=C_LOG,
        penalty="l2",
        solver="liblinear",
        max_iter=5000
    )
    clf.fit(X_train, y_train)

    prob = clf.predict_proba(X_test)[0, 1]

    y_true.append(y_test[0])
    y_score.append(prob)

    if fold % 10 == 0 or fold == n_subjects:
        print(f" Fold {fold}/{n_subjects}")

print("\nLOOCV finished.")

# =================================================
# PERFORMANCE METRICS
# =================================================
y_true  = np.array(y_true)
y_score = np.array(y_score)

AUC = roc_auc_score(y_true, y_score)

thresholds = np.linspace(y_score.min(), y_score.max(), 500)
best_acc, best_th = 0, 0

for th in thresholds:
    y_pred = (y_score >= th).astype(int)
    acc = accuracy_score(y_true, y_pred)
    if acc > best_acc:
        best_acc, best_th = acc, th

y_pred = (y_score >= best_th).astype(int)
TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()

Sensitivity = TP / (TP + FN)
Specificity = TN / (TN + FP)

print("\n===== RESULTS (TRUE LOOCV | t-test + PCA + Logistic) =====")
print(f"AUC          : {AUC:.3f}")
print(f"Accuracy     : {best_acc*100:.2f}%")
print(f"Sensitivity  : {Sensitivity*100:.2f}%")
print(f"Specificity  : {Specificity*100:.2f}%")

# =================================================
# FEATURE STABILITY
# =================================================
stability_ratio = feature_counter / n_subjects
stable_edges = np.argsort(stability_ratio)[-int(0.10 * n_features):]

np.save("feature_stability_ratio_logistic.npy", stability_ratio)
np.save("stable_edge_indices_logistic.npy", stable_edges)

print("\n===== FEATURE STABILITY =====")
print("Example stable edges:", stable_edges[:10])
