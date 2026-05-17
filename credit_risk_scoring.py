"""
=============================================================
  Credit Risk Scoring — GiveMeSomeCredit Dataset
  Author  : Generated for Viva / Academic Submission
  Dataset : https://www.kaggle.com/c/GiveMeSomeCredit
=============================================================
  Models  : Logistic Regression, Decision Tree, KNN, SVM,
            Naive Bayes, Random Forest, XGBoost, AdaBoost,
            Gradient Boosting, Bagging Classifier
  Metrics : Accuracy, Precision, Recall, F1, ROC-AUC,
            Confusion Matrix, Classification Report
=============================================================
"""

# ─────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────
import os
import warnings
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                    # non-interactive backend for script mode
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection   import train_test_split
from sklearn.preprocessing     import StandardScaler
from sklearn.linear_model      import LogisticRegression
from sklearn.tree              import DecisionTreeClassifier
from sklearn.neighbors         import KNeighborsClassifier
from sklearn.svm               import LinearSVC
from sklearn.naive_bayes       import GaussianNB
from sklearn.ensemble          import (RandomForestClassifier,
                                        AdaBoostClassifier,
                                        GradientBoostingClassifier,
                                        BaggingClassifier)
from sklearn.metrics           import (accuracy_score, precision_score,
                                        recall_score, f1_score,
                                        roc_auc_score, confusion_matrix,
                                        classification_report,
                                        ConfusionMatrixDisplay,
                                        RocCurveDisplay)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# OUTPUT DIRECTORIES
# ─────────────────────────────────────────────
PLOTS_DIR   = os.path.join("outputs", "plots")
MODELS_DIR  = os.path.join("outputs", "models")
REPORTS_DIR = os.path.join("outputs", "reports")

for d in [PLOTS_DIR, MODELS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────
# HELPER — save figure
# ─────────────────────────────────────────────
def save_fig(name: str, tight: bool = True):
    if tight:
        plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Plot saved → {path}")


# ═════════════════════════════════════════════
# 1. DATA LOADING
# ═════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 1 — DATA LOADING")
print("="*60)

DATA_PATH = os.path.join("data", "cs-training.csv")

df = pd.read_csv(DATA_PATH, index_col=0)   # first column is a row-index
print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  Columns: {list(df.columns)}")


# ═════════════════════════════════════════════
# 2. EDA — EXPLORATORY DATA ANALYSIS
# ═════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 2 — EXPLORATORY DATA ANALYSIS")
print("="*60)

TARGET = "SeriousDlqin2yrs"

# ── 2a. Class Distribution ─────────────────
print("\n  [2a] Class Distribution")
class_counts = df[TARGET].value_counts()
print(class_counts)

fig, ax = plt.subplots(figsize=(6, 4))
class_counts.plot(kind="bar", color=["steelblue", "tomato"], ax=ax, edgecolor="white")
ax.set_title("Class Distribution — SeriousDlqin2yrs", fontsize=13, fontweight="bold")
ax.set_xlabel("Class (0 = Good, 1 = Default)", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_xticklabels(["0 — Good", "1 — Default"], rotation=0)
for bar in ax.patches:
    ax.annotate(f"{int(bar.get_height()):,}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center", va="bottom", fontsize=10)
save_fig("01_class_distribution")

# ── 2b. Missing Value Analysis ─────────────
print("\n  [2b] Missing Value Analysis")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values("Missing %", ascending=False)
print(missing_df)

if not missing_df.empty:
    fig, ax = plt.subplots(figsize=(8, 4))
    missing_df["Missing %"].plot(kind="bar", color="coral", edgecolor="white", ax=ax)
    ax.set_title("Missing Values (%) per Feature", fontsize=13, fontweight="bold")
    ax.set_ylabel("Missing %", fontsize=11)
    ax.set_xlabel("")
    for bar in ax.patches:
        ax.annotate(f"{bar.get_height():.1f}%",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9)
    save_fig("02_missing_values")

# ── 2c. Correlation Heatmap ────────────────
print("\n  [2c] Correlation Heatmap")
fig, ax = plt.subplots(figsize=(11, 8))
corr = df.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, linecolor="white", ax=ax,
            annot_kws={"size": 8})
ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
save_fig("03_correlation_heatmap")

# ── 2d. Feature Distributions ─────────────
print("\n  [2d] Feature Distributions")
features = [c for c in df.columns if c != TARGET]
n_cols = 3
n_rows = int(np.ceil(len(features) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.5))
axes = axes.flatten()

for i, feat in enumerate(features):
    data_clipped = df[feat].clip(
        lower=df[feat].quantile(0.01),
        upper=df[feat].quantile(0.99)
    )
    axes[i].hist(data_clipped, bins=40, color="steelblue", edgecolor="white", alpha=0.85)
    axes[i].set_title(feat, fontsize=9, fontweight="bold")
    axes[i].set_xlabel("")
    axes[i].set_ylabel("Freq", fontsize=8)
    axes[i].tick_params(labelsize=7)

for j in range(i + 1, len(axes)):     # hide empty subplots
    axes[j].set_visible(False)

fig.suptitle("Feature Distributions (1st–99th percentile clipped)", fontsize=13, fontweight="bold")
save_fig("04_feature_distributions")


# ═════════════════════════════════════════════
# 3. DATA PREPROCESSING
# ═════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 3 — DATA PREPROCESSING")
print("="*60)

# ── 3a. Drop duplicates ────────────────────
before = len(df)
df.drop_duplicates(inplace=True)
print(f"  Duplicates removed : {before - len(df)}")

# ── 3b. Handle missing values ─────────────
# MonthlyIncome  → median imputation (skewed distribution)
# NumberOfDependents → median imputation
for col in ["MonthlyIncome", "NumberOfDependents"]:
    if col in df.columns:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"  Imputed '{col}' with median = {median_val}")

print(f"  Missing values after imputation: {df.isnull().sum().sum()}")

# ── 3c. Feature / Target split ────────────
X = df.drop(columns=[TARGET])
y = df[TARGET]

print(f"\n  Features : {X.shape[1]}")
print(f"  Samples  : {X.shape[0]:,}")
print(f"  Class balance  0={y.value_counts()[0]:,}  1={y.value_counts()[1]:,}")

# ── 3d. Train-Test Split ──────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n  Train : {X_train.shape[0]:,}  |  Test : {X_test.shape[0]:,}")

# ── 3e. Feature Scaling ───────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
print("  ✓ Scaler saved.")

# Subsets for memory-intensive models (KNN, SVM)
SMALL_N = 20_000
X_tr_small = X_train_sc[:SMALL_N]
y_tr_small = y_train.iloc[:SMALL_N]


# ═════════════════════════════════════════════
# 4. MODEL TRAINING
# ═════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 4 — MODEL TRAINING")
print("="*60)

# ─────────────────────────────────────────────
# Metric helper
# ─────────────────────────────────────────────
def evaluate(name, model, X_tr, y_tr, X_te, y_te, use_proba=True):
    """Fit model and return a metrics dictionary."""
    print(f"  Training {name} ...", end=" ", flush=True)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    if use_proba and hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_te)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_te)
    else:
        y_prob = y_pred

    auc = roc_auc_score(y_te, y_prob)

    metrics = {
        "Model"     : name,
        "Accuracy"  : accuracy_score(y_te, y_pred),
        "Precision" : precision_score(y_te, y_pred, zero_division=0),
        "Recall"    : recall_score(y_te, y_pred, zero_division=0),
        "F1-Score"  : f1_score(y_te, y_pred, zero_division=0),
        "ROC-AUC"   : auc,
    }
    print(f"Done  |  Acc={metrics['Accuracy']:.4f}  AUC={auc:.4f}")

    # Save classification report
    report = classification_report(y_te, y_pred, target_names=["Good (0)", "Default (1)"])
    report_path = os.path.join(REPORTS_DIR, f"{name.replace(' ', '_')}_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Model: {name}\n\n")
        f.write(report)

    return metrics, model, y_pred, y_prob


results = []   # list of metric dicts
trained = {}   # name → fitted model

# ── Logistic Regression ───────────────────
m, model, _, _ = evaluate("Logistic Regression",
    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, n_jobs=-1),
    X_train_sc, y_train, X_test_sc, y_test)
results.append(m); trained["Logistic Regression"] = model

# ── Decision Tree ─────────────────────────
m, model, _, _ = evaluate("Decision Tree",
    DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42),
    X_train_sc, y_train, X_test_sc, y_test)
results.append(m); trained["Decision Tree"] = model

# ── KNN (small sample) ────────────────────
m, model, _, _ = evaluate("KNN",
    KNeighborsClassifier(n_neighbors=9, n_jobs=-1),
    X_tr_small, y_tr_small, X_test_sc, y_test)
results.append(m); trained["KNN"] = model

# ── SVM (LinearSVC, small sample) ─────────
m, model, _, _ = evaluate("SVM (LinearSVC)",
    LinearSVC(class_weight="balanced", max_iter=2000, random_state=42),
    X_tr_small, y_tr_small, X_test_sc, y_test, use_proba=False)
results.append(m); trained["SVM (LinearSVC)"] = model

# ── Naive Bayes ───────────────────────────
m, model, _, _ = evaluate("Naive Bayes",
    GaussianNB(),
    X_train_sc, y_train, X_test_sc, y_test)
results.append(m); trained["Naive Bayes"] = model

# ── Random Forest ─────────────────────────
m, model, _, _ = evaluate("Random Forest",
    RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced",
                           n_jobs=-1, random_state=42),
    X_train_sc, y_train, X_test_sc, y_test)
results.append(m); trained["Random Forest"] = model

# ── XGBoost ───────────────────────────────
scale_pos = int((y_train == 0).sum() / (y_train == 1).sum())
m, model, _, _ = evaluate("XGBoost",
    XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                  scale_pos_weight=scale_pos, use_label_encoder=False,
                  eval_metric="logloss", tree_method="hist",
                  n_jobs=-1, random_state=42, verbosity=0),
    X_train_sc, y_train, X_test_sc, y_test)
results.append(m); trained["XGBoost"] = model

# ── AdaBoost ──────────────────────────────
m, model, _, _ = evaluate("AdaBoost",
    AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42),
    X_train_sc, y_train, X_test_sc, y_test)
results.append(m); trained["AdaBoost"] = model

# ── Gradient Boosting ─────────────────────
m, model, _, _ = evaluate("Gradient Boosting",
    GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                               subsample=0.8, random_state=42),
    X_train_sc, y_train, X_test_sc, y_test)
results.append(m); trained["Gradient Boosting"] = model

# ── Bagging ───────────────────────────────
m, model, _, _ = evaluate("Bagging",
    BaggingClassifier(n_estimators=50, max_samples=0.8, max_features=0.8,
                      n_jobs=-1, random_state=42),
    X_train_sc, y_train, X_test_sc, y_test)
results.append(m); trained["Bagging"] = model


# ═════════════════════════════════════════════
# 5. RESULTS & COMPARISON
# ═════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 5 — MODEL COMPARISON")
print("="*60)

results_df = pd.DataFrame(results).set_index("Model")
results_df = results_df.sort_values("ROC-AUC", ascending=False)
print("\n" + results_df.to_string(float_format="{:.4f}".format))

# ── Save comparison table ─────────────────
results_df.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"))

# ── Accuracy Comparison Bar Chart ─────────
fig, ax = plt.subplots(figsize=(12, 6))
metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
x = np.arange(len(results_df))
width = 0.15
colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2"]

for i, metric in enumerate(metrics_to_plot):
    bars = ax.bar(x + i * width, results_df[metric], width, label=metric, color=colors[i], alpha=0.88)

ax.set_xticks(x + width * 2)
ax.set_xticklabels(results_df.index, rotation=30, ha="right", fontsize=9)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("Model Comparison — All Metrics", fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
save_fig("05_model_comparison")

# ── ROC-AUC bar ───────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
colors_roc = sns.color_palette("viridis", len(results_df))
bars = ax.barh(results_df.index, results_df["ROC-AUC"], color=colors_roc, edgecolor="white")
ax.set_xlabel("ROC-AUC Score", fontsize=11)
ax.set_title("ROC-AUC Comparison Across Models", fontsize=13, fontweight="bold")
ax.set_xlim(0, 1.05)
for bar, val in zip(bars, results_df["ROC-AUC"]):
    ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9)
save_fig("06_roc_auc_comparison")


# ═════════════════════════════════════════════
# 6. BEST MODEL — CONFUSION MATRIX & ROC CURVE
# ═════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 6 — BEST MODEL ANALYSIS")
print("="*60)

best_name  = results_df["ROC-AUC"].idxmax()
best_model = trained[best_name]
print(f"  Best model by ROC-AUC : {best_name}")

# Decide training set for best model
use_small_models = {"KNN", "SVM (LinearSVC)"}
X_tr_b = X_tr_small if best_name in use_small_models else X_train_sc
y_tr_b = y_tr_small if best_name in use_small_models else y_train

best_model.fit(X_tr_b, y_tr_b)
y_pred_best = best_model.predict(X_test_sc)

if hasattr(best_model, "predict_proba"):
    y_prob_best = best_model.predict_proba(X_test_sc)[:, 1]
elif hasattr(best_model, "decision_function"):
    y_prob_best = best_model.decision_function(X_test_sc)
else:
    y_prob_best = y_pred_best

# ── Confusion Matrix ──────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(cm, display_labels=["Good (0)", "Default (1)"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"Confusion Matrix — {best_name}", fontsize=12, fontweight="bold")
save_fig("07_confusion_matrix_best")

# ── ROC Curve ─────────────────────────────
if not isinstance(y_prob_best[0], (int, np.integer)):
    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(y_test, y_prob_best,
                                     name=best_name, ax=ax, color="tomato")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random")
    ax.set_title(f"ROC Curve — {best_name}", fontsize=12, fontweight="bold")
    ax.legend()
    save_fig("08_roc_curve_best")

# ── Save best model ───────────────────────
joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.joblib"))
print(f"  ✓ Best model saved → outputs/models/best_model.joblib")


# ═════════════════════════════════════════════
# 7. FEATURE IMPORTANCE
# ═════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 7 — FEATURE IMPORTANCE")
print("="*60)

feature_names = list(X.columns)
tree_models   = ["Decision Tree", "Random Forest", "XGBoost",
                 "AdaBoost", "Gradient Boosting", "Bagging"]

for name in tree_models:
    model = trained[name]

    # Extract importances
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "estimators_"):
        # BaggingClassifier — average base estimator importances if available
        try:
            importances = np.mean(
                [est.feature_importances_ for est in model.estimators_], axis=0
            )
        except AttributeError:
            print(f"  Skipped {name} (base estimator has no feature_importances_)")
            continue
    else:
        print(f"  Skipped {name}")
        continue

    fi_df = pd.Series(importances, index=feature_names).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors_fi = sns.color_palette("Blues_r", len(fi_df))
    fi_df.plot(kind="barh", ax=ax, color=colors_fi, edgecolor="white")
    ax.set_title(f"Feature Importance — {name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance Score", fontsize=10)
    ax.axvline(fi_df.mean(), color="red", linestyle="--", linewidth=1, label="Mean")
    ax.legend(fontsize=9)
    save_fig(f"09_feature_importance_{name.replace(' ', '_')}")
    print(f"  ✓ Feature importance plot saved for {name}")


# ═════════════════════════════════════════════
# 8. CONFUSION MATRICES — ALL MODELS
# ═════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 8 — CONFUSION MATRICES (All Models)")
print("="*60)

model_names = list(trained.keys())
n = len(model_names)
n_cols = 4
n_rows = int(np.ceil(n / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
axes = axes.flatten()

for i, name in enumerate(model_names):
    model = trained[name]
    use_small = name in use_small_models
    X_tr_i = X_tr_small if use_small else X_train_sc
    y_tr_i = y_tr_small if use_small else y_train
    model.fit(X_tr_i, y_tr_i)
    y_pred_i = model.predict(X_test_sc)
    cm_i = confusion_matrix(y_test, y_pred_i)
    ConfusionMatrixDisplay(cm_i, display_labels=["Good", "Default"]).plot(
        ax=axes[i], colorbar=False, cmap="Blues")
    axes[i].set_title(name, fontsize=9, fontweight="bold")

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Confusion Matrices — All Models", fontsize=13, fontweight="bold")
save_fig("10_all_confusion_matrices")


# ═════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════
print("\n" + "="*60)
print("  COMPLETE — Summary")
print("="*60)
print(f"\n  Best Model  : {best_name}")
print(f"  ROC-AUC     : {results_df.loc[best_name, 'ROC-AUC']:.4f}")
print(f"  Accuracy    : {results_df.loc[best_name, 'Accuracy']:.4f}")
print(f"  F1-Score    : {results_df.loc[best_name, 'F1-Score']:.4f}")
print(f"\n  Plots  saved → {PLOTS_DIR}/")
print(f"  Models saved → {MODELS_DIR}/")
print(f"  Reports saved→ {REPORTS_DIR}/")
print("="*60)
