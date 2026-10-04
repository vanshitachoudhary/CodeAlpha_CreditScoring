"""CodeAlpha Task 1 - Credit Scoring Model
Run:  python train.py
"""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve, accuracy_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from features import add_features

RANDOM_STATE = 42
CSV_PATH = "data/german_credit.csv"
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ---------------------------------------------------------------- 1. LOAD
def load_data() -> pd.DataFrame:
    if os.path.exists(CSV_PATH):
        print(f"Loading local file: {CSV_PATH}")
        return pd.read_csv(CSV_PATH)
    print("Downloading German Credit dataset from OpenML (internet chahiye)...")
    from sklearn.datasets import fetch_openml
    data = fetch_openml("credit-g", version=1, as_frame=True)
    df = data.frame
    df.to_csv(CSV_PATH, index=False)  # next time download nahi karna padega
    return df


df = load_data()
print("Shape:", df.shape)

# Target: 'bad' = default (risky) = 1, 'good' = 0
df["target"] = (df["class"] == "bad").astype(int)
df = df.drop(columns=["class"])
print("\nTarget distribution (1 = default):")
print(df["target"].value_counts(normalize=True).round(3))

# ---------------------------------------------------------------- 2. EDA plots
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
df["target"].value_counts().plot(kind="bar", ax=ax[0], color=["#4c9", "#e55"])
ax[0].set_title("Class balance (0=good, 1=default)")
df.boxplot(column="credit_amount", by="target", ax=ax[1])
ax[1].set_title("Credit amount vs default")
df.boxplot(column="duration", by="target", ax=ax[2])
ax[2].set_title("Duration vs default")
plt.suptitle("")
plt.tight_layout()
plt.savefig("outputs/eda.png", dpi=120)
plt.close()

# ---------------------------------------------------------------- 3. FEATURES
df = add_features(df)
X = df.drop(columns=["target"])
y = df["target"]

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]
print(f"\n{len(cat_cols)} categorical, {len(num_cols)} numeric features")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
])

# class_weight="balanced" -> imbalanced data handle karta hai (default sirf 30%)
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=10, class_weight="balanced",
        random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=3,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
        class_weight="balanced", random_state=RANDOM_STATE),
}

# ---------------------------------------------------------------- 4. TRAIN + EVAL
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results, fitted, curves = {}, {}, {}

for name, clf in models.items():
    pipe = Pipeline([("prep", preprocess), ("clf", clf)])
    cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc").mean()
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    results[name] = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
        "Brier": brier_score_loss(y_test, proba),
        "CV ROC-AUC (5-fold)": cv_auc,
    }
    fitted[name] = pipe
    curves[name] = roc_curve(y_test, proba)
    print(f"\n===== {name} =====")
    print(classification_report(y_test, pred, target_names=["Good", "Default"]))

res_df = pd.DataFrame(results).T.round(3)
print("\n=========== MODEL COMPARISON ===========")
print(res_df)
res_df.to_csv("outputs/model_comparison.csv")

# ---------------------------------------------------------------- 5. PLOTS
plt.figure(figsize=(6, 5))
for name, (fpr, tpr, _) in curves.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC={results[name]['ROC-AUC']:.2f})")
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/roc_curve.png", dpi=120)
plt.close()

best_name = max(results, key=lambda k: results[k]["ROC-AUC"])
best_model = fitted[best_name]
print(f"\nBest model (by ROC-AUC): {best_name}")

cm = confusion_matrix(y_test, (best_model.predict_proba(X_test)[:, 1] >= 0.5).astype(int))
plt.figure(figsize=(4, 4))
plt.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
plt.xticks([0, 1], ["Good", "Default"])
plt.yticks([0, 1], ["Good", "Default"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=120)
plt.close()

# ---- Business-aware threshold: German Credit cost matrix
# Defaulter ko approve karna (FN) = 5x costly, achhe customer ko reject karna (FP) = 1x
COST_FN, COST_FP = 5, 1
p_best = best_model.predict_proba(X_test)[:, 1]
grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
costs = [COST_FN * ((p_best < t) & (y_test == 1)).sum()
         + COST_FP * ((p_best >= t) & (y_test == 0)).sum() for t in grid]
cost_threshold = float(grid[int(np.argmin(costs))])
print(f"Cost-optimal threshold: {cost_threshold} (default 0.5 cost = {costs[list(grid).index(0.5)]}, "
      f"optimal cost = {min(costs)})")
plt.figure(figsize=(6, 4))
plt.plot(grid, costs, color="#2563eb")
plt.axvline(cost_threshold, color="red", ls="--", label=f"Optimal = {cost_threshold}")
plt.xlabel("Threshold")
plt.ylabel(f"Total cost (FN x{COST_FN}, FP x{COST_FP})")
plt.title("Cost vs decision threshold")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/cost_curve.png", dpi=120)
plt.close()

# Calibration: kya "30% risk" sach mein ~30% defaults hote hain?
frac_pos, mean_pred = calibration_curve(y_test, p_best, n_bins=6, strategy="quantile")
plt.figure(figsize=(5, 4))
plt.plot([0, 1], [0, 1], "k--", label="Perfect")
plt.plot(mean_pred, frac_pos, "o-", color="#2563eb", label=best_name)
plt.xlabel("Predicted default probability")
plt.ylabel("Observed default rate")
plt.title("Calibration curve")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/calibration.png", dpi=120)
plt.close()

# Feature importance (Random Forest) - "kaunsa feature sabse important?"
rf = fitted["Random Forest"]
names = rf.named_steps["prep"].get_feature_names_out()
imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=names)
top = imp.sort_values(ascending=False).head(12)[::-1]
plt.figure(figsize=(7, 5))
top.plot(kind="barh", color="#4a7")
plt.title("Top 12 features (Random Forest)")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=120)
plt.close()

# ---------------------------------------------------------------- 6. SAVE
raw_X = X.drop(columns=[c for c in ["monthly_payment", "log_credit_amount",
                                    "burden_score", "long_duration",
                                    "young_borrower"] if c in X.columns])
meta = {
    "best_model": best_name,
    "metrics": results,
    "cat_options": {c: sorted(raw_X[c].astype(str).unique().tolist())
                    for c in raw_X.columns if c in cat_cols},
    "num_ranges": {c: [float(raw_X[c].min()), float(raw_X[c].max()),
                       float(raw_X[c].median())]
                   for c in raw_X.columns if c not in cat_cols},
    "raw_columns": raw_X.columns.tolist(),
    "cat_mode": {c: str(raw_X[c].mode()[0]) for c in raw_X.columns if c in cat_cols},
    "cost_threshold": cost_threshold,
    "cost_matrix": {"FN": COST_FN, "FP": COST_FP},
}
joblib.dump(best_model, "models/credit_model.joblib")
with open("models/meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print("\nSaved: models/credit_model.joblib, models/meta.json, outputs/*.png")
