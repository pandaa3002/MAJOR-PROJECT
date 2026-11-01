# ============================================================
# RANDOM FOREST — STRATIFIED 5-FOLD CROSS-VALIDATION
# ============================================================

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

# ------------------------------------------------------------
# STEP 1: LOAD TRAINING DATA ONLY
# ------------------------------------------------------------

X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").iloc[:, 0]

print("Training samples:", len(X_train))
print("Features:", X_train.shape[1])

# ------------------------------------------------------------
# STEP 2: DEFINE THE BASELINE MODEL
# ------------------------------------------------------------

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# ------------------------------------------------------------
# STEP 3: SET UP STRATIFIED 5-FOLD CROSS-VALIDATION
# ------------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

# ------------------------------------------------------------
# STEP 4: RUN CROSS-VALIDATION
# ------------------------------------------------------------

results = cross_validate(
    rf_model,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring,
    n_jobs=-1,
    return_train_score=False
)

# ------------------------------------------------------------
# STEP 5: REPORT MEAN AND STANDARD DEVIATION
# ------------------------------------------------------------

print("\n===== 5-FOLD CROSS-VALIDATION RESULTS =====")

summary = {}

for metric in scoring:
    scores = results[f"test_{metric}"]

    mean_score = np.mean(scores)
    std_score = np.std(scores, ddof=1)

    summary[metric] = {
        "fold_scores": scores.tolist(),
        "mean": float(mean_score),
        "std": float(std_score)
    }

    print(
        f"{metric.capitalize():<12}: "
        f"{mean_score:.4f} +/- {std_score:.4f}"
    )

# Save the validation results
import json

with open("rf_cross_validation.json", "w") as f:
    json.dump(summary, f, indent=4)

print("\nSaved: rf_cross_validation.json")
from sklearn.metrics import make_scorer, confusion_matrix

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return tn / (tn + fp) if (tn + fp) > 0 else 0.0