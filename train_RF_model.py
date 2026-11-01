import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)
import joblib
import json
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")

y_train = pd.read_csv("y_train.csv").iloc[:, 0]
y_test = pd.read_csv("y_test.csv").iloc[:, 0]

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("Number of features:", X_train.shape[1])
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
print("\nRandom Forest training completed!")
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)
tn, fp, fn, tp = confusion_matrix(
    y_test, y_pred, labels=[0, 1]
).ravel()

specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

print("\n========== RANDOM FOREST RESULTS ==========")

print(f"Accuracy:    {accuracy:.4f}")
print(f"Precision:   {precision:.4f}")
print(f"Recall:      {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1-score:    {f1:.4f}")
print(f"ROC-AUC:     {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["No ASD Traits", "ASD Traits"],
        zero_division=0
    )
)
# ------------------------------------------------------------
# STEP 7: SAVE MODEL AND EVALUATION METRICS
# ------------------------------------------------------------

joblib.dump(rf_model, "random_forest_model.joblib")

metrics = {
    "model": "Random Forest",
    "n_estimators": 100,
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "specificity": float(specificity),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "true_negatives": int(tn),
    "false_positives": int(fp),
    "false_negatives": int(fn),
    "true_positives": int(tp)
}

with open("rf_baseline_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nModel saved: random_forest_model.joblib")
print("Metrics saved: rf_baseline_metrics.json")

