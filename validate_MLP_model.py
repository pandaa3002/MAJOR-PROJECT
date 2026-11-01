
# ============================================================
# MLP — STRATIFIED 5-FOLD CROSS-VALIDATION
# ============================================================

import numpy as np
import pandas as pd
import tensorflow as tf
import json

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ------------------------------------------------------------
# STEP 1: LOAD TRAINING DATA ONLY
# ------------------------------------------------------------

X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").iloc[:, 0].to_numpy()

X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").iloc[:, 0].to_numpy()

if list(X_train.columns) != list(X_test.columns):
    raise ValueError("Training and testing feature columns do not match.")

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

print("Training samples:", len(X_train))
print("Test samples (reserved for final evaluation):", len(X_test))
print("Number of features:", X_train.shape[1])


# ------------------------------------------------------------
# STEP 2: DEFINE THE MLP MODEL
# ------------------------------------------------------------

def build_mlp(n_features):
    model = Sequential([
        Input(shape=(n_features,)),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ------------------------------------------------------------
# STEP 3: SET UP STRATIFIED 5-FOLD CROSS-VALIDATION
# ------------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

metric_names = [
    "accuracy", "precision", "recall",
    "specificity", "f1", "roc_auc"
]

results = {metric: [] for metric in metric_names}


# ------------------------------------------------------------
# STEP 4: TRAIN AND VALIDATE ON EACH FOLD
# ------------------------------------------------------------

for fold, (train_idx, val_idx) in enumerate(
    cv.split(X_train, y_train), start=1
):
    print(f"\n========== FOLD {fold} ==========")

    # Reset the random seed for reproducibility
    tf.keras.utils.set_random_seed(42 + fold)

    X_fold_train = X_train.iloc[train_idx].copy()
    X_fold_val = X_train.iloc[val_idx].copy()

    y_fold_train = y_train[train_idx]
    y_fold_val = y_train[val_idx]

    # Fit scaling on this fold's training portion only
    scaler = StandardScaler()
    numeric_cols = ["Age_Mons"]

    X_fold_train.loc[:, numeric_cols] = scaler.fit_transform(
        X_fold_train[numeric_cols]
    )
    X_fold_val.loc[:, numeric_cols] = scaler.transform(
        X_fold_val[numeric_cols]
    )

    # Calculate class weights from this fold's training labels only
    classes = np.unique(y_fold_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_fold_train
    )

    class_weights = {
        int(label): float(weight)
        for label, weight in zip(classes, weights)
    }

    model = build_mlp(X_fold_train.shape[1])

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    model.fit(
        X_fold_train,
        y_fold_train,
        validation_data=(X_fold_val, y_fold_val),
        epochs=100,
        batch_size=32,
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=0
    )

    # Predict probabilities and class labels
    probabilities = model.predict(
        X_fold_val, verbose=0
    ).ravel()

    predictions = (probabilities >= 0.5).astype(int)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_fold_val, predictions)
    precision = precision_score(
        y_fold_val, predictions, zero_division=0
    )
    recall = recall_score(
        y_fold_val, predictions, zero_division=0
    )
    f1 = f1_score(
        y_fold_val, predictions, zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_fold_val, predictions, labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    auc = (
        roc_auc_score(y_fold_val, probabilities)
        if len(np.unique(y_fold_val)) == 2
        else float("nan")
    )

    fold_metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": auc
    }

    for metric, value in fold_metrics.items():
        results[metric].append(float(value))

    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1-score:    {f1:.4f}")
    print(f"ROC-AUC:     {auc:.4f}")

    # Release the current model before starting the next fold
    del model


# ------------------------------------------------------------
# STEP 5: REPORT MEAN AND STANDARD DEVIATION
# ------------------------------------------------------------

print("\n===== MLP 5-FOLD CROSS-VALIDATION SUMMARY =====")

summary = {}

for metric, scores in results.items():
    mean_score = float(np.nanmean(scores))
    std_score = float(np.nanstd(scores, ddof=1))

    summary[metric] = {
        "fold_scores": scores,
        "mean": mean_score,
        "std": std_score
    }

    print(
        f"{metric.capitalize():<12}: "
        f"{mean_score:.4f} +/- {std_score:.4f}"
    )


# ------------------------------------------------------------
# STEP 6: SAVE RESULTS
# ------------------------------------------------------------

with open("mlp_cross_validation.json", "w") as f:
    json.dump(summary, f, indent=4)

print("\nSaved: mlp_cross_validation.json")
print("Cross-validation completed.")
