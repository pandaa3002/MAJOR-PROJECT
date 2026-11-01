
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent

# Load the held-out test dataset and true labels
X_test = pd.read_csv(BASE_DIR / "X_test.csv").apply(
    pd.to_numeric, errors="raise"
)
y_test = pd.read_csv(BASE_DIR / "y_test.csv").iloc[:, 0].astype(int)

# Load the saved models
rf_model = joblib.load(BASE_DIR / "random_forest_model.joblib")
mlp_model = load_model(BASE_DIR / "toddler_asd_model.keras")


def evaluate_model(name, model, X, y, is_mlp=False):
    print(f"\n{'=' * 50}")
    print(name)
    print("=" * 50)

    if is_mlp:
        probabilities = model.predict(
            X.to_numpy(dtype=np.float32), verbose=0
        ).ravel()
        predictions = (probabilities >= 0.5).astype(int)
    else:
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

    print(f"Test samples: {len(y)}")
    print(f"Accuracy: {accuracy_score(y, predictions):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y, probabilities):.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y, predictions, labels=[0, 1]))

    print("\nClassification report:")
    print(
        classification_report(
            y, predictions, labels=[0, 1], zero_division=0
        )
    )


evaluate_model("RANDOM FOREST", rf_model, X_test, y_test)
evaluate_model("MLP NEURAL NETWORK", mlp_model, X_test, y_test, is_mlp=True)

print("\nInference test completed.")
