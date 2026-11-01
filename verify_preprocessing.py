
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Locate files in the current project folder
BASE_DIR = Path(__file__).resolve().parent

# --------------------------------------------------
# 1. Load and clean the original dataset
# --------------------------------------------------
df = pd.read_csv(BASE_DIR / "toddler_autism.csv")
df.columns = df.columns.str.strip()

df["Who completed the test"] = (
    df["Who completed the test"]
    .str.strip()
    .str.lower()
)

df = df.drop(columns=["Case_No", "Qchat-10-Score"])

# --------------------------------------------------
# 2. Separate features and target
# --------------------------------------------------
y = (df["Class/ASD Traits"] == "Yes").astype(int)
X = df.drop(columns=["Class/ASD Traits"])

categorical_cols = [
    "Sex",
    "Ethnicity",
    "Jaundice",
    "Family_mem_with_ASD",
    "Who completed the test",
]

X = pd.get_dummies(
    X,
    columns=categorical_cols,
    drop_first=False
)

# --------------------------------------------------
# 3. Recreate the original train-test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# 4. Recreate scaling using training data only
# --------------------------------------------------
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[["Age_Mons"]] = scaler.fit_transform(
    X_train[["Age_Mons"]]
)

X_test_scaled[["Age_Mons"]] = scaler.transform(
    X_test[["Age_Mons"]]
)

# --------------------------------------------------
# 5. Load saved feature datasets as numeric values
# --------------------------------------------------
saved_X_train = pd.read_csv(
    BASE_DIR / "X_train.csv"
).apply(pd.to_numeric, errors="raise")

saved_X_test = pd.read_csv(
    BASE_DIR / "X_test.csv"
).apply(pd.to_numeric, errors="raise")

# --------------------------------------------------
# 6. Verify columns and feature values
# --------------------------------------------------
train_columns_match = (
    list(X_train_scaled.columns)
    == list(saved_X_train.columns)
)

test_columns_match = (
    list(X_test_scaled.columns)
    == list(saved_X_test.columns)
)

train_values_match = (
    train_columns_match
    and X_train_scaled.shape == saved_X_train.shape
    and np.allclose(
        X_train_scaled.to_numpy(dtype=float),
        saved_X_train.to_numpy(dtype=float),
        atol=1e-6
    )
)

test_values_match = (
    test_columns_match
    and X_test_scaled.shape == saved_X_test.shape
    and np.allclose(
        X_test_scaled.to_numpy(dtype=float),
        saved_X_test.to_numpy(dtype=float),
        atol=1e-6
    )
)

# --------------------------------------------------
# 7. Display verification results
# --------------------------------------------------
print("\n=== PREPROCESSING VERIFICATION ===")

print("Training feature columns match:", train_columns_match)
print("Test feature columns match:", test_columns_match)
print("Training features match saved data:", train_values_match)
print("Test features match saved data:", test_values_match)

print("\n=== SCALER INFORMATION ===")
print("Age_Mons scaler mean:", scaler.mean_[0])
print("Age_Mons scaler standard deviation:", scaler.scale_[0])

print("\n=== TARGET LABEL DISTRIBUTION ===")
print(y.value_counts().sort_index().rename(
    index={0: "Class 0", 1: "Class 1"}
))

print("\n=== FINAL STATUS ===")

if all([
    train_columns_match,
    test_columns_match,
    train_values_match,
    test_values_match
]):
    print("SUCCESS: Preprocessing matches the saved datasets.")
else:
    print("CHECK REQUIRED: One or more verification checks failed.")

