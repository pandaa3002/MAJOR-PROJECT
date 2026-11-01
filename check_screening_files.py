
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

files_to_check = [
    "toddler_autism.csv",
    "X_train.csv",
    "X_test.csv",
    "y_train.csv",
    "y_test.csv",
    "toddler_asd_model.keras",
    "random_forest_model.joblib",
]

print("\n=== SCREENING FILE CHECK ===")

for filename in files_to_check:
    path = BASE_DIR / filename
    print(f"{filename}: {'FOUND' if path.exists() else 'NOT FOUND'}")

x_train_path = BASE_DIR / "X_train.csv"

if x_train_path.exists():
    X_train = pd.read_csv(x_train_path)

    print("\n=== TRAINING FEATURES ===")
    print("Number of rows:", len(X_train))
    print("Number of features:", len(X_train.columns))
    print("\nFeature columns:")
    for column in X_train.columns:
        print("-", column)

    print("\nAge_Mons values in saved training data:")
    if "Age_Mons" in X_train.columns:
        print(X_train["Age_Mons"].describe())
    else:
        print("Age_Mons column not found.")

print("\n=== CHECK COMPLETE ===")
