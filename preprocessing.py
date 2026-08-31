"""
==========================================================================
STEP-BY-STEP DATA PREPROCESSING
Toddler Autism Screening (Q-CHAT-10) Dataset
==========================================================================
Goal: turn the raw CSV into clean, numeric, properly-split arrays that a
neural network can actually train on.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

RANDOM_STATE = 42   # fixed seed -> same random split every time we re-run this


# --------------------------------------------------------------------
# STEP 1: LOAD THE DATA AND FIX MESSY TEXT
# --------------------------------------------------------------------
def load_and_clean(path="toddler_autism.csv"):
    df = pd.read_csv(path)

    # The original column name is "Class/ASD Traits " with a trailing space.
    # .strip() on all column names removes accidental whitespace so we can
    # refer to columns reliably by name later.
    df.columns = df.columns.str.strip()

    # "Health Care Professional" and "Health care professional" are the same
    # category typed with different capitalization. If we don't fix this now,
    # the encoding step (Step 3) will treat them as two separate categories,
    # artificially splitting real data and confusing the model.
    df["Who completed the test"] = df["Who completed the test"].str.strip().str.lower()

    # Case_No is just a row ID (1, 2, 3, ...) - it has no relationship to
    # autism traits, so it would only add noise if left in.
    #
    # Qchat-10-Score is the SUM of A1..A10, and the label itself is created
    # by thresholding this score (score > 3 -> "Yes"). Keeping it would let
    # the model just re-learn that threshold rule instead of learning from
    # actual behavioral answers. This is called DATA LEAKAGE: a feature that
    # gives away the answer directly, making results look artificially
    # perfect while the model learns nothing generalizable.
    df = df.drop(columns=["Case_No", "Qchat-10-Score"])

    return df


# --------------------------------------------------------------------
# STEP 2: SEPARATE FEATURES (X) FROM THE TARGET LABEL (y)
# --------------------------------------------------------------------
def build_features(df):
    # Convert the text label ("Yes"/"No") into 1/0. Neural networks (and
    # basically all ML models) need numeric targets, not strings.
    y = (df["Class/ASD Traits"] == "Yes").astype(int)

    X = df.drop(columns=["Class/ASD Traits"])

    categorical_cols = ["Sex", "Ethnicity", "Jaundice",
                         "Family_mem_with_ASD", "Who completed the test"]

    # ONE-HOT ENCODING: turns one text column with N categories into N
    # separate 0/1 columns. Example: Ethnicity="asian" becomes
    # Ethnicity_asian=1 and every other Ethnicity_* column=0.
    #
    # Why not just assign each category a number (asian=1, black=2, ...)?
    # Because that would imply a false ORDER and DISTANCE between
    # categories (as if black - asian = white_european - black), which
    # means nothing for a nominal (unordered) category like ethnicity. A
    # neural net could wrongly try to learn from that fake numeric
    # relationship. One-hot encoding avoids inventing an order that isn't
    # there.
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)

    numeric_cols = ["Age_Mons"]   # the one truly continuous column
    return X, y, numeric_cols


# --------------------------------------------------------------------
# STEP 3: SPLIT INTO TRAIN / TEST, THEN SCALE
# --------------------------------------------------------------------
def split_and_scale(X, y, numeric_cols, test_size=0.2):
    # STRATIFIED SPLIT: our label is imbalanced (~69% Yes / 31% No). A plain
    # random split could by chance put too many or too few "No" cases into
    # the test set, making evaluation unreliable. stratify=y forces both the
    # train and test sets to keep the same 69/31 ratio as the full dataset.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    # FEATURE SCALING: neural networks train via gradient descent, which
    # works much better when all input features are on a similar numeric
    # scale. Age_Mons ranges from 12-36 (raw months), while every other
    # feature is already 0 or 1. Left unscaled, Age_Mons would dominate the
    # early gradients simply because its numbers are bigger - not because
    # it's actually more important.
    #
    # StandardScaler transforms a column to have mean=0 and standard
    # deviation=1: new_value = (value - mean) / std_dev.
    #
    # CRITICAL RULE: fit the scaler on TRAINING data only (scaler.fit_transform
    # on X_train), then reuse that exact same scaler on the test data
    # (scaler.transform on X_test, no fit). If we fit on the full dataset
    # before splitting, information about the test set's distribution
    # "leaks" into training - a subtler cousin of the leakage problem from
    # Step 1. The test set must stay something the model (and its
    # preprocessing) has never seen.
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test, y_train, y_test, scaler


# --------------------------------------------------------------------
# STEP 4: HANDLE CLASS IMBALANCE
# --------------------------------------------------------------------
def get_class_weights(y_train):
    # Our training set is ~69% "Yes" / ~31% "No". Left alone, a model can
    # get deceptively high accuracy just by leaning toward predicting the
    # majority class "Yes" for almost everything - similar to what happened
    # with the face-image model earlier, but here we can actually fix it,
    # because there IS a real signal in the features.
    #
    # compute_class_weight("balanced", ...) gives each class a weight
    # inversely proportional to how often it appears, so the loss function
    # penalizes a wrong "No" prediction more heavily than a wrong "Yes"
    # prediction, since "No" is rarer. This nudges the model to actually pay
    # attention to the minority class instead of ignoring it.
    weights = compute_class_weight(
        class_weight="balanced", classes=np.array([0, 1]), y=y_train
    )
    return {0: weights[0], 1: weights[1]}


# --------------------------------------------------------------------
# RUN EVERYTHING AND SHOW WHAT HAPPENED AT EACH STEP
# --------------------------------------------------------------------
if __name__ == "__main__":
    df = load_and_clean()
    print(f"[Step 1] Loaded {df.shape[0]} rows, {df.shape[1]} columns after cleaning + dropping leaky/ID columns")

    X, y, numeric_cols = build_features(df)
    print(f"[Step 2] Encoded categoricals -> {X.shape[1]} total numeric features")

    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y, numeric_cols)
    print(f"[Step 3] Split into train={X_train.shape[0]} rows / test={X_test.shape[0]} rows (stratified)")
    print(f"         Train label ratio: {y_train.mean():.3f} | Test label ratio: {y_test.mean():.3f}")

    class_weights = get_class_weights(y_train)
    print(f"[Step 4] Class weights for training: {class_weights}")

    # Save the processed arrays so the training script can load them directly
    X_train.to_csv("X_train.csv", index=False)
    X_test.to_csv("X_test.csv", index=False)
    y_train.to_csv("y_train.csv", index=False)
    y_test.to_csv("y_test.csv", index=False)
    print("\nSaved: X_train.csv, X_test.csv, y_train.csv, y_test.csv")
