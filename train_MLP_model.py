
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. Reproducibility
tf.keras.utils.set_random_seed(42)

# 2. Load processed data
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")

y_train = pd.read_csv("y_train.csv").iloc[:, 0].to_numpy()
y_test = pd.read_csv("y_test.csv").iloc[:, 0].to_numpy()

# Verify that the train/test features match
if list(X_train.columns) != list(X_test.columns):
    raise ValueError("Training and testing feature columns do not match.")

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)
print("Training labels:", np.unique(y_train, return_counts=True))

# 3. Calculate weights from training labels only
classes = np.unique(y_train)
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weights = {
    int(label): float(weight)
    for label, weight in zip(classes, weights)
}
print("Class weights:", class_weights)

# 4. Build the neural network
model = Sequential([
    Input(shape=(X_train.shape[1],)),
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

model.summary()

# 5. Train the model
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stopping],
    verbose=1
)

# 6. Evaluate using the held-out test set
test_loss, test_accuracy = model.evaluate(
    X_test, y_test, verbose=0
)

probabilities = model.predict(X_test, verbose=0).ravel()
predictions = (probabilities >= 0.5).astype(int)

print("\nTest loss:", round(float(test_loss), 4))
print("Test accuracy:", round(float(test_accuracy), 4))

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["No label", "ASD traits label"],
    zero_division=0
))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

if len(np.unique(y_test)) == 2:
    print("ROC-AUC:", round(
        roc_auc_score(y_test, probabilities), 4
    ))

# 7. Save the trained model
model.save("toddler_asd_model.keras")
print("\nSaved model: toddler_asd_model.keras")