"""
train.py  --  Train ASL Hand Sign CNN (Sign Language MNIST)
Reads CSV from data/, trains a CNN, saves model to artifacts/
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ===== CONFIG =====
DATA_DIR = "data"
TRAIN_CSV = os.path.join(DATA_DIR, "sign_mnist_train.csv")
TEST_CSV = os.path.join(DATA_DIR, "sign_mnist_test.csv")

OUT_DIR = "artifacts"
os.makedirs(OUT_DIR, exist_ok=True)
MODEL_PATH = os.path.join(OUT_DIR, "asl_cnn.h5")

EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 1e-3


# ===== DATA =====
def load_csv(path: str):
    df = pd.read_csv(path)
    y = df["label"].values.astype(np.int64)
    X = df.drop(columns=["label"]).values.astype(np.float32)
    X = X.reshape(-1, 28, 28, 1) / 255.0
    return X, y


print("Loading data...")
X_train, y_train = load_csv(TRAIN_CSV)
X_test, y_test = load_csv(TEST_CSV)
num_classes = int(max(y_train.max(), y_test.max()) + 1)

print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
print(f"  X_test:  {X_test.shape}  y_test:  {y_test.shape}")
print(f"  num_classes: {num_classes}")
print(f"  labels: {sorted(set(y_train))}")


# ===== MODEL =====
def build_model(num_classes: int):
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


print("\nBuilding model...")
model = build_model(num_classes)
model.summary()


# ===== TRAIN =====
callbacks = [
    keras.callbacks.ModelCheckpoint(
        MODEL_PATH, save_best_only=True,
        monitor="val_accuracy", mode="max",
    ),
    keras.callbacks.EarlyStopping(
        patience=5, restore_best_weights=True,
        monitor="val_accuracy", mode="max",
    ),
]

print("\nTraining...")
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
)


# ===== EVALUATE =====
print("\nEvaluating on test set...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"  Test accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

print(f"\nModel saved to: {MODEL_PATH}")
