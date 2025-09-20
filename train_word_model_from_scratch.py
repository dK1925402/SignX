import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# =============================
# Base Project Directory
# =============================
BASE_DIR = r"C:\Users\Admin\Downloads\Indian-Sign-Language-Recognition-pycode"

# =============================
# Paths for Dataset and Outputs
# =============================
X_path = os.path.join(BASE_DIR, "test", "labels", "train_word_input_hands.npy")  # Hand keypoints
y_path = os.path.join(BASE_DIR, "test", "labels", "train_word_labels.txt")       # Labels for each sample

# Output files
MODEL_OUTPUT = os.path.join(BASE_DIR, "test", "model", "word_model_hands.keras")
LABELS_OUTPUT = os.path.join(BASE_DIR, "test", "labels", "word_labels_nodes.npy")

# =============================
# Load Keypoints and Labels
# =============================
print("[INFO] Loading keypoints and labels...")
X = np.load(X_path)

with open(y_path, "r") as f:
    labels = f.read().splitlines()

print(f"[INFO] Dataset loaded: {X.shape[0]} samples, each with {X.shape[1]} features")

# =============================
# Encode Labels
# =============================
print("[INFO] Encoding labels...")
le = LabelEncoder()
y = le.fit_transform(labels)

# Save label classes for later use
np.save(LABELS_OUTPUT, le.classes_)
print(f"[INFO] Labels saved to: {LABELS_OUTPUT}")

# =============================
# Train/Test Split
# =============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print(f"[INFO] Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# =============================
# Build Dense Model
# =============================
model = Sequential([
    Dense(512, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("[INFO] Model compiled successfully!")

# =============================
# Callbacks
# =============================
checkpoint = ModelCheckpoint(MODEL_OUTPUT, monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

# =============================
# Train Model
# =============================
print("[INFO] Starting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[checkpoint, early_stop],
    verbose=1
)

print(f"[INFO] Training complete. Best model saved at: {MODEL_OUTPUT}")
