# Required imports
import os
import random
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from tensorflow.keras.layers import Dense, Dropout, MaxPooling2D, Flatten, Conv2D, BatchNormalization

# ================= GPU CONFIG =================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("GPU detected and configured")
else:
    print("Running on CPU")

# ================= SEED =================
def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

reset_random_seeds(42)

# ================= LOAD DATA (FIXED) =================
X_train_df = pd.read_pickle("img_train.pkl")
X_test_df  = pd.read_pickle("img_test.pkl")

y_train_df = pd.read_pickle("img_y_train.pkl")
y_test_df  = pd.read_pickle("img_y_test.pkl")

# Extract columns (adjust if names differ)
X_train_ = X_train_df["img_array"]
X_test_  = X_test_df["img_array"]

y_train = y_train_df["label"].values.astype(np.int32)
y_test  = y_test_df["label"].values.astype(np.int32)

# Convert to numpy (vectorized, faster)
X_train = np.stack(X_train_.values) / 255.0
X_test  = np.stack(X_test_.values) / 255.0

print("Data shapes:", X_train.shape, y_train.shape)

# ================= MODEL =================
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(72, 72, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.35),

    Conv2D(512, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.4),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(3, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"]
)

model.summary()

# ================= TRAIN =================
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,   # increased for GPU
    validation_split=0.1,
    verbose=1
)

# ================= EVALUATE =================
score = model.evaluate(X_test, y_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

# ================= PREDICTION =================
preds = model.predict(X_test)
predicted_label = np.argmax(preds, axis=1)

print("Classification Report:")
print(classification_report(y_test, predicted_label))

# ================= PLOT =================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['sparse_categorical_accuracy'])
plt.plot(history.history['val_sparse_categorical_accuracy'])
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')

plt.show()