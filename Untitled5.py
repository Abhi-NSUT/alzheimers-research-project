import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

print("====================================")
print("1. LOADING DATA FROM PICKLE FILES...")
print("====================================")

X_train_ = pd.read_pickle("img_train.pkl")["img_array"]
X_test_  = pd.read_pickle("img_test.pkl")["img_array"]

y_train = np.array(pd.read_pickle("img_y_train.pkl")["label"].values.astype(np.float32)).flatten()
y_test  = np.array(pd.read_pickle("img_y_test.pkl")["label"].values.astype(np.float32)).flatten()

# Convert out of pandas arrays
X_train = np.array([x for x in X_train_.values]) 
X_test  = np.array([x for x in X_test_.values]) 

# CRITICAL: Shuffle the dataset so validation_split=0.1 doesn't grab just the majority class
print("2. Shuffling Dataset...")
X_train, y_train = shuffle(X_train, y_train, random_state=42)

print(f"Data successfully loaded and shuffled!")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}\n")

print("====================================")
print("3. BUILDING THE PURE 93% MODEL...")
print("====================================")

model_pure = Sequential([
    keras.Input(shape=(72, 72, 3)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5), # Standard Overfitting Protection
    Dense(3, activation='softmax')
])

# --- TRAP PREVENTION: DYNAMIC CLASS WEIGHTING ---
# Ensure the model is harshly punished for constantly guessing the 54% majority class!
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights_array)}

model_pure.compile(
    # --- TRAP PREVENTION: LOWER LEARNING RATE ---
    # 0.0001 forces Adam to step carefully out of the local minimum instead of blindly crashing!
    optimizer=keras.optimizers.Adam(learning_rate=0.0001), 
    loss="sparse_categorical_crossentropy", 
    metrics=["sparse_categorical_accuracy"] 
)

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=8, 
    restore_best_weights=True
)

print("====================================")
print("4. TRAINING MODEL WITH TRAP PREVENTION...")
print("====================================")

history_pure = model_pure.fit(
    X_train, y_train, 
    epochs=100,           # Given more time because the learning rate is 10x slower
    batch_size=32, 
    validation_split=0.1, 
    class_weight=class_weights_dict, 
    callbacks=[early_stop],
    verbose=1 
)

print("")
print("====================================")
print("5. EVALUATING ACCURACY...")
print("====================================")

score = model_pure.evaluate(X_test, y_test, verbose=0)
print(f"\nFINAL True Test Accuracy: {score[1]:.4f}")

# Make predictions (for potential classification reports or confusion matrices later)
test_predictions = model_pure.predict(X_test)
predicted_label = np.argmax(test_predictions, axis=1)

print("\nDone! Script executed flawlessly.")
