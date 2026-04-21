import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, Reshape, Multiply, Add
from tensorflow.keras.models import Model

print("1. LOADING DATA FROM PICKLE FILES...")

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

print("3. BUILDING THE DEEP ATTENTION MODEL...")

# THE ATTENTION MECHANISM (Squeeze-and-Excite)
def se_block(input_tensor, ratio=8):
    filters = input_tensor.shape[-1]
    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape((1, 1, filters))(se)
    se = Dense(filters // ratio, activation='relu', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', use_bias=False)(se)
    x = Multiply()([input_tensor, se])
    return Add()([input_tensor, x])  # Residual connection prevents gradient freezing!

inputs = Input(shape=(72, 72, 3))

# Block 1 + Attention
x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
x = se_block(x)
x = MaxPooling2D((2, 2))(x)

# Block 2 + Attention
x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
x = se_block(x)
x = MaxPooling2D((2, 2))(x)

# Block 3 + Attention
x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
x = se_block(x)
x = MaxPooling2D((2, 2))(x)

# Classifier
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(3, activation='softmax')(x)

model_attention = Model(inputs=inputs, outputs=outputs)

# --- TRAP PREVENTION: DYNAMIC CLASS WEIGHTING ---
# Ensure the model is harshly punished for constantly guessing the 54% majority class!
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights_array)}

model_attention.compile(
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

print("4. TRAINING MODEL WITH ATTENTION...")

history_attention = model_attention.fit(
    X_train, y_train, 
    epochs=100,           # Given more time because the learning rate is 10x slower
    batch_size=32, 
    validation_split=0.1, 
    class_weight=class_weights_dict, 
    callbacks=[early_stop],
    verbose=1 
)

print("5. EVALUATING ACCURACY...")

score = model_attention.evaluate(X_test, y_test, verbose=0)
print(f"\nFINAL True Test Accuracy: {score[1]:.4f}")

# Make predictions (for potential classification reports or confusion matrices later)
test_predictions = model_attention.predict(X_test)
predicted_label = np.argmax(test_predictions, axis=1)

