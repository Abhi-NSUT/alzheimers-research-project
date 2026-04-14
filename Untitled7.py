import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
                                     GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, 
                                     Multiply, Add, Concatenate, Activation)
from tensorflow.keras.models import Model

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

print("2. Shuffling Dataset...")
X_train, y_train = shuffle(X_train, y_train, random_state=42)

print(f"Data successfully loaded and shuffled!")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}\n")

print("====================================")
print("3. BUILDING THE CBAM (SPATIAL & CHANNEL ATTENTION) MODEL...")
print("====================================")

# THE CBAM MECHANISM (Convolutional Block Attention Module)
def cbam_block(input_tensor, ratio=8):
    filters = input_tensor.shape[-1]
    
    # === 1. CHANNEL ATTENTION (WHAT to look at) ===
    # Shared Dense layers
    shared_dense_one = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)
    shared_dense_two = Dense(filters, kernel_initializer='he_normal', use_bias=False)
    
    # Process Avg Pool
    avg_pool = GlobalAveragePooling2D()(input_tensor)
    avg_pool = Reshape((1, 1, filters))(avg_pool)
    avg_out = shared_dense_two(shared_dense_one(avg_pool))
    
    # Process Max Pool
    max_pool = GlobalMaxPooling2D()(input_tensor)
    max_pool = Reshape((1, 1, filters))(max_pool)
    max_out = shared_dense_two(shared_dense_one(max_pool))
    
    # Add them together and activate
    channel_attention = Add()([avg_out, max_out])
    channel_attention = Activation('sigmoid')(channel_attention)
    channel_refined = Multiply()([input_tensor, channel_attention])
    
    # === 2. SPATIAL ATTENTION (WHERE to look) ===
    # Compress the channels down to 2 feature maps (1 average, 1 max)
    avg_pool_spatial = tf.reduce_mean(channel_refined, axis=-1, keepdims=True)
    max_pool_spatial = tf.reduce_max(channel_refined, axis=-1, keepdims=True)
    
    concat = Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])
    
    # Apply a 7x7 Convolution to map the physical shape of the tumor/marker coordinates
    spatial_attention = Conv2D(1, (7, 7), padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)
    spatial_refined = Multiply()([channel_refined, spatial_attention])
    
    # Residual Connection (Crucial so gradients never die!)
    return Add()([input_tensor, spatial_refined])


inputs = Input(shape=(72, 72, 3))

# Block 1 + CBAM Attention
x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
x = cbam_block(x)
x = MaxPooling2D((2, 2))(x)

# Block 2 + CBAM Attention
x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
x = cbam_block(x)
x = MaxPooling2D((2, 2))(x)

# Block 3 + CBAM Attention
x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
x = cbam_block(x)
x = MaxPooling2D((2, 2))(x)

# Classifier
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(3, activation='softmax')(x)

model_attention = Model(inputs=inputs, outputs=outputs)

# --- TRAP PREVENTION: DYNAMIC CLASS WEIGHTING ---
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights_array)}

model_attention.compile(
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
print("4. TRAINING MODEL WITH CBAM ATTENTION...")
print("====================================")

history_attention = model_attention.fit(
    X_train, y_train, 
    epochs=100,           
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

score = model_attention.evaluate(X_test, y_test, verbose=0)
print(f"\nFINAL True Test Accuracy: {score[1]:.4f}")

# Make predictions
test_predictions = model_attention.predict(X_test)
predicted_label = np.argmax(test_predictions, axis=1)

print("\nCBAM Script executed successfully!")
