# ================= GPU CONFIG =================
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("GPU active")
else:
    print("CPU mode")

# ================= LOAD DATA (SERVER PATH) =================
import numpy as np
import pandas as pd

base_path = "./"   # running inside /workspace/alz-fmri

X_train = np.stack(pd.read_pickle(base_path + "img_train.pkl")["img_array"].values).astype(np.float32) / 255.0
X_test  = np.stack(pd.read_pickle(base_path + "img_test.pkl")["img_array"].values).astype(np.float32) / 255.0

y_train = pd.read_pickle(base_path + "img_y_train.pkl")["label"].values.astype(np.int32)
y_test  = pd.read_pickle(base_path + "img_y_test.pkl")["label"].values.astype(np.int32)

# ================= LABEL FIX =================
y_test[y_test == 2] = -1
y_test[y_test == 1] = 2
y_test[y_test == -1] = 1

y_train[y_train == 2] = -1
y_train[y_train == 1] = 2
y_train[y_train == -1] = 1

print("Shapes:", X_train.shape, y_train.shape)

# ================= MODEL =================
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, GlobalAveragePooling2D,
    Reshape, Multiply
)

def squeeze_excite_block(input_tensor, ratio=8):
    filters = input_tensor.shape[-1]
    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape((1, 1, filters))(se)
    se = Dense(filters // ratio, activation='relu', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', use_bias=False)(se)
    return Multiply()([input_tensor, se])

inputs = Input(shape=(72, 72, 3))

x = Conv2D(64, (3, 3), activation='relu')(inputs)
x = squeeze_excite_block(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu')(x)
x = squeeze_excite_block(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(256, (3, 3), activation='relu')(x)
x = squeeze_excite_block(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(3, activation='softmax')(x)

model_attention = Model(inputs=inputs, outputs=outputs)

model_attention.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"]
)

# ================= TRAIN =================
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=6,
    restore_best_weights=True
)

history = model_attention.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,   # increased for GPU
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# ================= EVALUATE =================
score = model_attention.evaluate(X_test, y_test, verbose=0)
print("Attention Model Accuracy:", score[1])