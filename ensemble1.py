import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, Reshape, Multiply, Add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

print("====================================")
print("1. LOADING & PREPARING DATA...")
print("====================================")

X_train_ = pd.read_pickle("img_train.pkl")["img_array"]
X_test_  = pd.read_pickle("img_test.pkl")["img_array"]

y_train = np.array(pd.read_pickle("img_y_train.pkl")["label"].values.astype(np.float32)).flatten()
y_test  = np.array(pd.read_pickle("img_y_test.pkl")["label"].values.astype(np.float32)).flatten()

X_train = np.array([x for x in X_train_.values]) 
X_test  = np.array([x for x in X_test_.values]) 

X_train, y_train = shuffle(X_train, y_train, random_state=42)

# --- THE 94% TRICK 1: LABEL SMOOTHING (SOFT PROBABILITIES) ---
print("Applying One-Hot Encoding to support Label Smoothing...")
y_train_ohe = to_categorical(y_train, num_classes=3)
y_test_ohe = to_categorical(y_test, num_classes=3)

class_weights_array = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights_array)}

print("====================================")
print("2. DEFINING THE SE-ATTENTION MODEL W/ REGULARIZATION...")
print("====================================")

def build_se_model():
    def se_block(input_tensor, ratio=8):
        filters = input_tensor.shape[-1]
        se = GlobalAveragePooling2D()(input_tensor)
        se = Reshape((1, 1, filters))(se)
        se = Dense(filters // ratio, activation='relu', use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', use_bias=False)(se)
        x = Multiply()([input_tensor, se])
        return Add()([input_tensor, x]) 
    
    inputs = Input(shape=(72, 72, 3))
    
    # --- THE 94% TRICK 2: L2 WEIGHT DECAY ---
    # Shrinks over-confident weights actively during training, killing exact-pixel memorization
    x = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001))(inputs)
    x = se_block(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001))(x)
    x = se_block(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001))(x)
    x = se_block(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Flatten()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(3, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # --- THE 94% TRICK 3: COSINE DECAY RESTARTS ---
    # Acts like a defibrillator, periodically jumping the learning rate up to clear out of deep local minimums
    lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=0.0003, 
        first_decay_steps=15
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule), 
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1), # The Magic Smoothing Math
        metrics=["categorical_accuracy"] 
    )
    return model

print("====================================")
print("3. TRAINING THE 5x ENSEMBLE SYSTEM...")
print("====================================")

# --- THE 94% TRICK 4: INCREASE EXPERT COUNT ---
NUM_MODELS = 5
all_test_predictions = []

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

for i in range(NUM_MODELS):
    print(f"\n---> TRAINING MODEL {i+1} STARTING NOW <---")
    
    keras.backend.clear_session()
    
    model = build_se_model()
    
    model.fit(
        X_train, y_train_ohe, 
        epochs=100,           
        batch_size=32, 
        validation_split=0.1, 
        class_weight=class_weights_dict, 
        callbacks=[early_stop],
        verbose=1 
    )
    
    print(f"\nModel {i+1} finished! Generating prediction probabilities...")
    test_preds = model.predict(X_test)
    all_test_predictions.append(test_preds)

print("\n====================================")
print("4. AVERAGING THE EXPERTS...")
print("====================================")

all_test_predictions = np.array(all_test_predictions)
ensemble_probabilities = np.mean(all_test_predictions, axis=0)
ensemble_final_labels = np.argmax(ensemble_probabilities, axis=1)

y_test_integers = np.argmax(y_test_ohe, axis=1)

final_accuracy = accuracy_score(y_test_integers, ensemble_final_labels)

print(f"\n🎉 ULTIMATE 5-MODEL ENSEMBLE TEST ACCURACY (LABEL-SMOOTHED): {final_accuracy:.4f} 🎉")
