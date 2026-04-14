import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, UpSampling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.applications import DenseNet121

print("====================================")
print("1. LOADING & PREPARING DATA...")
print("====================================")

X_train_ = pd.read_pickle("img_train.pkl")["img_array"]
X_test_  = pd.read_pickle("img_test.pkl")["img_array"]

y_train = np.array(pd.read_pickle("img_y_train.pkl")["label"].values.astype(np.float32)).flatten()
y_test  = np.array(pd.read_pickle("img_y_test.pkl")["label"].values.astype(np.float32)).flatten()

X_train = np.array([x for x in X_train_.values]) 
X_test  = np.array([x for x in X_test_.values]) 

# NO PIXEL DIVISION! Your fMRI Z-Score arrays are kept pristine.

X_train, y_train = shuffle(X_train, y_train, random_state=42)

class_weights_array = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights_array)}

print("====================================")
print("2. DEFINING THE TRANSFER LEARNING MODEL (DenseNet121)...")
print("====================================")

def build_densenet_model():
    inputs = Input(shape=(72, 72, 3))
    
    # 1. UPSAMPLING: Stretch the tiny 72x72 to a massive 216x216 so DenseNet can physically see the edges
    x = UpSampling2D(size=(3, 3), interpolation='bilinear')(inputs)
    
    # 2. BATCH NORMALIZATION: This mathematically re-centers your unscaled fMRI data (Z-Scores)
    # perfectly so the ImageNet weights don't get completely confused or destroyed!
    x = BatchNormalization()(x)
    
    # 3. HEAVY DATA AUGMENTATION (Anti-Overfitting against a massive Model)
    # Removing RandomContrast because TensorFlow's AdjustContrastv2 has a known internal bug preventing backpropagation
    x = keras.layers.RandomFlip("horizontal")(x)
    x = keras.layers.RandomRotation(0.05)(x)
    
    # 4. LOAD PRE-TRAINED BEHEMOTH (DenseNet121)
    # CRITICAL FIX: Do NOT pass input_tensor=x because Keras weight loaders get confused by the Augmentation layers!
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(216, 216, 3))
    
    # We fine-tune the entire beast, but VERY gently using a microscopic Learning Rate.
    base_model.trainable = True 
    
    # 5. ATTACH CUSTOM MEDICAL CLASSIFIER
    # Pass your compiled Image variable `x` physically INTO the base_model!
    out = base_model(x)
    out = GlobalAveragePooling2D()(out)
    out = Dense(256, activation='relu')(out)
    out = Dropout(0.5)(out)
    outputs = Dense(3, activation='softmax')(out)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        # EXACTLY 0.0001. If this is higher, you instantly wipe out the pre-trained ImageNet intelligence!
        optimizer=keras.optimizers.Adam(learning_rate=0.0001), 
        loss="sparse_categorical_crossentropy", 
        metrics=["sparse_categorical_accuracy"] 
    )
    return model

print("====================================")
print("3. TRAINING THE ENSEMBLE (3 DENSENET EXPERTS)...")
print("====================================")

NUM_MODELS = 3
all_test_predictions = []

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

for i in range(NUM_MODELS):
    print(f"\n---> TRAINING DENSENET EXPERT {i+1} STARTING NOW <---")
    
    # Extremely important so your Server's GPU memory doesn't violently overload during the loop
    keras.backend.clear_session()
    
    model = build_densenet_model()
    
    model.fit(
        X_train, y_train, 
        epochs=50,      # DenseNet hits its peak accuracy much faster than custom CNNs      
        batch_size=16,  # LOWERED TO 16! DenseNet hogs massive GPU RAM. 32 will likely crash an online server. 
        validation_split=0.1, 
        class_weight=class_weights_dict, 
        callbacks=[early_stop],
        verbose=1 
    )
    
    print(f"\nModel {i+1} finished! Generating prediction probabilities...")
    test_preds = model.predict(X_test, batch_size=16)
    all_test_predictions.append(test_preds)

print("\n====================================")
print("4. AVERAGING THE DENSENET EXPERTS...")
print("====================================")

# Convert list of predictions to a 3D Numpy Array
all_test_predictions = np.array(all_test_predictions)

# The absolute Magic: Average the probabilities of all 3 Massive models!
ensemble_probabilities = np.mean(all_test_predictions, axis=0)

# The final class is simply the one with the highest averaged probability
ensemble_final_labels = np.argmax(ensemble_probabilities, axis=1)

final_accuracy = accuracy_score(y_test, ensemble_final_labels)
print(f"\n🏆 ULTIMATE 216x216 DENSENET ENSEMBLE ACCURACY: {final_accuracy:.4f} 🏆")
