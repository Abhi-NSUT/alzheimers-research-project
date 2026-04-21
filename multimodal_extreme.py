import os, random, time, pickle, numpy as np, pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from sklearn.utils import compute_class_weight, shuffle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- SETUP ---
ACTIVE_SEED = int(time.time() * 1000) % 100000 
os.environ['PYTHONHASHSEED'] = str(ACTIVE_SEED)
np.random.seed(ACTIVE_SEED); random.seed(ACTIVE_SEED); tf.random.set_seed(ACTIVE_SEED)

# --- 1. DATA LOADING ---
print("1. LOADING DATASETS...")

# Imaging (8,063 scans)
X_img_tr_f = pd.read_pickle("img_train.pkl")["img_array"]
X_img_te_f = pd.read_pickle("img_test.pkl")["img_array"]
y_img_tr   = pd.read_pickle("img_y_train.pkl")["label"].values.astype(int)
y_img_te   = pd.read_pickle("img_y_test.pkl")["label"].values.astype(int)

# Use full float32 precision for maximum accuracy
X_img_tr = np.array([x for x in X_img_tr_f.values], dtype=np.float32)
X_img_te = np.array([x for x in X_img_te_f.values], dtype=np.float32)

# Full Clinical (Built by build_full_clinical_data.py)
with open('X_train_full.pkl', 'rb') as f: X_cl_tr = pickle.load(f).astype(np.float32)
with open('X_test_full.pkl', 'rb') as f:  X_cl_te = pickle.load(f).astype(np.float32)
with open('y_train_full.pkl', 'rb') as f: y_cl_tr = pickle.load(f)
with open('y_test_full.pkl', 'rb') as f:  y_cl_te = pickle.load(f)

X_img_tr, y_img_tr = shuffle(X_img_tr, y_img_tr, random_state=42)
X_cl_tr, y_cl_tr = shuffle(X_cl_tr, y_cl_tr, random_state=42)

cw_img = dict(enumerate(compute_class_weight('balanced', classes=np.unique(y_img_tr), y=y_img_tr)))
cw_cl  = dict(enumerate(compute_class_weight('balanced', classes=np.unique(y_cl_tr), y=y_cl_tr)))

# --- 2. ARCHITECTURES (STATE-OF-THE-ART) ---
def build_img_expert():
    def se_block(t, f):
        s = GlobalAveragePooling2D()(t)
        s = Reshape((1, 1, f))(s)
        s = Dense(f // 8, activation='relu', use_bias=False)(s)
        s = Dense(f, activation='sigmoid', use_bias=False)(s)
        return Add()([t, Multiply()([t, s])])
    
    inp = Input(shape=(72, 72, 3))
    x = Conv2D(64, (3,3), padding='same', activation='relu')(inp); x = se_block(x, 64); x = MaxPooling2D()(x)
    x = Conv2D(128, (3,3), padding='same', activation='relu')(x); x = se_block(x, 128); x = MaxPooling2D()(x)
    x = Conv2D(256, (3,3), padding='same', activation='relu')(x); x = se_block(x, 256); x = MaxPooling2D()(x)
    x = Flatten()(x); x = Dense(128, activation='relu')(x); x = Dropout(0.5)(x)
    return Model(inp, Dense(3, activation='softmax')(x))

def build_clin_expert(nf):
    inp = Input(shape=(nf,))
    x = Dense(512, activation='relu')(inp);  x = BatchNormalization()(x); x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x);  x = BatchNormalization()(x); x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x);  x = BatchNormalization()(x); x = Dropout(0.3)(x)
    return Model(inp, Dense(3, activation='softmax')(x))

# --- 3. TRAINING 20-MODEL ENSEMBLE ---
NUM_MODELS = 10 
S_DIR = f"final_research_models_{ACTIVE_SEED}"
os.makedirs(S_DIR, exist_ok=True)

print(f"2. TRAINING (10 IMAGING + 10 CLINICAL)...")
i_probs, c_probs = [], []
es_img = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
es_cl = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

for n in range(NUM_MODELS):
    print(f"Training Expert Round {n+1}/10...")
    # Imaging Expert
    m1 = build_img_expert()
    m1.compile(optimizer=keras.optimizers.Adam(1e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    m1.fit(X_img_tr, y_img_tr, epochs=100, batch_size=32, validation_split=0.1, class_weight=cw_img, verbose=0, callbacks=[es_img])
    i_probs.append(m1.predict(X_img_te, verbose=0))
    # Clinical Expert
    m2 = build_clin_expert(X_cl_tr.shape[1])
    m2.compile(optimizer=keras.optimizers.Adam(5e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    m2.fit(X_cl_tr, y_cl_tr, epochs=150, batch_size=64, validation_split=0.1, class_weight=cw_cl, verbose=0, callbacks=[es_cl])
    c_probs.append(m2.predict(X_cl_te, verbose=0))
    # Save & Clear
    m1.save(f"{S_DIR}/img_{n+1}.h5"); m2.save(f"{S_DIR}/clin_{n+1}.h5")
    del m1, m2; keras.backend.clear_session()

# --- 4. ULTIMATE RESULTS ---
i_acc = accuracy_score(y_img_te, np.argmax(np.mean(i_probs, axis=0), axis=1))
c_acc = accuracy_score(y_cl_te, np.argmax(np.mean(c_probs, axis=0), axis=1))

print(f"\n RESEARCH RESULTS COMPILED:")
print(f"Imaging Ensemble Accuracy  : {i_acc:.4f}  (Highest Achievement)")
print(f"Clinical Ensemble Accuracy : {c_acc:.4f}  (Highest Achievement)")
print(f"Models Saved In: {S_DIR}/")
