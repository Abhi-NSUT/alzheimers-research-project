import os, pickle, numpy as np
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from sklearn.utils import compute_class_weight, shuffle
from sklearn.metrics import accuracy_score

# --- 1. DATA LOADING ---
with open('X_train_full.pkl', 'rb') as f: X_tr = pickle.load(f).astype(np.float32)
with open('X_test_full.pkl', 'rb') as f:  X_te = pickle.load(f).astype(np.float32)
with open('y_train_full.pkl', 'rb') as f: y_tr = pickle.load(f)
with open('y_test_full.pkl', 'rb') as f:  y_te = pickle.load(f)

cw = dict(enumerate(compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)))
os.makedirs("updated_clinical_experts", exist_ok=True)

# --- 2. ARCHITECTURE ---
def build_clin_expert(nf):
    inp = Input(shape=(nf,))
    x = Dense(512, activation='relu')(inp); x = BatchNormalization()(x); x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x); x = BatchNormalization()(x); x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x); x = BatchNormalization()(x); x = Dropout(0.3)(x)
    return Model(inp, Dense(3, activation='softmax')(x))

# --- 3. TRAINING 10-EXPERT SYSTEM ---
all_probs = []
for n in range(10):
    print(f"Training Clinical Expert {n+1}/10...")
    m = build_clin_expert(X_tr.shape[1])
    m.compile(optimizer=keras.optimizers.Adam(5e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    m.fit(X_tr, y_tr, epochs=150, batch_size=32, validation_split=0.1, class_weight=cw, verbose=0, callbacks=[es])
    all_probs.append(m.predict(X_te, verbose=0))
    m.save(f"updated_clinical_experts/clin_{n+1}.h5")
    keras.backend.clear_session()

# --- 4. RESULTS ---
final_acc = accuracy_score(y_te, np.argmax(np.mean(all_probs, axis=0), axis=1))
print(f"\n RE-TRAINING COMPLETE!")
print(f"Update Clinical Ensemble Accuracy: {final_acc:.4f}")
print("Models saved in: updated_clinical_experts/")
