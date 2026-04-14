import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, Reshape, Multiply, Add
from tensorflow.keras.models import Model

# Setup aesthetics for plots
sns.set_theme(style="whitegrid")

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

class_weights_array = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights_array)}

print("====================================")
print("2. DEFINING THE SE-ATTENTION MODEL (92.64% BASELINE)...")
print("====================================")

# We are using the pure 3-model SE Engine without the 5x L2 Regularization
# Because this is what gave us the absolute best 92.64% score!
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
    
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = se_block(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = se_block(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = se_block(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(3, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001), 
        loss="sparse_categorical_crossentropy", 
        metrics=["sparse_categorical_accuracy"] 
    )
    return model

print("====================================")
print("3. TRAINING THE 3x ENSEMBLE SYSTEM...")
print("====================================")

NUM_MODELS = 3
all_test_predictions = []
all_histories = [] # NEW: We need to capture the Epoch training history!

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

for i in range(NUM_MODELS):
    print(f"\n---> TRAINING MODEL {i+1} STARTING NOW <---")
    
    keras.backend.clear_session()
    model = build_se_model()
    
    history_obj = model.fit(
        X_train, y_train, 
        epochs=100,           
        batch_size=32, 
        validation_split=0.1, 
        class_weight=class_weights_dict, 
        callbacks=[early_stop],
        verbose=1 
    )
    all_histories.append(history_obj.history)
    
    print(f"\nModel {i+1} finished! Generating prediction probabilities...")
    test_preds = model.predict(X_test)
    all_test_predictions.append(test_preds)

print("\n====================================")
print("4. AVERAGING THE EXPERTS...")
print("====================================")

all_test_predictions = np.array(all_test_predictions)
ensemble_probabilities = np.mean(all_test_predictions, axis=0)
ensemble_final_labels = np.argmax(ensemble_probabilities, axis=1)

final_accuracy = accuracy_score(y_test, ensemble_final_labels)
print(f"\n🎉 ULTIMATE ENSEMBLE TEST ACCURACY: {final_accuracy:.4f} 🎉")

print("\n====================================")
print("5. GENERATING MAXIMUM VISUALIZATION GRAPHS...")
print("====================================")
class_names = ['Class 0', 'Class 1', 'Class 2']

# 1. CONFUSION MATRIX HEATMAP
try:
    cm = confusion_matrix(y_test, ensemble_final_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                linewidths=1, linecolor='black')
    plt.title('Ensemble Confusion Matrix', fontsize=15, fontweight='bold', pad=15)
    plt.ylabel('True Actual Label', fontsize=12, fontweight='bold')
    plt.xlabel('Model Predicted Label', fontsize=12, fontweight='bold')
    plt.savefig("1_confusion_matrix.png", bbox_inches='tight')
    print("-> Saved 1_confusion_matrix.png")
except Exception as e:
    print(f"Failed to plot Confusion Matrix: {e}")

# 2. CLASSIFICATION REPORT HEATMAP
try:
    report = classification_report(y_test, ensemble_final_labels, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report_plot = df_report.drop(['support'], axis=1)
    if 'accuracy' in df_report_plot.index:
        df_report_plot = df_report_plot.drop(['accuracy'])

    plt.figure(figsize=(8, 5))
    sns.heatmap(df_report_plot, annot=True, cmap='viridis', fmt='.3f', linewidths=2, linecolor='black')
    plt.title('Classification Report Heatmap', fontsize=15, fontweight='bold', pad=15)
    plt.savefig("2_classification_report.png", bbox_inches='tight')
    print("-> Saved 2_classification_report.png")
except Exception as e:
    print(f"Failed to plot Classification Report: {e}")

# 3. ROC CURVES (Multiclass)
try:
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    plt.figure(figsize=(10, 7))
    colors = ['blue', 'red', 'green']
    
    for i, color in zip(range(3), colors):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], ensemble_probabilities[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'ROC curve class {i} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold', fontsize=12)
    plt.ylabel('True Positive Rate', fontweight='bold', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontweight='bold', fontsize=15, pad=15)
    plt.legend(loc="lower right")
    plt.savefig("3_roc_curve.png", bbox_inches='tight')
    print("-> Saved 3_roc_curve.png")
except Exception as e:
    print(f"Failed to plot ROC Curve: {e}")

# 4. PRECISION-RECALL CURVES
try:
    plt.figure(figsize=(10, 7))
    colors = ['blue', 'red', 'green']
    for i, color in zip(range(3), colors):
        precision_c, recall_c, _ = precision_recall_curve(y_test_bin[:, i], ensemble_probabilities[:, i])
        plt.plot(recall_c, precision_c, color=color, lw=2, label=f'PR curve class {i}')
    plt.xlabel('Recall', fontweight='bold', fontsize=12)
    plt.ylabel('Precision', fontweight='bold', fontsize=12)
    plt.title('Precision-Recall (PR) Curve', fontweight='bold', fontsize=15, pad=15)
    plt.legend(loc="lower left")
    plt.savefig("4_pr_curve.png", bbox_inches='tight')
    print("-> Saved 4_pr_curve.png")
except Exception as e:
    print(f"Failed to plot PR Curve: {e}")

# 5. OVERALL METRICS BAR GRAPH
try:
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    latest_metrics = [
        final_accuracy, 
        report['macro avg']['precision'], 
        report['macro avg']['recall'], 
        report['macro avg']['f1-score']
    ]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(metrics_names, latest_metrics, color=['#4C72B0', '#55A868', '#C44E52', '#8172B3'], edgecolor='black')
    plt.ylim(0, 1.1)
    plt.title('Overall Performance Metrics (Bar Graph)', fontsize=15, fontweight='bold', pad=15)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.3f}", ha='center', va='bottom', fontweight='bold')
    plt.savefig("5_metrics_bar_graph.png", bbox_inches='tight')
    print("-> Saved 5_metrics_bar_graph.png")
except Exception as e:
    print(f"Failed to plot Bar Graph: {e}")

# 6. SPYDER (RADAR) PLOT
try:
    from math import pi
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    N = len(categories)
    values = latest_metrics.copy()
    values += values[:1]
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, size=12, fontweight='bold')
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=10)
    plt.ylim(0, 1.1)
    ax.plot(angles, values, linewidth=2, linestyle='solid', color='darkorange', label='Ensemble')
    ax.fill(angles, values, 'orange', alpha=0.25)
    plt.title('Ensemble Performance Spyder Plot', pad=30, fontsize=15, fontweight='bold')
    plt.savefig("6_spyder_plot.png", bbox_inches='tight')
    print("-> Saved 6_spyder_plot.png")
except Exception as e:
    print(f"Failed to plot Spyder Plot: {e}")

# 7. PREDICTION CONFIDENCE DISTRIBUTION (HISTOGRAM)
try:
    confidences = np.max(ensemble_probabilities, axis=1)
    correct_guesses = (ensemble_final_labels == y_test)

    plt.figure(figsize=(10, 6))
    sns.histplot(confidences[correct_guesses], color='green', label='Correct Predictions', kde=True, bins=20, alpha=0.6)
    sns.histplot(confidences[~correct_guesses], color='red', label='Incorrect Predictions', kde=True, bins=20, alpha=0.6)
    plt.title('Ensemble Confidence Distribution', fontsize=15, fontweight='bold', pad=15)
    plt.xlabel('Prediction Confidence Probability (0.33 to 1.0)', fontsize=12)
    plt.ylabel('Number of Brain Scans', fontsize=12)
    plt.legend()
    plt.savefig("7_confidence_distribution.png", bbox_inches='tight')
    print("-> Saved 7_confidence_distribution.png")
except Exception as e:
    print(f"Failed to plot Confidence Distribution: {e}")

# 8. ENSEMBLE AGREEMENT MATRIX (How often did the 3 models completely agree?)
try:
    model_labels = np.argmax(all_test_predictions, axis=2) 
    std_dev = np.std(model_labels, axis=0)
    
    agreement_text = ["Complete Agreement (100%)" if val == 0 else "Disagreed (Slightly)" for val in std_dev]
    
    plt.figure(figsize=(6, 4))
    sns.countplot(x=agreement_text, palette="pastel", order=["Disagreed (Slightly)", "Complete Agreement (100%)"])
    plt.title("Did all 3 Master Models Agree?", fontweight='bold', fontsize=14)
    plt.ylabel("Number of Scans")
    plt.savefig("8_ensemble_agreement.png", bbox_inches='tight')
    print("-> Saved 8_ensemble_agreement.png")
except Exception as e:
    print(f"Failed to plot Ensemble Agreement: {e}")

# 9. TRUE VS PREDICTED ACTUAL IMAGE GRID (9 Random Test Scans)
try:
    plt.figure(figsize=(10, 10))
    indices = np.random.choice(range(len(X_test)), 9, replace=False)
    for i, idx in enumerate(indices):
        plt.subplot(3, 3, i + 1)
        img_display = np.clip(X_test[idx], 0, 1) # Prevent Matplotlib visual bugs
        plt.imshow(img_display)
        
        true_c = int(y_test[idx])
        pred_c = int(ensemble_final_labels[idx])
        color = 'green' if true_c == pred_c else 'red'
        plt.title(f"True: {true_c} | Pred: {pred_c}", color=color, fontweight='bold', fontsize=12)
        plt.axis('off')
    plt.suptitle('Ensemble Random Test Set Predictions', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("9_image_grid_predictions.png", bbox_inches='tight')
    print("-> Saved 9_image_grid_predictions.png")
except Exception as e:
    print(f"Failed to plot Image Grid: {e}")

# 10. TRAINING HISTORY LEARNING CURVES (Epoch by Epoch tracking)
try:
    plt.figure(figsize=(14, 5))
    
    # We will use the history of the 1st Model as our visual representative
    history_dict = all_histories[0] 
    
    # 10a. Accuracy Subplot
    plt.subplot(1, 2, 1)
    if 'sparse_categorical_accuracy' in history_dict:
        plt.plot(history_dict['sparse_categorical_accuracy'], label='Train Accuracy', marker='o', color='#1f77b4', linewidth=2)
        plt.plot(history_dict['val_sparse_categorical_accuracy'], label='Validation Accuracy', marker='^', color='#ff7f0e', linewidth=2)
    plt.title('Representative Learning Curve (Accuracy)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # 10b. Loss Subplot
    plt.subplot(1, 2, 2)
    if 'loss' in history_dict:
        plt.plot(history_dict['loss'], label='Train Loss', marker='o', color='#d62728', linewidth=2)
        plt.plot(history_dict['val_loss'], label='Validation Loss', marker='^', color='#2ca02c', linewidth=2)
    plt.title('Representative Learning Curve (Loss Penalty)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss/Error Penalty')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("10_learning_curves.png", bbox_inches='tight')
    print("-> Saved 10_learning_curves.png")
except Exception as e:
    print(f"Failed to plot Learning Curves: {e}")

print("\nALL 10 VISUALIZATION GRAPHS HAVE BEEN COMPILED AND SAVED!")
