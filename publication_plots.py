
import os, pickle, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize
from tensorflow import keras

# --- 1. CONFIGURATION ---
MODEL_DIR = "final_research_models_78725" # Update this to your latest folder
CLASS_NAMES = ['MCI', 'CN', 'AD']
LABELS = [0, 1, 2]

# Load Data (assuming same data used for training)
print("Loading Validation Datasets...")
X_img_te = pd.read_pickle("img_test.pkl")["img_array"]
X_img_te = np.array([x for x in X_img_te.values], dtype=np.float32)
y_img_te = pd.read_pickle("img_y_test.pkl")["label"].values.astype(int)

with open('X_test_full.pkl', 'rb') as f: X_cl_te = pickle.load(f).astype(np.float32)
with open('y_test_full.pkl', 'rb') as f: y_cl_te = pickle.load(f).astype(int)

# --- 2. ENSEMBLE INFERENCE ---
def gather_predictions(modality, X, count=10):
    print(f"Aggregating {count} {modality} experts...")
    all_probs = []
    for i in range(1, count + 1):
        path = f"{MODEL_DIR}/{modality}_{i}.h5"
        if os.path.exists(path):
            m = keras.models.load_model(path)
            all_probs.append(m.predict(X, verbose=0))
            keras.backend.clear_session()
        else:
            print(f"Skipping: {path} not found.")
    return np.mean(all_probs, axis=0)

print("Inferring Imaging probabilities...")
i_probs = gather_predictions("img", X_img_te)
print("Inferring Clinical probabilities...")
c_probs = gather_predictions("clin", NUM_MODELS=10, X=X_cl_te) # Helper fix: 10

# --- 3. THE PLOTS ---
sns.set_theme(style="whitegrid", context="talk")

# Plot 1: Imaging Confusion Matrix (Normalized)
def plot_cm(y_true, probs, title, filename, cmap='Blues'):
    y_pred = np.argmax(probs, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap=cmap,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"Confusion Matrix: {title}", fontsize=18, fontweight='bold')
    plt.ylabel('True Category', fontweight='bold')
    plt.xlabel('Predicted Category', fontweight='bold')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Saved {filename}")

plot_cm(y_img_te, i_probs, "Imaging SE-Ensemble", "Fig1_Imaging_CM.png", 'Blues')

# Plot 2: Multimodal ROC curves
def plot_roc(y_true, probs, title, filename):
    y_bin = label_binarize(y_true, classes=LABELS)
    plt.figure(figsize=(10, 8))
    colors = ['#4C72B0', '#55A868', '#C44E52']
    
    for i in range(len(CLASS_NAMES)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=3,
                 label=f'{CLASS_NAMES[i]} (AUC = {roc_auc:.3f})')
        
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title(f"ROC Curves: {title}", fontsize=18, fontweight='bold')
    plt.legend(loc="lower right", fontsize=14)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Saved {filename}")

plot_roc(y_img_te, i_probs, "Imaging Expert Ensemble", "Fig2_Imaging_ROC.png")

# Plot 3: Performance Bar Chart (Horizontal)
def plot_metrics_bars(y_img, i_probs, y_cl, c_probs):
    report_i = classification_report(y_img, np.argmax(i_probs, axis=1), output_dict=True)
    report_c = classification_report(y_cl, np.argmax(c_probs, axis=1), output_dict=True)
    
    data = {
        'Modality': ['Imaging', 'Imaging', 'Clinical', 'Clinical'],
        'Metric': ['Macro Accuracy', 'F1-Score', 'Macro Accuracy', 'F1-Score'],
        'Score': [report_i['accuracy'], report_i['macro avg']['f1-score'],
                  report_c['accuracy'], report_c['macro avg']['f1-score']]
    }
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='Score', y='Metric', hue='Modality', data=df, palette='viridis')
    plt.xlim(0, 1.05)
    plt.title("Performance Benchmarking Summary", fontsize=18, fontweight='bold')
    for p in ax.patches:
        ax.annotate(f"{p.get_width():.4f}", (p.get_width() + 0.01, p.get_y() + p.get_height()/2), 
                    ha='left', va='center', fontweight='bold')
    plt.savefig("Fig3_Performance_Benchmarking.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(" -> Saved Fig3_Performance_Benchmarking.png")

plot_metrics_bars(y_img_te, i_probs, y_cl_te, c_probs)

print("\n🎉 ALL PUBLICATION FIGURES GENERATED SUCCESSFULLY! 🎉")
print("- Fig1_Imaging_CM.png: Imaging Confusion Matrix")
print("- Fig2_Imaging_ROC.png: Imaging ROC Curves")
print("- Fig3_Performance_Benchmarking.png: Modality Comparison")
