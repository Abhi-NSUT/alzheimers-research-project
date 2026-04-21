
import pandas as pd
import numpy as np
import pickle, time
from sklearn.model_selection import train_test_split

ACTIVE_SEED = int(time.time() * 1000) % 100000 
print(f"--- WIDER MERGE DATA PREPARATION (SEED: {ACTIVE_SEED}) ---")

# 1. Load the 4 Key ADNI CSVs
demo  = pd.read_csv("csv/PTDEMOG.csv")
neuro = pd.read_csv("csv/NEUROEXM.csv")
cogn  = pd.read_csv("csv/ADSP_PHC_COGN.csv").rename(columns={"PHASE":"Phase"})
dx    = pd.read_csv("csv/DXSUM_PDXCONV_ADNIALL.csv")[["RID", "Phase"]]
meta  = pd.read_csv("csv/metadataimage.csv")

# 2. Wider Merge Strategy (Relaxing 'Phase' constraint to find all data)
print("Standardizing IDs and performing Wide Merge (RID-based)...")

# Get most recent clinical data per RID
neuro_clean = neuro.sort_values(['RID']).drop_duplicates('RID', keep='last')
demo_clean  = demo.sort_values(['RID']).drop_duplicates('RID', keep='last')
cogn_clean  = cogn.sort_values(['RID']).drop_duplicates('RID', keep='last')

m = (demo_clean.merge(neuro_clean, on="RID", how="outer")
               .merge(cogn_clean, on="RID", how="outer"))

# 3. Join with ground truth labels
map3 = {'MCI': 0, 'CN': 1, 'AD': 2}
gt = meta.drop_duplicates('Subject')[['Subject', 'Group']]
gt['label'] = gt['Group'].map(map3)

# PTID <-> RID mapping
id_map = demo[['PTID', 'RID']].drop_duplicates('PTID')
gt = gt.merge(id_map, left_on='Subject', right_on='PTID', how='inner')

final_m = m.merge(gt[['RID', 'label']], on='RID', how='inner')
print(f"Successfully recovered {len(final_m)} unique patients with labels!")

# 4. Filter Features
official_29 = [
    'PTGENDER', 'PTDOBYY', 'PTHAND', 'PTMARRY', 'PTEDUCAT', 'PTWORK', 
    'PTNOTRT', 'PTHOME', 'PTTLANG', 'PTPLANG', 'PTCOGBEG', 'PTETHCAT', 
    'PTRACCAT', 'NXVISUAL', 'NXAUDITO', 'NXTREMOR', 'NXCONSCI', 'NXNERVE', 
    'NXMOTOR', 'NXFINGER', 'NXHEEL', 'NXSENSOR', 'NXTENDON', 'NXPLANTA', 
    'NXGAIT', 'NXABNORM', 'PHC_MEM', 'PHC_EXF', 'PHC_LAN', 'PHC_VSP'
]
t = final_m[['label'] + [c for c in official_29 if c in final_m.columns]].fillna(-4)

# One-Hot Encoding
categorical = ['PTGENDER', 'PTWORK', 'PTMARRY', 'PTEDUCAT', 'PTETHCAT', 
               'PTRACCAT', 'PTHAND', 'NXPLANTA', 'NXGAIT', 'NXABNORM']

encoded_list = [t[['label']]]
for col in official_29:
    if col in categorical and col in t.columns:
        encoded_list.append(pd.get_dummies(t[col], prefix=col))
    elif col in t.columns:
        encoded_list.append(t[[col]])

final_matrix = pd.concat(encoded_list, axis=1)
X = final_matrix.drop('label', axis=1).values.astype(np.float32)
y = final_matrix['label'].values.astype(int)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1, random_state=42)

with open('X_train_full.pkl', 'wb') as f: pickle.dump(X_tr, f)
with open('X_test_full.pkl', 'wb') as f:  pickle.dump(X_te, f)
with open('y_train_full.pkl', 'wb') as f: pickle.dump(y_tr, f)
with open('y_test_full.pkl', 'wb') as f:  pickle.dump(y_te, f)

print(f"✅ FINAL NN-Ready File built for {len(y_tr)} patients.")
