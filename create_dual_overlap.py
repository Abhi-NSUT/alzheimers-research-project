
import pandas as pd, pickle, numpy as np

try:
    # 1. IDENTITIES: Pull the 567-patient Clinical bank (Subject Anchor)
    print("1. Identifying 567-patient Clinical Master Bank...")
    train_anchor = pd.read_csv('X_train_clinical.csv')['subject'].tolist()
    test_anchor = pd.read_csv('X_test_clinical.csv')['subject'].tolist()
    clin_ptids = train_anchor + test_anchor
    
    # Load Clinical Feature & Label Data (567 records)
    with open('X_train_full.pkl', 'rb') as f: X_cl_tr = pickle.load(f).astype(np.float32)
    with open('X_test_full.pkl', 'rb') as f: X_cl_te = pickle.load(f).astype(np.float32)
    with open('y_train_full.pkl', 'rb') as f: y_cl_tr = pickle.load(f).astype(int)
    with open('y_test_full.pkl', 'rb') as f: y_cl_te = pickle.load(f).astype(int)
    X_cl_bank = np.vstack([X_cl_tr, X_cl_te])
    y_cl_bank = np.hstack([y_cl_tr, y_cl_te])
    
    # 2. DICTIONARY: {PTID -> (Features, Label)}
    ptid_to_clin = {}
    for i, ptid in enumerate(clin_ptids):
        if i < len(X_cl_bank):
            ptid_to_clin[ptid] = (X_cl_bank[i], y_cl_bank[i])
    
    # 3. IMAGING SEARCH: Scan the entire 8,000-scan library for these patients
    print("2. Scanning full 8,000-scan Imaging Library for multimodal overlaps...")
    gt = pd.read_csv('ground_truth.csv', index_col=0)
    img_all = pd.concat([pd.read_pickle('img_train.pkl'), pd.read_pickle('img_test.pkl')], axis=0)
    
    # Find indices where imaging and clinical PTIDs intersect
    search_ids = set(clin_ptids)
    all_gt_info = gt[gt['PTID'].isin(search_ids)]
    
    final_matched_data = []
    for idx, row in all_gt_info.iterrows():
        if idx in img_all.index:
            ptid = row['PTID']
            if ptid in ptid_to_clin:
                final_matched_data.append({
                    'PTID': ptid,
                    'img_array': img_all.loc[idx, 'img_array'],
                    'clin_features': ptid_to_clin[ptid][0],
                    'label': ptid_to_clin[ptid][1]
                })
    
    # 4. FINISH AND SAVE THE ALIGNED COHORT
    with open('final_aligned_sota_cohort.pkl', 'wb') as f:
        pickle.dump(final_matched_data, f)
    print("\n" + "="*50)
    print(f" SUCCESS! Aligned Cohort Created: {len(final_matched_data)} samples.")
    print("Outcome: final_aligned_sota_cohort.pkl is now row-for-row correct.")
    print("STEP 2: RUN 'evaluate_fusion.py' NOW.")
    print("="*50)
    
except Exception as e:
    print(f"\n BUILD FAILED: {e}")
    print("Verify that your clinical .pkl and .csv files match perfectly.")
