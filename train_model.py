"""
Re-trains the final model on THIS computer and saves models/readmission_model.joblib.
Use it if the app says the saved model can't be loaded (library version mismatch).
Run from the project folder:   python train_model.py      (takes about 1 minute)
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import joblib
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from pipeline import *

BEST_XGB = dict(max_depth=5, learning_rate=0.03, n_estimators=300, min_child_weight=20, subsample=0.8, colsample_bytree=0.9)

print('Loading and cleaning data...')
raw = load_raw(ROOT / 'data' / 'diabetic_data.csv')
df, top_specialties = clean(raw)
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE).split(df, df['readmit_30'], df['patient_nbr']))
train, test = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
Xtr, ytr, gtr = train[FEATURES], train['readmit_30'], train['patient_nbr']
spw = (ytr == 0).sum() / (ytr == 1).sum()

def make_model():
    return Pipeline([('prep', make_preprocessor()),
                     ('clf', XGBClassifier(**BEST_XGB, scale_pos_weight=spw, eval_metric='logloss',
                                           tree_method='hist', n_jobs=-1, random_state=RANDOM_STATE))])

print('Out-of-fold predictions for calibration (5 folds)...')
oof = np.zeros(len(train))
for f_tr, f_va in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(Xtr, ytr, gtr):
    oof[f_va] = make_model().fit(Xtr.iloc[f_tr], ytr.iloc[f_tr]).predict_proba(Xtr.iloc[f_va])[:, 1]
calibrator = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(oof, ytr)

print('Training final model...')
model = make_model().fit(Xtr, ytr)
p = model.predict_proba(test[FEATURES])[:, 1]
auc, ap = roc_auc_score(test['readmit_30'], p), average_precision_score(test['readmit_30'], p)
print(f'Test ROC-AUC {auc:.3f} | PR-AUC {ap:.3f}')

joblib.dump({'model': model, 'calibrator': calibrator, 'threshold': 0.55, 'top_specialties': top_specialties,
             'features': FEATURES, 'metrics': {'test_roc_auc': float(auc), 'test_pr_auc': float(ap)}},
            ROOT / 'models' / 'readmission_model.joblib')
print('Saved models/readmission_model.joblib. Now run:  streamlit run app/app.py')
