"""
Paper evaluation: test-set AUROC, AUPRC, sensitivity, specificity, PPV, NPV and F1 with 95% bootstrap CIs
(1,000 resamples) for LR, RF, XGBoost and the prior-admissions rule, plus Brier score before/after
isotonic calibration. Thresholds are chosen by maximum F1 on out-of-fold TRAINING predictions only.
Run from the project root (about 5 minutes):   python evaluate_ci.py
Outputs: tables/table_metrics_ci.csv, tables/metrics_with_ci.json
"""
import sys, json; from pathlib import Path; ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT / 'src'))
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, brier_score_loss
from xgboost import XGBClassifier
from pipeline import *
BEST_XGB = dict(max_depth=5, learning_rate=0.03, n_estimators=300, min_child_weight=20, subsample=0.8, colsample_bytree=0.9)
raw = load_raw(ROOT / 'data' / 'diabetic_data.csv'); df, tops = clean(raw)
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE).split(df, df['readmit_30'], df['patient_nbr']))
train, test = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
Xtr, ytr, gtr = train[FEATURES], train['readmit_30'].values, train['patient_nbr']
Xte, yte = test[FEATURES], test['readmit_30'].values
spw = (ytr==0).sum()/(ytr==1).sum()
def models():
    return {'Logistic Regression': Pipeline([('prep', make_preprocessor(scale=True)),('clf', LogisticRegression(C=0.1, max_iter=3000, class_weight='balanced'))]),
     'Random Forest': Pipeline([('prep', make_preprocessor()),('clf', RandomForestClassifier(n_estimators=300, min_samples_leaf=20, max_features='sqrt', class_weight='balanced_subsample', n_jobs=-1, random_state=RANDOM_STATE))]),
     'XGBoost': Pipeline([('prep', make_preprocessor()),('clf', XGBClassifier(**BEST_XGB, scale_pos_weight=spw, eval_metric='logloss', tree_method='hist', n_jobs=-1, random_state=RANDOM_STATE))])}
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
oof = {k: np.zeros(len(train)) for k in models()}; pte = {}
for a,b in cv.split(Xtr, ytr, gtr):
    for k,m in models().items(): oof[k][b] = m.fit(Xtr.iloc[a], ytr[a]).predict_proba(Xtr.iloc[b])[:,1]
for k,m in models().items(): pte[k] = m.fit(Xtr, ytr).predict_proba(Xte)[:,1]
heur = (Xte['number_inpatient']*1000 + Xte['number_emergency']*10 + Xte['number_outpatient']).values.astype(float)
heur_tr = (Xtr['number_inpatient']*1000 + Xtr['number_emergency']*10 + Xtr['number_outpatient']).values.astype(float)
def best_thr(y,p):
    pr,rc,t = precision_recall_curve(y,p); f = 2*pr*rc/(pr+rc+1e-12); return t[np.argmax(f[:-1])]
def cls(y,p,t):
    yh = p>=t; tp=(yh&(y==1)).sum(); fp=(yh&(y==0)).sum(); fn=(~yh&(y==1)).sum(); tn=(~yh&(y==0)).sum()
    se=tp/(tp+fn); sp=tn/(tn+fp); ppv=tp/max(tp+fp,1); f1=2*ppv*se/(ppv+se) if ppv+se>0 else 0
    return dict(sens=se,spec=sp,ppv=ppv,npv=tn/max(tn+fn,1),f1=f1)
rng = np.random.default_rng(42); B=1000; n=len(yte); idx=[rng.integers(0,n,n) for _ in range(B)]
cal = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(oof['XGBoost'], ytr)
res={}
scores = dict(pte); scores['Prior-admissions rule'] = heur
oofs = dict(oof); oofs['Prior-admissions rule'] = heur_tr
for k,p in scores.items():
    t = best_thr(ytr, oofs[k])
    def allm(ii):
        y,pp = yte[ii],p[ii]; d = dict(auroc=roc_auc_score(y,pp), auprc=average_precision_score(y,pp)); d.update(cls(y,pp,t)); return d
    point = allm(np.arange(n)); boots = pd.DataFrame([allm(ii) for ii in idx])
    res[k] = {m:[point[m], boots[m].quantile(.025), boots[m].quantile(.975)] for m in point}; res[k]['thr']=float(t); res[k]['flag']=float((p>=t).mean())
pc = cal.predict(pte['XGBoost'])
bs = [brier_score_loss(yte[ii], pc[ii]) for ii in idx]; bsr=[brier_score_loss(yte[ii], pte['XGBoost'][ii]) for ii in idx]
res['brier_cal'] = [brier_score_loss(yte,pc), np.quantile(bs,.025), np.quantile(bs,.975)]
res['brier_raw'] = [brier_score_loss(yte,pte['XGBoost']), np.quantile(bsr,.025), np.quantile(bsr,.975)]
json.dump(res, open(ROOT / 'tables' / 'metrics_with_ci.json', 'w'), indent=1, default=float)
np.savez(ROOT / 'tables' / 'test_predictions.npz', y=yte, **{k.replace(' ','_'):v for k,v in pte.items()}, heur=heur, cal=pc)
rows = []
for k, v in res.items():
    if isinstance(v, dict):
        rows.append({'Model': k, 'Threshold': round(v['thr'], 3), **{m.upper(): f"{v[m][0]:.3f} ({v[m][1]:.3f}-{v[m][2]:.3f})" for m in ['auroc', 'auprc', 'sens', 'spec', 'ppv', 'npv', 'f1']}})
pd.DataFrame(rows).to_csv(ROOT / 'tables' / 'table_metrics_ci.csv', index=False)
print(pd.DataFrame(rows).to_string(index=False))
print('Brier calibrated %.4f (%.4f-%.4f); raw %.4f' % (*res['brier_cal'], res['brier_raw'][0]))
