# Predicting and Prioritising 30-Day Readmission Follow-Up for Diabetic Inpatients

Calibrated gradient boosting on the Diabetes 130-US Hospitals EHR dataset, with a decision layer that decides
**who should get limited post-discharge follow-up**.

BCSE335L Healthcare Data Analytics · VIT Chennai
Dwetta Pati (23BDS1072) · Sia Meel (23BDS1058) · Aaditya Arora (23BCE1867) · Faculty: Dr. Maheswari S

> Academic project. Not a medical device and not for clinical use.

## What it does

1. Cleans the data with clinical exclusions (deaths and hospice discharges), groups ICD-9 codes and administrative codes.
2. Splits **by patient** (no patient appears in both training and test data); all preprocessing is fitted on training folds only.
3. Trains logistic regression (baseline), random forest and XGBoost, and compares them with a prior-admissions rule.
4. Calibrates XGBoost with isotonic regression (fitted on out-of-fold predictions) and explains it with SHAP.
5. Ranks patients and measures how many true readmissions a fixed follow-up capacity captures, against random selection and the rule, plus a cost–benefit analysis.
6. Streamlit app: ranked discharge list, model vs random vs rule, and a SHAP explanation for each patient.

## Results (held-out test set: 19,773 encounters, 13,998 patients, 2,244 readmissions)

| Metric (95% CI) | Logistic Regression | Random Forest | **XGBoost** | Prior-admissions rule |
|---|---|---|---|---|
| AUROC | 0.665 (0.653–0.676) | 0.675 (0.664–0.687) | **0.678 (0.667–0.690)** | 0.616 (0.604–0.627) |
| AUPRC | 0.213 (0.199–0.229) | 0.226 (0.211–0.242) | **0.235 (0.220–0.252)** | 0.197 (0.183–0.211) |
| Sensitivity | 0.479 (0.459–0.500) | 0.481 (0.461–0.502) | 0.525 (0.506–0.546) | 0.499 (0.479–0.518) |
| Specificity | 0.733 (0.726–0.739) | 0.758 (0.751–0.764) | 0.722 (0.715–0.728) | 0.692 (0.685–0.699) |
| F1 | 0.269 (0.256–0.281) | 0.285 (0.272–0.298) | 0.284 (0.271–0.297) | 0.255 (0.244–0.267) |

- Calibration: Brier score 0.217 (raw) → 0.095 (isotonic).
- Five extra patient-level splits: AUROC 0.667 ± 0.008.
- At 10% follow-up capacity, XGBoost captures **546** readmissions vs **224** for random selection (2.44×) and **496** for the rule.
- Weaker discrimination for patients aged 80+ (AUROC 0.595).

## Repository structure

```
data/README.md                          How to download the dataset (data are NOT included)
src/pipeline.py                         Cleaning, feature engineering, preprocessing, prioritisation helpers
Readmission_Prioritization_Final.ipynb  Full analysis (EDA, models, calibration, SHAP, prioritisation, cost, subgroups, stability)
build_notebook.py                       Script that generates the notebook
train_model.py                          Re-trains and saves the final model (~1 min)
evaluate_ci.py                          All test metrics with 95% bootstrap CIs (paper Table III)
app/app.py                              Streamlit decision-support demo
app/make_demo_sample.py                 Builds the demo input from unseen test-set patients
models/readmission_model.joblib         Final XGBoost pipeline + isotonic calibrator + threshold
figures/, tables/                       All figures and result tables
paper/                                  Paper PDF
docs/                                   TRIPOD+AI reporting checklist
```

## How to reproduce

```bash
git clone https://github.com/<username>/readmission-prioritisation.git
cd readmission-prioritisation
pip install -r requirements.txt
# put diabetic_data.csv in data/ (see data/README.md)

jupyter nbconvert --to notebook --execute --inplace Readmission_Prioritization_Final.ipynb   # full analysis, ~8 min
python evaluate_ci.py        # metrics with 95% CIs, ~5 min
python train_model.py        # only if the saved model will not load on your machine
```

## Run the app

```bash
python app/make_demo_sample.py   # once, after downloading the data
streamlit run app/app.py
```

Use the slider to set follow-up capacity. You can also upload any CSV in the original dataset format.

## Leakage controls

- Patient-level `GroupShuffleSplit` (80/20) and `StratifiedGroupKFold` for all cross-validation.
- Imputation, scaling and encoding inside a scikit-learn `Pipeline`, fitted on training folds only.
- Class weighting instead of oversampling; no synthetic patients.
- Thresholds and the calibrator are fitted on out-of-fold training predictions; the test set is used once.

## Data, ethics and licence

- Data: Diabetes 130-US Hospitals for Years 1999–2008, UCI Machine Learning Repository, doi:10.24432/C5230J, CC BY 4.0.
  Strack et al., *BioMed Research International*, 2014. De-identified, public; no patient-level data are shared in this repository.
- Any use on Indian hospital data would need institutional ethics approval and must follow the Digital Personal Data
  Protection Act, 2023 and the ICMR Ethical Guidelines for Application of AI in Biomedical Research and Healthcare (2023).
- Reported following TRIPOD+AI (Collins et al., BMJ 2024).
- AI tool use: generative AI (Claude, Anthropic) assisted with code, drafting and formatting; the authors reviewed and verified all outputs.
- Code licence: MIT (see `LICENSE`).
