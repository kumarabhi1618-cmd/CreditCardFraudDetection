# Credit Card Fraud Detection — UpdatedNotebook

## Overview

This notebook builds a fraud detection pipeline on the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). It follows a deliberate two-phase evaluation journey: starting with the conventional metrics most notebooks use, then systematically uncovering why those metrics are misleading for heavily imbalanced fraud data, and replacing them with metrics that reflect real-world cost.

The dataset contains **284,807 transactions** of which only **492 (0.17%) are fraud** — a 577:1 class imbalance that exposes every weakness in standard evaluation practice.

---

## The Journey: From Naive to Fraud-Aware Evaluation

### Phase 1 — Standard Approach (and why it fails)

The notebook opens by doing exactly what most fraud detection notebooks do: train several classifiers, evaluate with **Accuracy** and **ROC-AUC**, rank models by ROC-AUC, pick the best one.

This immediately surfaces a problem. Every model achieves **99.9%+ accuracy** — because predicting "normal" for every single transaction gives 99.83% accuracy by default. Accuracy is completely uninformative here.

ROC-AUC looks more promising, with models scoring 0.87–0.99. But ROC-AUC measures how well a model separates the two classes *across all thresholds*, weighting both classes equally. With 99.83% normal transactions, being good at classifying the easy majority class inflates ROC-AUC even when the model misses most fraud. A model that misses 40 out of 98 frauds (FN=40) can still report ROC-AUC=0.97.

The confusion matrix makes the gap visible: high ROC-AUC models are often letting a significant number of frauds slip through as False Negatives — the worst possible outcome in fraud detection.

### Phase 2 — Fraud-Aware Evaluation

Recognising that ROC-AUC is insufficient, the notebook introduces metrics calibrated to the asymmetric cost of fraud:

**PR-AUC (Precision-Recall AUC)** becomes the primary metric. Unlike ROC-AUC, it focuses entirely on the minority class. The random baseline for PR-AUC on this dataset is 0.0017 (the fraud prevalence) — so even a PR-AUC of 0.70 represents a massive real improvement. Crucially, PR-AUC is *not* inflated by correct majority-class predictions.

**F2 Score** generalises F1 by weighting Recall twice as heavily as Precision — directly encoding the business reality that missing a fraud costs more than a false alarm. The notebook sweeps 300 thresholds to find the F2-optimal decision boundary rather than defaulting to 0.5.

**FN count** (False Negatives — missed frauds) becomes an explicit tracked metric alongside FP count. Every model now reports both at the default threshold and at the F2-optimal threshold, making the tradeoff between catching fraud and false alarms transparent.

All three metrics are computed inside a single `evaluate_model()` function called by every model under every methodology — so model selection and evaluation are always on the same basis.

---

## Notebook Structure

```
creditcard.csv
    │
    ▼
1.  Imports
2.  Exploratory Data Analysis
        • Class distribution (0.17% fraud)
        • Correlation matrix (feature-to-feature multicollinearity only)
3.  Feature Engineering
        • Time → Hours of day
        • Amount → StandardScaler  ← critical fix; unscaled Amount
                                      breaks KNN and Logistic Regression
4.  Train / Test Split
        • stratify=Y preserves fraud ratio in both splits
        • x_test held out for final evaluation only
5.  Variable Distribution Plots
        • KDE plots: fraud vs normal distribution per feature
6.  Feature Relevance Analysis  (imbalance-aware)
        • Mann-Whitney U test  — non-parametric class separation
        • Mutual Information   — non-linear dependence
        • Combined ranking     — consensus feature importance
        • Correlation heatmap  — multicollinearity only (not target relevance)
7.  Model Building
        • Unified evaluate_model() — computes all metrics in one call
        • 6 model families: LR (L1+L2), KNN, Decision Tree, Random Forest,
          XGBoost, SVM
8.  Six Methodologies (cross-validation + class balancing)
        • Baseline            — full x_train, no resampling
        • RepeatedKFold CV
        • StratifiedKFold CV
        • StratifiedKFold + RandomOverSampler
        • StratifiedKFold + SMOTE
        • StratifiedKFold + ADASYN
9.  Results Summary
        • Full table sorted by PR-AUC
        • Best model per metric (ROC-AUC, PR-AUC, F2, FN)
10. Results Analysis (4 plots)
        • PR-AUC grouped bar: every model × methodology
        • ROC-AUC vs PR-AUC scatter: where they disagree
        • FN vs FP at F2-threshold: top 12 combinations
        • Mean PR-AUC and mean FN by methodology
11. Hyperparameter Tuning — XGBoost
        • Model chosen: XGBoost (RepeatedKFold CV)
        • Nearly equal PR-AUC to Random Forest; 7× faster to train
        • Scoring: average_precision (PR-AUC) throughout
        • Fair comparison: untuned and tuned both evaluated on same x_test
12. Save results and model
```

---

## Key Fixes Over the Original Notebook

| # | Problem | Fix |
|---|---|---|
| 1 | CV indexing used `X.iloc[train_idx]` on full dataset with positional indices from `x_train` split | Changed to `x_train.iloc[train_idx]` throughout |
| 2 | Oversampling applied before train/test isolation — data leakage | Oversampling applied only inside training fold; test fold never touched |
| 3 | `Amount` not scaled — scale mismatch with PCA features | `StandardScaler` applied to `Amount` in feature engineering |
| 4 | No `stratify=Y` in train/test split | Added `stratify=Y` |
| 5 | `range(60, 130, 150)` for n_estimators → only `[60]` | Replaced with explicit list `[60, 90, 120]` |
| 6 | Hyperparameter search fitted on one CV fold, not full x_train | Changed to `search.fit(x_train, y_train)` |
| 7 | XGBoost missing `scale_pos_weight` for class imbalance | Added `scale_pos_weight = neg/pos` |
| 8 | `tol=10` in Logistic Regression — solver stops after 1–2 iterations | Changed to `tol=1e-4` |
| 9 | Decision Tree: `scores` dict overwritten with scalar each loop | Removed stale variable; results logged directly |
| 10 | SVM fitted twice with inconsistent settings | Single fit with `probability=True` |
| 11 | `StratifiedKFold(random_state=42)` without `shuffle=True` — no effect | Added `shuffle=True` |
| 12 | `sns.distplot` removed in seaborn ≥ 0.12 | Replaced with `sns.kdeplot(..., fill=True)` |
| 13 | `df.corr()` raises TypeError in pandas ≥ 2.0 | Added `numeric_only=True` |
| 14 | Google Colab `drive.mount()` hardcoded | Replaced with local `pd.read_csv()` |

---

## What the Results Showed

After running 42 combinations (6 methodologies × 7 models):

**ROC-AUC and PR-AUC frequently disagreed.** The worst case: Decision Trees with SMOTE/ADASYN showed ROC-AUC ~0.88 but PR-AUC of only 0.28–0.37. A gap of 0.60 — ROC-AUC was reporting a model as "decent" that was nearly useless for catching fraud.

**Oversampling hurt Logistic Regression severely.** LR without oversampling: 9 false alarms. With SMOTE: 1,116 false alarms. With ADASYN: 3,634 false alarms. Oversampling pushed LR's decision boundary so far it over-flagged massively — a 400× increase in false alarms.

**Random Forest and XGBoost were robust.** Both models maintained PR-AUC ~0.83 regardless of resampling strategy. Their internal class-weighting mechanisms already handle the imbalance.

**Best overall: RepeatedKFold + Random Forest** (PR-AUC 0.8569, FN=8 at F2-threshold).
**Best speed/performance: RepeatedKFold + XGBoost** (PR-AUC 0.8514, trains in 8s vs 62s).

XGBoost was selected for hyperparameter tuning.

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn
scipy
joblib
```

Install with:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn scipy joblib
```

---

## How to Run

1. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the same directory as the notebook
2. Update the path in Cell 3 if needed: `df = pd.read_csv("creditcard.csv")`
3. Run all cells top to bottom — cells are ordered so every dependency is available before it is used
4. After all 6 methodologies complete, run the Results Analysis cells (plots) then proceed to tuning

---

## Output Files

| File | Contents |
|---|---|
| `./results/model_evaluation_results.csv` | Full 42-row results table with all metrics |
| `./saved_models/xgb_tuned_final.pkl` | Best tuned XGBoost model (joblib) |

---

## Metric Reference

| Metric | Used for | Note |
|---|---|---|
| Accuracy | Baseline sanity check only | Meaningless on imbalanced data |
| ROC-AUC | Shown for comparison | Inflated by majority class — do not use for model selection |
| **PR-AUC** | **Primary model selection metric** | Focuses on minority class; honest on imbalanced data |
| **F2 Score** | **Secondary — recall-weighted** | Penalises missed frauds 2× more than false alarms |
| **FN count** | **Operational — missed frauds** | The most direct business cost metric |
| FP count | Operational — false alarms | Secondary operational concern |
