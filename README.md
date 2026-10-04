# CODSOFT_TASK3 — Customer Churn Prediction

A machine-learning project that predicts whether a subscription customer will
**churn**, using historical customer data (demographics + usage behaviour).
Built for the **CodSoft Machine Learning Internship (Task 3)**.

## Overview

A `ColumnTransformer` imputes and scales numeric features and one-hot encodes
categorical ones, then three classifiers are compared:

| Model | Notes |
|---|---|
| Logistic Regression | linear baseline, `class_weight="balanced"` |
| Random Forest | 300 trees, handles non-linearity |
| Gradient Boosting | usually the strongest on tabular data |

The best model is chosen by **ROC-AUC** (the right metric for imbalanced churn),
saved, and reused to score new customers.

## Project structure

```
CODSOFT_TASK3/
├── train.py           # train + evaluate + save the best model
├── predict.py         # score new customers
├── requirements.txt
├── sample_data.csv    # small synthetic demo dataset
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
python train.py --data sample_data.csv
python train.py --data Telco-Customer-Churn.csv      # real Kaggle/IBM dataset
```

Artefacts land in `models/`:

* `best_model.joblib` — preprocessing + classifier
* `metrics.json` — accuracy / precision / recall / F1 / ROC-AUC per model
* `model_comparison.png`, `confusion_matrix.png`, `roc_curves.png`,
  `feature_importance.png`

## Predict

```bash
python predict.py --file customers.csv
python predict.py --json '{"gender":"Female","tenure":2,"MonthlyCharges":95.5,
  "Contract":"Month-to-month","InternetService":"Fiber optic", ...}'
```

## Results

On the **Telco Customer Churn** dataset (7,043 customers, 26.5% churn),
80/20 stratified split:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.738 | 0.504 | 0.783 | 0.614 | 0.841 |
| Random Forest | 0.766 | 0.552 | 0.623 | 0.585 | 0.822 |
| **Gradient Boosting** (best) | **0.806** | **0.674** | 0.524 | 0.590 | **0.843** |

Churn is imbalanced, so ROC-AUC and recall matter more than raw accuracy.
Logistic Regression has the highest recall (catches the most churners) while
Gradient Boosting has the best overall separation. Try `GridSearchCV`, SMOTE,
or XGBoost/LightGBM for further gains.

## Dataset

* **Kaggle:** "Telco Customer Churn" (also mirrored by IBM) — file
  `WA_Fn-UseC_-Telco-Customer-Churn.csv`.
* Any CSV with a binary churn column also works; the target column is
  auto-detected (or pass `--target NAME`).

## Notes

* `TotalCharges` sometimes arrives as text with blanks — the loader coerces it
  to numeric and imputes missing values with the median.
* `customerID` (and similar id columns) are dropped automatically.
