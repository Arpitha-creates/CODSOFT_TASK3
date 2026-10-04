#!/usr/bin/env python3
"""
CODSOFT Machine Learning Internship
Task 3 -- Customer Churn Prediction
====================================

Predicts whether a subscription customer will churn, from historical customer
data (demographics + usage behaviour).

Models compared:
    * Logistic Regression
    * Random Forest
    * Gradient Boosting

Usage
-----
    python train.py --data sample_data.csv
    python train.py --data Telco-Customer-Churn.csv

Outputs (into --outdir, default ./models)
-----------------------------------------
    best_model.joblib       best pipeline (preprocessing + classifier)
    metrics.json            accuracy / precision / recall / F1 / ROC-AUC per model
    model_comparison.png    bar chart comparing the models
    confusion_matrix.png    confusion matrix of the best model
    roc_curves.png          ROC curves of all models
    feature_importance.png  top features driving churn (best tree/linear model)
"""

from __future__ import annotations

import argparse
import json
import os
import re

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _find_target(df: pd.DataFrame, target: str | None) -> str:
    if target:
        return target
    for cand in ("Churn", "churn", "Exited", "exited", "target", "Target"):
        if cand in df.columns:
            return cand
    # else: a binary column with two unique values
    for c in df.columns:
        if df[c].nunique() == 2:
            return c
    raise ValueError("could not find a target column; pass --target")


def load_dataset(path: str, target: str | None = None):
    df = pd.read_csv(path)
    df = df.dropna(how="all").reset_index(drop=True)

    # Drop obvious id columns that carry no signal.
    for idc in ("customerID", "CustomerID", "customer_id", "id", "ID"):
        if idc in df.columns:
            df = df.drop(columns=[idc])

    target = _find_target(df, target)
    y_raw = df[target].astype(str).str.strip().str.lower()
    pos = {"yes", "1", "true", "churn", "exited"}
    y = y_raw.isin(pos).astype(int)

    X = df.drop(columns=[target])

    # Coerce numeric-looking columns that were read as text (e.g. TotalCharges).
    for c in X.columns:
        if pd.api.types.is_string_dtype(X[c]) and not pd.api.types.is_numeric_dtype(X[c]):
            coerced = pd.to_numeric(X[c].str.strip(), errors="coerce")
            if coerced.notna().mean() > 0.9:      # mostly numeric -> treat as numeric
                X[c] = coerced

    print(f"[data] {len(X)} rows, {X.shape[1]} features, target='{target}' "
          f"(positive rate {y.mean():.1%})")
    return X, y, target


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def build_models(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline([
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                       random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                           n_jobs=-1, random_state=RANDOM_STATE)),
        ]),
        "Gradient Boosting": Pipeline([
            ("prep", preprocessor),
            ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]),
    }


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_comparison(results: dict, outpath: str) -> None:
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    names = list(results)
    x = np.arange(len(metrics))
    w = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, n in enumerate(names):
        vals = [results[n][m] for m in metrics]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=n)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Customer Churn - model comparison")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_confusion(cm, outpath: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No churn", "Churn"]); ax.set_yticklabels(["No churn", "Churn"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix - best model")
    thresh = cm.max() / 2 if cm.max() else 0.5
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_roc(curves: dict, outpath: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves"); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_feature_importance(model: Pipeline, top_n: int, outpath: str) -> None:
    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]
    try:
        names = list(prep.get_feature_names_out())
    except Exception:
        return
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        imp = np.abs(clf.coef_).ravel()
    else:
        return
    order = np.argsort(imp)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
    ax.barh([names[i] for i in order][::-1], [imp[i] for i in order][::-1],
            color="#4C72B0")
    ax.set_title(f"Top {top_n} features driving churn")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Customer churn trainer")
    ap.add_argument("--data", default="sample_data.csv")
    ap.add_argument("--target", default=None, help="name of the target column")
    ap.add_argument("--outdir", default="models")
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    X, y, target = load_dataset(args.data, args.target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y)

    preprocessor = build_preprocessor(X)
    models = build_models(preprocessor)

    results, curves, fitted = {}, {}, {}
    for name, pipe in models.items():
        print(f"[train] {name} ...")
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        results[name] = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds, zero_division=0), 4),
            "recall": round(recall_score(y_test, preds, zero_division=0), 4),
            "f1": round(f1_score(y_test, preds, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, proba), 4),
        }
        fpr, tpr, _ = roc_curve(y_test, proba)
        curves[name] = (fpr, tpr, results[name]["roc_auc"])
        fitted[name] = pipe
        print("        " + "  ".join(f"{k}={v}" for k, v in results[name].items()))

    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    best_model = fitted[best_name]
    print(f"\n[best] {best_name} (roc_auc={results[best_name]['roc_auc']})")

    preds = best_model.predict(X_test)
    report = classification_report(y_test, preds, target_names=["No churn", "Churn"],
                                   zero_division=0)
    print("\n" + report)

    joblib.dump({"pipeline": best_model, "model_name": best_name,
                 "feature_columns": list(X.columns), "target": target},
                os.path.join(args.outdir, "best_model.joblib"))
    with open(os.path.join(args.outdir, "metrics.json"), "w") as fh:
        json.dump({"results": results, "best_model": best_name,
                   "classification_report": report}, fh, indent=2)

    plot_comparison(results, os.path.join(args.outdir, "model_comparison.png"))
    plot_confusion(confusion_matrix(y_test, preds),
                   os.path.join(args.outdir, "confusion_matrix.png"))
    plot_roc(curves, os.path.join(args.outdir, "roc_curves.png"))
    plot_feature_importance(best_model, 15,
                            os.path.join(args.outdir, "feature_importance.png"))

    print(f"\n[done] artefacts written to '{args.outdir}/'")


if __name__ == "__main__":
    main()
