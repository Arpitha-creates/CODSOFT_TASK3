#!/usr/bin/env python3
"""
CODSOFT Machine Learning Internship -- Task 3
Predict customer churn.

Usage
-----
    python predict.py --file customers.csv          # one customer per row
    python predict.py --json '{"gender":"Female","tenure":2,"MonthlyCharges":95.5,...}'
"""

from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Customer churn predictor")
    ap.add_argument("--file", help="CSV with one customer per row")
    ap.add_argument("--json", help="a single customer as a JSON object")
    ap.add_argument("--model", default="models/best_model.joblib")
    args = ap.parse_args()

    if not args.file and not args.json:
        ap.error("provide either --file or --json")

    bundle = joblib.load(args.model)
    model = bundle["pipeline"]
    cols = bundle["feature_columns"]
    print(f"[info] using model: {bundle.get('model_name', 'unknown')}\n")

    if args.json:
        row = json.loads(args.json)
        df = pd.DataFrame([row]).reindex(columns=cols)
    else:
        df = pd.read_csv(args.file)
        df = df.reindex(columns=cols)

    proba = model.predict_proba(df)[:, 1]
    preds = (proba >= 0.5).astype(int)
    for i, (p, pr) in enumerate(zip(preds, proba), 1):
        label = "CHURN" if p else "stay"
        print(f"{i:>3}. {label:<6}  churn probability = {pr:.3f}")


if __name__ == "__main__":
    main()
