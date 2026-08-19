"""
evaluate.py
-----------
A standalone evaluation script, separate from train.py, so I can re-check
the persisted model's performance at any time without re-training it.

It loads:
  - the persisted pipeline (model/model.joblib)
  - the exact test set that train.py saved (data/test_set.csv)

...and reports the same metrics train.py reported, plus a confusion matrix,
so I can confirm the saved artifact still behaves the way metadata.json
claims it does.

Run it with:
    python src/evaluate.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "model" / "model.joblib"
METADATA_PATH = ROOT_DIR / "model" / "metadata.json"
TEST_SET_PATH = ROOT_DIR / "data" / "test_set.csv"
REPORT_PATH = ROOT_DIR / "model" / "evaluation_report.json"


def main():
    print("=" * 74)
    print("Loading persisted pipeline and test set")
    print("=" * 74)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Run `python src/train.py` first."
        )
    if not TEST_SET_PATH.exists():
        raise FileNotFoundError(
            f"{TEST_SET_PATH} not found. Run `python src/train.py` first "
            "(it saves the test set alongside the trained model)."
        )

    pipeline = joblib.load(MODEL_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    test_df = pd.read_csv(TEST_SET_PATH)
    target_col = metadata["target_column"]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    print(f"Loaded model version {metadata['model_version']} "
          f"({metadata['algorithm']}), trained {metadata['training_date_utc']}")
    print(f"Test set: {X_test.shape[0]} rows")

    print("\n" + "=" * 74)
    print("Scoring the pipeline on the test set")
    print("=" * 74)
    # As in train.py: this only transforms X_test through the already-fitted
    # pipeline. No fit or fit_transform is called on test data here.
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    for k, v in metrics.items():
        print(f"{k:10s}: {v:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print("                 pred: no-stroke   pred: stroke")
    print(f"actual no-stroke      {cm[0][0]:>8d}       {cm[0][1]:>8d}")
    print(f"actual stroke         {cm[1][0]:>8d}       {cm[1][1]:>8d}")

    print("\n" + "=" * 74)
    print("Comparing against the metrics recorded in metadata.json at training time")
    print("=" * 74)
    for k, v in metrics.items():
        trained_val = metadata["test_metrics"].get(k)
        match = "MATCH" if trained_val is not None and abs(trained_val - v) < 1e-9 else "DIFFERENT"
        print(f"{k:10s}: now={v:.4f}  trained={trained_val:.4f}  [{match}]")

    report = {
        "model_version": metadata["model_version"],
        "evaluated_rows": int(X_test.shape[0]),
        "metrics": metrics,
        "confusion_matrix": {
            "true_negative": int(cm[0][0]), "false_positive": int(cm[0][1]),
            "false_negative": int(cm[1][0]), "true_positive": int(cm[1][1]),
        },
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved evaluation report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
