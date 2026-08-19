"""
drift_check.py
---------------
Part G of the assignment: simulate a population whose feature distribution
has shifted away from what the model was trained on, and check whether the
model's behavior on that shifted population looks like data drift,
performance degradation, or both.

How I build the "shifted" dataset:
I start from the real, untouched test set (data/test_set.csv) -- the same
rows the model was already evaluated on once -- and deliberately shift a
few feature distributions to simulate an older, higher-risk population
arriving at a clinic a few years later:
  - age:               +15 years (capped at 95)
  - avg_glucose_level:  x1.3   (simulating a less metabolically healthy group)
  - bmi:                +5     (simulating more obesity)
  - hypertension:       flip a random 20% of 0s to 1s (more diagnosed hypertension)
  - smoking_status:     shift 25% of "never smoked" rows to "smokes"

I deliberately KEEP the original stroke labels attached to each row. That's
a simplification worth being upfront about: changing a patient's age or
glucose level would, in reality, also change their true stroke risk, so
"performance on the shifted set" here is a controlled experiment about how
the model *reacts* to distribution shift, not a claim that these are real
patients with real new outcomes. I say this plainly in observations.md too.

Outputs:
  - data/shifted_data.csv          the shifted dataset itself
  - monitoring/drift_report.json    all the comparison numbers below

Run it with:
    python src/drift_check.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "model" / "model.joblib"
METADATA_PATH = ROOT_DIR / "model" / "metadata.json"
TEST_SET_PATH = ROOT_DIR / "data" / "test_set.csv"
SHIFTED_DATA_PATH = ROOT_DIR / "data" / "shifted_data.csv"
DRIFT_REPORT_PATH = ROOT_DIR / "monitoring" / "drift_report.json"

RANDOM_SEED = 42


def make_shifted_dataset(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.RandomState(RANDOM_SEED)
    shifted = df.copy()

    shifted["age"] = np.clip(shifted["age"] + 15, 0, 95)
    shifted["avg_glucose_level"] = shifted["avg_glucose_level"] * 1.3
    shifted["bmi"] = shifted["bmi"] + 5

    # Flip 20% of hypertension=0 rows to 1
    hyp_zero_idx = shifted.index[shifted["hypertension"] == 0]
    flip_n = int(len(hyp_zero_idx) * 0.20)
    flip_idx = rng.choice(hyp_zero_idx, size=flip_n, replace=False)
    shifted.loc[flip_idx, "hypertension"] = 1

    # Shift 25% of "never smoked" rows to "smokes"
    never_smoked_idx = shifted.index[shifted["smoking_status"] == "never smoked"]
    smoke_n = int(len(never_smoked_idx) * 0.25)
    smoke_idx = rng.choice(never_smoked_idx, size=smoke_n, replace=False)
    shifted.loc[smoke_idx, "smoking_status"] = "smokes"

    return shifted


def compare_numerical(original: pd.DataFrame, shifted: pd.DataFrame, cols):
    comparison = {}
    for col in cols:
        comparison[col] = {
            "original_mean": round(float(original[col].mean()), 3),
            "shifted_mean": round(float(shifted[col].mean()), 3),
            "original_std": round(float(original[col].std()), 3),
            "shifted_std": round(float(shifted[col].std()), 3),
            "mean_shift_pct": round(
                (shifted[col].mean() - original[col].mean()) / original[col].mean() * 100, 1
            ),
        }
    return comparison


def compare_categorical(original: pd.DataFrame, shifted: pd.DataFrame, cols):
    comparison = {}
    for col in cols:
        orig_dist = (original[col].value_counts(normalize=True) * 100).round(1).to_dict()
        shift_dist = (shifted[col].value_counts(normalize=True) * 100).round(1).to_dict()
        comparison[col] = {"original_pct": orig_dist, "shifted_pct": shift_dist}
    return comparison


def main():
    print("=" * 74)
    print("STEP 1: Load the real test set and build the shifted dataset")
    print("=" * 74)
    if not TEST_SET_PATH.exists():
        raise FileNotFoundError(f"{TEST_SET_PATH} not found. Run `python src/train.py` first.")

    original = pd.read_csv(TEST_SET_PATH)
    shifted = make_shifted_dataset(original)
    shifted.to_csv(SHIFTED_DATA_PATH, index=False)
    print(f"Original test rows: {len(original)}")
    print(f"Shifted rows saved to {SHIFTED_DATA_PATH}")

    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    target_col = metadata["target_column"]
    numerical_cols = metadata["numerical_features"]
    categorical_cols = metadata["categorical_features"]

    print("\n" + "=" * 74)
    print("STEP 2: Compare numerical feature statistics")
    print("=" * 74)
    numeric_comparison = compare_numerical(original, shifted, numerical_cols)
    for col, stats in numeric_comparison.items():
        print(f"{col}: original mean={stats['original_mean']}  shifted mean={stats['shifted_mean']}"
              f"  ({stats['mean_shift_pct']:+.1f}%)")

    print("\n" + "=" * 74)
    print("STEP 3: Compare categorical distributions")
    print("=" * 74)
    categorical_comparison = compare_categorical(original, shifted, categorical_cols)
    for col, stats in categorical_comparison.items():
        print(f"{col}:")
        print(f"  original: {stats['original_pct']}")
        print(f"  shifted:  {stats['shifted_pct']}")

    print("\n" + "=" * 74)
    print("STEP 4: Compare model prediction distributions")
    print("=" * 74)
    pipeline = joblib.load(MODEL_PATH)

    X_original = original.drop(columns=[target_col])
    X_shifted = shifted.drop(columns=[target_col])
    y_original = original[target_col]
    y_shifted = shifted[target_col]  # same labels as original, see module docstring

    pred_original = pipeline.predict(X_original)
    proba_original = pipeline.predict_proba(X_original)[:, 1]
    pred_shifted = pipeline.predict(X_shifted)
    proba_shifted = pipeline.predict_proba(X_shifted)[:, 1]

    prediction_comparison = {
        "original_predicted_stroke_rate": round(float(pred_original.mean()), 4),
        "shifted_predicted_stroke_rate": round(float(pred_shifted.mean()), 4),
        "original_mean_probability": round(float(proba_original.mean()), 4),
        "shifted_mean_probability": round(float(proba_shifted.mean()), 4),
    }
    for k, v in prediction_comparison.items():
        print(f"{k}: {v}")

    print("\n" + "=" * 74)
    print("STEP 5: Compare model performance (ground-truth labels available)")
    print("=" * 74)
    def score(y_true, y_pred, y_proba):
        return {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        }

    original_metrics = score(y_original, pred_original, proba_original)
    shifted_metrics = score(y_shifted, pred_shifted, proba_shifted)
    print("Original test-set metrics:", original_metrics)
    print("Shifted-data metrics:     ", shifted_metrics)

    report = {
        "note": (
            "The shifted dataset reuses the original test set's stroke labels "
            "unchanged -- only feature values were shifted. This is a controlled "
            "check of how the model REACTS to distribution shift, not a claim "
            "about real patient outcomes. See observations.md for the full read."
        ),
        "numerical_feature_comparison": numeric_comparison,
        "categorical_feature_comparison": categorical_comparison,
        "prediction_distribution_comparison": prediction_comparison,
        "performance_comparison": {
            "original_test_set": original_metrics,
            "shifted_data": shifted_metrics,
        },
    }

    DRIFT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DRIFT_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved full drift report to {DRIFT_REPORT_PATH}")


if __name__ == "__main__":
    main()
