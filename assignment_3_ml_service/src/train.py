"""
train.py
--------
This script does the full training job for the stroke-risk model that
Assignment 3 turns into a service:

1. Load and clean the raw data (same cleaning logic as Assignment 1).
2. Split it into train / validation / test (60 / 20 / 20, stratified).
3. Build ONE scikit-learn Pipeline that contains both preprocessing
   (ColumnTransformer) and the model, so preprocessing and modeling are
   never separated.
4. Tune the model's hyperparameters with RandomizedSearchCV, using
   cross-validation on the training set only.
5. Check the tuned model once on the validation set (a true held-out
   check, separate from the cross-validation score reported by the
   search itself).
6. Evaluate the final model exactly once on the untouched test set.
7. Persist the complete fitted Pipeline (preprocessing + model together)
   to model/model.joblib, and write model/metadata.json alongside it.

Run it with:
    python src/train.py
"""

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import randint

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "stroke-data.csv"
MODEL_DIR = ROOT_DIR / "model"
MODEL_PATH = MODEL_DIR / "model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"

# I save the exact test split to disk so evaluate.py can score the exact
# same held-out rows later, without silently re-splitting the data a
# second time (which could accidentally leak a different random split).
TEST_SET_PATH = ROOT_DIR / "data" / "test_set.csv"

RANDOM_SEED = 42
MODEL_VERSION = "1.0"


def load_and_clean_data(path: Path) -> pd.DataFrame:
    """Load the raw CSV and apply the same cleaning I used in Assignment 1."""
    df = pd.read_csv(path)

    # bmi is stored as text in the raw file, with "N/A" for missing values.
    # I convert it to a real numeric column; non-numeric text becomes NaN,
    # which SimpleImputer then handles properly downstream.
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")

    # id is a unique identifier per patient -- it carries no medical signal,
    # so I drop it before it ever reaches the model.
    df = df.drop(columns=["id"])

    return df


def build_pipeline(numerical_cols, categorical_cols) -> Pipeline:
    """
    Build ONE pipeline containing preprocessing + model. Numerical and
    categorical columns get different preprocessing steps
    (ColumnTransformer), and everything downstream -- including the
    classifier -- lives inside the same Pipeline object. This is what lets
    me call pipeline.fit(X_train) once and then safely call
    pipeline.predict(X_val) / pipeline.predict(X_test) afterwards: those
    calls only ever *transform* validation/test data through the
    already-fitted preprocessing steps. They never re-fit on it.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])

    # class_weight is tuned below (None vs "balanced"), so I leave it as a
    # placeholder here -- RandomizedSearchCV will set the real value.
    model = RandomForestClassifier(random_state=RANDOM_SEED)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])
    return pipeline


def main():
    print("=" * 74)
    print("STEP 1: Load and clean data")
    print("=" * 74)
    df = load_and_clean_data(DATA_PATH)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns after cleaning.")
    print(f"Missing bmi values after conversion: {df['bmi'].isnull().sum()}")

    target_col = "stroke"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # pd.api.types.is_numeric_dtype is version-safe across pandas releases
    # (older pandas stores text as "object" dtype, pandas 3+ uses a "str"
    # dtype instead -- this check works correctly either way).
    categorical_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    numerical_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    # hypertension and heart_disease are 0/1 flags, not magnitudes -- treat
    # them as categorical so they get one-hot encoded, not scaled.
    for flag_col in ["hypertension", "heart_disease"]:
        if flag_col in numerical_cols:
            numerical_cols.remove(flag_col)
            categorical_cols.append(flag_col)

    print(f"Numerical columns: {numerical_cols}")
    print(f"Categorical columns: {categorical_cols}")

    print("\n" + "=" * 74)
    print("STEP 2: Train / Validation / Test split (60 / 20 / 20, stratified)")
    print("=" * 74)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=RANDOM_SEED, stratify=y_temp
    )
    print(f"Train: {X_train.shape[0]} rows | stroke rate: {y_train.mean():.4f}")
    print(f"Val:   {X_val.shape[0]} rows | stroke rate: {y_val.mean():.4f}")
    print(f"Test:  {X_test.shape[0]} rows | stroke rate: {y_test.mean():.4f}")

    # Save the test set to disk so evaluate.py scores the exact same rows.
    test_df = X_test.copy()
    test_df[target_col] = y_test.values
    test_df.to_csv(TEST_SET_PATH, index=False)
    print(f"Saved test set to {TEST_SET_PATH}")

    print("\n" + "=" * 74)
    print("STEP 3: Build the preprocessing + model Pipeline")
    print("=" * 74)
    pipeline = build_pipeline(numerical_cols, categorical_cols)
    print("Pipeline steps:", [name for name, _ in pipeline.steps])

    print("\n" + "=" * 74)
    print("STEP 4: Hyperparameter tuning with RandomizedSearchCV")
    print("=" * 74)
    print("I'm tuning RandomForestClassifier -- it has several hyperparameters that")
    print("genuinely change how it behaves, and it's the model whose ROC-AUC held up")
    print("best on validation/test in Assignment 1's benchmark once the deep single")
    print("tree overfit and the boosting models struggled to generalize recall.")
    print()
    print("I score on ROC-AUC, not accuracy, because stroke cases are only ~5% of the")
    print("data -- a model that always predicts 'no stroke' would score ~95% accuracy")
    print("while being useless, so accuracy would actively mislead the search.")

    param_distributions = {
        "classifier__n_estimators": randint(100, 500),
        "classifier__max_depth": [None, 5, 10, 15, 20, 25],
        "classifier__min_samples_leaf": randint(1, 10),
        "classifier__max_features": ["sqrt", "log2", None],
        "classifier__class_weight": [None, "balanced"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=25,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    t0 = time.perf_counter()
    # IMPORTANT: the search is fit ONLY on the training set. Cross-validation
    # happens by splitting X_train internally into folds -- the validation
    # and test sets are never touched here, so nothing about them can leak
    # into the hyperparameter choice.
    search.fit(X_train, y_train)
    tuning_time = time.perf_counter() - t0

    print(f"\nTuning finished in {tuning_time:.1f}s")
    print(f"Best cross-validation ROC-AUC (on train folds): {search.best_score_:.4f}")
    print("Best parameters found:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")

    best_pipeline = search.best_estimator_

    print("\n" + "=" * 74)
    print("STEP 5: Check the tuned model on the VALIDATION set (held out, untouched by CV)")
    print("=" * 74)
    # This call only transforms X_val through the already-fitted preprocessor
    # inside best_pipeline -- it does NOT call fit or fit_transform on it.
    y_val_pred = best_pipeline.predict(X_val)
    y_val_proba = best_pipeline.predict_proba(X_val)[:, 1]
    val_roc_auc = roc_auc_score(y_val, y_val_proba)
    val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
    val_recall = recall_score(y_val, y_val_pred, zero_division=0)
    val_precision = precision_score(y_val, y_val_pred, zero_division=0)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation ROC-AUC: {val_roc_auc:.4f}")
    print(f"Validation F1:      {val_f1:.4f}")
    print(f"Validation Recall:  {val_recall:.4f}")
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation Accuracy:  {val_accuracy:.4f}")

    print("\n" + "=" * 74)
    print("STEP 6: Final evaluation on the TEST set (touched exactly once)")
    print("=" * 74)
    # Same discipline as above: predict() only transforms X_test through the
    # already-fitted pipeline. No fit_transform call happens on test data
    # anywhere in this script.
    y_test_pred = best_pipeline.predict(X_test)
    y_test_proba = best_pipeline.predict_proba(X_test)[:, 1]
    test_metrics = {
        "accuracy": accuracy_score(y_test, y_test_pred),
        "precision": precision_score(y_test, y_test_pred, zero_division=0),
        "recall": recall_score(y_test, y_test_pred, zero_division=0),
        "f1": f1_score(y_test, y_test_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_test_proba),
    }
    for k, v in test_metrics.items():
        print(f"Test {k}: {v:.4f}")

    print("\n" + "=" * 74)
    print("STEP 7: Persist the pipeline and its metadata")
    print("=" * 74)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # I persist the WHOLE pipeline object -- preprocessing steps and the
    # classifier together -- not just the bare RandomForestClassifier. That
    # way the API never has to reimplement imputation/scaling/encoding by
    # hand; loading model.joblib and calling .predict() on raw input columns
    # does the entire job correctly and consistently with how it was trained.
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"Saved pipeline to {MODEL_PATH}")

    metadata = {
        "model_version": MODEL_VERSION,
        "algorithm": "RandomForestClassifier",
        "training_date_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "target_column": target_col,
        "raw_feature_names": numerical_cols + categorical_cols,
        "numerical_features": numerical_cols,
        "categorical_features": categorical_cols,
        "train_rows": int(X_train.shape[0]),
        "val_rows": int(X_val.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "tuning": {
            "search_method": "RandomizedSearchCV",
            "n_iter": 25,
            "cv_folds": 5,
            "scoring": "roc_auc",
            "best_params": search.best_params_,
            "best_cv_roc_auc": search.best_score_,
            "tuning_time_sec": tuning_time,
        },
        "validation_metrics": {
            "accuracy": val_accuracy,
            "precision": val_precision,
            "recall": val_recall,
            "f1": val_f1,
            "roc_auc": val_roc_auc,
        },
        "test_metrics": test_metrics,
        "package_versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {METADATA_PATH}")

    print("\n" + "=" * 74)
    print("DONE. model/model.joblib and model/metadata.json are ready for the API.")
    print("=" * 74)


if __name__ == "__main__":
    main()
