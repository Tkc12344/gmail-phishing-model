"""
train_model.py

Trains Logistic Regression, Random Forest, and XGBoost on a labeled email CSV,
keeps the best by stratified ROC-AUC, and writes:

  models/phishing_model.joblib
  models/feature_builder.joblib

Usage:
    python src/train_model.py --data data/sample_emails.csv
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from feature_engineering import FeatureBuilder

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "sample_emails.csv"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "phishing_model.joblib"
BUILDER_PATH = MODELS_DIR / "feature_builder.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

PHISHING_LABELS = {"phishing", "phish", "spam", "1", "true", "yes"}


def _ensure_src_on_path():
    src = str(Path(__file__).resolve().parent)
    if src not in sys.path:
        sys.path.insert(0, src)


def encode_labels(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.lower().str.strip()
    numeric = set(s.unique()) <= {"0", "1", "0.0", "1.0", ""}
    if numeric:
        return (s.replace("", "0").astype(float) >= 1).astype(int)
    return s.isin(PHISHING_LABELS).astype(int)


def candidates(random_state=42):
    return {
        "logreg": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=random_state,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            n_jobs=1,
            random_state=random_state,
        ),
    }


def cv_splits(y: pd.Series):
    pos = int(y.sum())
    neg = int(len(y) - pos)
    n_splits = min(5, pos, neg)
    if n_splits < 2:
        return None
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


def train(data_path: Path):
    df = pd.read_csv(data_path)
    if "label" not in df.columns:
        raise ValueError(f"{data_path} must include a 'label' column")

    y = encode_labels(df["label"])
    if y.nunique() < 2:
        raise ValueError("Need both phishing and legitimate rows in the training CSV.")

    fb = FeatureBuilder()
    X = fb.fit_transform(df)

    splits = cv_splits(y)
    metrics = {}
    best_name, best_score, best_model = None, -1.0, None

    print(f"Training on {len(df)} rows from {data_path}")
    print(f"  phishing={int(y.sum())}  legit={int((1 - y).sum())}  "
          f"features={X.shape[1]}\n")

    for name, model in candidates().items():
        if splits is None:
            model.fit(X, y)
            score = float("nan")
            print(f"  {name:15}  too few rows for CV — fitted on all data")
        else:
            scores = cross_val_score(model, X, y, cv=splits, scoring="roc_auc")
            score = float(scores.mean())
            print(f"  {name:15}  CV ROC-AUC {score:.3f}  "
                  f"(folds: {', '.join(f'{s:.3f}' for s in scores)})")
            model.fit(X, y)
        metrics[name] = score
        comparable = -1.0 if pd.isna(score) else score
        if comparable > best_score:
            best_name, best_score, best_model = name, comparable, model

    if best_model is None:
        raise RuntimeError("No model was trained.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(fb, BUILDER_PATH)
    METRICS_PATH.write_text(json.dumps({
        "winner": best_name,
        "cv_roc_auc": metrics,
        "n_rows": int(len(df)),
        "n_features": int(X.shape[1]),
        "phishing_rows": int(y.sum()),
        "legit_rows": int((1 - y).sum()),
        "data": str(data_path),
    }, indent=2))

    print(f"\nWinner: {best_name}"
          + (f"  (CV ROC-AUC {best_score:.3f})" if best_score >= 0 else ""))
    print(f"Wrote {MODEL_PATH}")
    print(f"Wrote {BUILDER_PATH}")
    return best_name, metrics


if __name__ == "__main__":
    _ensure_src_on_path()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"No training CSV at {args.data}")
    train(args.data)
