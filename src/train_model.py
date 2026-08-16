"""
train_model.py

Trains Logistic Regression, Random Forest, and XGBoost on a labeled email CSV,
keeps the best by hold-out ROC-AUC, refits the winner on all rows, and writes:

  models/phishing_model.joblib
  models/feature_builder.joblib
  models/metrics.json

Usage:
    python src/train_model.py
    python src/train_model.py --data data/phishing_legit_dataset_KD_10000.csv
    python src/train_model.py --data data/emails.csv
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

from feature_engineering import FeatureBuilder, load_email_csv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PREFERRED_DATA = (
    DATA_DIR / "phishing_legit_dataset_KD_10000.csv",
    DATA_DIR / "emails.csv",
    DATA_DIR / "sample_emails.csv",
)


def default_data_path() -> Path:
    for path in PREFERRED_DATA:
        if path.exists():
            return path
    return PREFERRED_DATA[0]
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


def candidates(y: pd.Series, random_state=42):
    n_pos = max(int(y.sum()), 1)
    n_neg = max(int(len(y) - y.sum()), 1)
    return {
        "logreg": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=2,
            max_depth=20,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
        "xgboost": XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=n_neg / n_pos,
            eval_metric="logloss",
            n_jobs=-1,
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


def _phishing_proba(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        classes = list(getattr(model, "classes_", [0, 1]))
        if 1 in classes:
            return proba[:, classes.index(1)]
        return proba[:, -1]
    return model.predict(X).astype(float)


def evaluate_holdout(model, X_test, y_test, types=None):
    proba = _phishing_proba(model, X_test)
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "avg_precision": float(average_precision_score(y_test, proba)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    if types is not None:
        types = pd.Series(types).reset_index(drop=True)
        y_arr = np.asarray(y_test)
        per_type = {}
        for name in sorted(types.unique()):
            mask = (types == name).to_numpy()
            if not mask.any():
                continue
            n_pos = int(y_arr[mask].sum())
            row = {
                "n": int(mask.sum()),
                "phishing_rate": float(y_arr[mask].mean()),
                "accuracy": float(accuracy_score(y_arr[mask], pred[mask])),
            }
            if n_pos > 0:
                row["recall"] = float(recall_score(y_arr[mask], pred[mask], zero_division=0))
            else:
                row["specificity"] = float((pred[mask] == 0).mean())
            per_type[str(name)] = row
        metrics["per_phishing_type"] = per_type
    return metrics


def _mix_structured(df: pd.DataFrame, data_path: Path) -> pd.DataFrame:
    """Append emails.csv so URL/auth/sender examples exist alongside text-only rows."""
    extra_path = DATA_DIR / "emails.csv"
    if not extra_path.exists() or extra_path.resolve() == Path(data_path).resolve():
        return df
    extra = load_email_csv(extra_path)
    combined = pd.concat([df, extra], ignore_index=True)
    print(f"  mixed in {len(extra)} structured rows from {extra_path.name}")
    return combined


def train(data_path: Path, mix_structured: bool = True):
    df = load_email_csv(data_path)
    if mix_structured:
        df = _mix_structured(df, data_path)
    if "label" not in df.columns:
        raise ValueError(f"{data_path} must include a 'label' column")

    y = encode_labels(df["label"])
    if y.nunique() < 2:
        raise ValueError("Need both phishing and legitimate rows in the training CSV.")

    print(f"Training on {len(df)} rows from {data_path}")
    print(f"  phishing={int(y.sum())}  legit={int((1 - y).sum())}")
    if "phishing_type" in df.columns:
        counts = df["phishing_type"].value_counts()
        print("  types:", ", ".join(f"{k}={v}" for k, v in counts.items()))
    print()

    use_holdout = len(df) >= 50
    types = df["phishing_type"] if "phishing_type" in df.columns else None
    split_types = None

    if use_holdout:
        split_kwargs = dict(test_size=0.2, stratify=y, random_state=42)
        if types is not None:
            df_train, df_test, y_train, y_test, _, types_test = train_test_split(
                df, y, types, **split_kwargs
            )
            split_types = types_test
        else:
            df_train, df_test, y_train, y_test = train_test_split(df, y, **split_kwargs)
        fb_eval = FeatureBuilder()
        X_train = fb_eval.fit_transform(df_train)
        X_test = fb_eval.transform(df_test)
        print(f"  holdout 80/20  train={len(df_train)}  test={len(df_test)}  "
              f"features={X_train.shape[1]}\n")
    else:
        fb_eval = FeatureBuilder()
        X_train = fb_eval.fit_transform(df)
        y_train = y
        X_test = y_test = None
        print(f"  features={X_train.shape[1]}  (too few rows for a hold-out set)\n")

    splits = cv_splits(y_train)
    cv_metrics = {}
    holdout_metrics = {}
    best_name, best_score, best_model = None, -1.0, None
    fitted = {}

    for name, model in candidates(y_train).items():
        if splits is None:
            model.fit(X_train, y_train)
            cv_score = float("nan")
            print(f"  {name:15}  too few rows for CV — fitted on train split")
        else:
            scores = cross_val_score(model, X_train, y_train, cv=splits, scoring="roc_auc")
            cv_score = float(scores.mean())
            print(f"  {name:15}  CV ROC-AUC {cv_score:.3f}  "
                  f"(folds: {', '.join(f'{s:.3f}' for s in scores)})")
            model.fit(X_train, y_train)
        cv_metrics[name] = cv_score
        fitted[name] = model

        if use_holdout:
            h_metrics = evaluate_holdout(model, X_test, y_test, types=split_types)
            holdout_metrics[name] = h_metrics
            cm = h_metrics["confusion_matrix"]
            print(
                f"  {name:15}  holdout ROC-AUC {h_metrics['roc_auc']:.3f}  "
                f"P={h_metrics['precision']:.3f}  R={h_metrics['recall']:.3f}  "
                f"F1={h_metrics['f1']:.3f}  "
                f"tn={cm['tn']} fp={cm['fp']} fn={cm['fn']} tp={cm['tp']}"
            )
            comparable = h_metrics["roc_auc"]
        else:
            comparable = -1.0 if pd.isna(cv_score) else cv_score

        # Near-ties: prefer logistic regression — it is better calibrated
        # than trees when this corpus is linearly separable.
        prefer = {"logreg": 3, "xgboost": 2, "random_forest": 1}
        take = False
        if best_name is None or comparable > best_score + 0.002:
            take = True
        elif abs(comparable - best_score) <= 0.002 and prefer.get(name, 0) > prefer.get(best_name, 0):
            take = True
        if take:
            best_name, best_score, best_model = name, comparable, model

    if best_model is None:
        raise RuntimeError("No model was trained.")

    if use_holdout:
        print(f"\nHold-out classification report for winner ({best_name}):")
        proba = _phishing_proba(fitted[best_name], X_test)
        pred = (proba >= 0.5).astype(int)
        print(classification_report(y_test, pred, target_names=["legit", "phishing"], digits=3))

    print("\nRefitting winner on the full dataset…")
    fb = FeatureBuilder()
    X_all = fb.fit_transform(df)
    production = candidates(y)[best_name]
    production.fit(X_all, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(production, MODEL_PATH)
    joblib.dump(fb, BUILDER_PATH)
    payload = {
        "winner": best_name,
        "selection_metric": "holdout_roc_auc" if use_holdout else "cv_roc_auc",
        "cv_roc_auc": cv_metrics,
        "n_rows": int(len(df)),
        "n_features": int(X_all.shape[1]),
        "phishing_rows": int(y.sum()),
        "legit_rows": int((1 - y).sum()),
        "data": str(data_path),
        "mixed_structured": bool(mix_structured and (DATA_DIR / "emails.csv").exists()),
    }
    if use_holdout:
        payload["holdout"] = holdout_metrics
        payload["holdout_winner"] = holdout_metrics[best_name]
    METRICS_PATH.write_text(json.dumps(payload, indent=2))

    print(f"\nWinner: {best_name}"
          + (f"  (hold-out ROC-AUC {best_score:.3f})" if best_score >= 0 else ""))
    print(f"Wrote {MODEL_PATH}")
    print(f"Wrote {BUILDER_PATH}")
    print(f"Wrote {METRICS_PATH}")
    return best_name, cv_metrics


if __name__ == "__main__":
    _ensure_src_on_path()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument(
        "--no-mix",
        action="store_true",
        help="Do not append data/emails.csv when training on another CSV",
    )
    args = parser.parse_args()
    data_path = args.data or default_data_path()
    if not data_path.exists():
        raise FileNotFoundError(
            f"No training CSV at {data_path}. "
            "Put phishing_legit_dataset_KD_10000.csv in data/ "
            "or run: python src/generate_dataset.py"
        )
    train(data_path, mix_structured=not args.no_mix)
