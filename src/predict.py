"""
predict.py

Loads the trained pipeline and scores one email. Prints a risk percentage
plus the structured signals that fired, in plain English.

Usage:
    python src/predict.py
    python src/predict.py --subject "..." --body "..." --sender "..."
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

from feature_engineering import FeatureBuilder, extract_urls, triggered_reasons

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "phishing_model.joblib"
BUILDER_PATH = MODELS_DIR / "feature_builder.joblib"

DEMO_EMAIL = {
    "subject": "URGENT: Your PayPal account will be suspended",
    "body": (
        "We detected unusual activity. Verify your password immediately or "
        "your account will be closed. Click http://192.168.1.50/paypal to "
        "confirm your identity."
    ),
    "sender": "security@paypa1.com",
    "urls": "http://192.168.1.50/paypal",
    "spf": "fail",
    "dkim": "fail",
    "dmarc": "fail",
    "num_attachments": 0,
    "attachment_types": "",
}


def _ensure_src_on_path():
    src = str(Path(__file__).resolve().parent)
    if src not in sys.path:
        sys.path.insert(0, src)


def load_pipeline():
    """Return (estimator, FeatureBuilder) saved by train_model.py."""
    if not MODEL_PATH.exists() or not BUILDER_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MODEL_PATH.name} or {BUILDER_PATH.name} in {MODELS_DIR}. "
            "Train first: python src/train_model.py --data data/sample_emails.csv"
        )
    model = joblib.load(MODEL_PATH)
    fb = joblib.load(BUILDER_PATH)
    if not isinstance(fb, FeatureBuilder):
        raise TypeError(f"Expected FeatureBuilder, got {type(fb)}")
    return model, fb


def _phishing_probability(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        classes = list(getattr(model, "classes_", [0, 1]))
        if 1 in classes:
            return float(proba[classes.index(1)])
        return float(proba[-1])
    pred = float(model.predict(X)[0])
    return pred


def score_email(fields, model=None, fb=None):
    """
    Score a dict of email fields.

    Returns:
      risk_pct (0-100), label ('phishing'|'legit'), reasons (list of str)
    """
    if model is None or fb is None:
        model, fb = load_pipeline()

    row = dict(fields)
    if not row.get("urls"):
        row["urls"] = " ".join(extract_urls(row.get("body", "") or ""))

    X = fb.transform_fields(row)
    p = float(np.clip(_phishing_probability(model, X), 0.0, 1.0))
    risk_pct = round(p * 100.0, 1)
    return {
        "risk_pct": risk_pct,
        "label": "phishing" if risk_pct >= 50 else "legit",
        "reasons": triggered_reasons(row),
        "proba": p,
    }


def _print_result(result):
    print(f"PHISHING RISK: {result['risk_pct']:.1f}% ({result['label']})")
    if result["reasons"]:
        print("Reasons:")
        for reason in result["reasons"]:
            print(f"  ✓ {reason}")
    else:
        print("Reasons: (no structured red flags fired)")


if __name__ == "__main__":
    _ensure_src_on_path()
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default=DEMO_EMAIL["subject"])
    parser.add_argument("--body", default=DEMO_EMAIL["body"])
    parser.add_argument("--sender", default=DEMO_EMAIL["sender"])
    parser.add_argument("--urls", default=DEMO_EMAIL["urls"])
    parser.add_argument("--spf", default=DEMO_EMAIL["spf"])
    parser.add_argument("--dkim", default=DEMO_EMAIL["dkim"])
    parser.add_argument("--dmarc", default=DEMO_EMAIL["dmarc"])
    parser.add_argument("--num_attachments", type=int, default=0)
    parser.add_argument("--attachment_types", default="")
    args = parser.parse_args()

    fields = {
        "subject": args.subject,
        "body": args.body,
        "sender": args.sender,
        "urls": args.urls,
        "spf": args.spf,
        "dkim": args.dkim,
        "dmarc": args.dmarc,
        "num_attachments": args.num_attachments,
        "attachment_types": args.attachment_types,
    }
    _print_result(score_email(fields))
