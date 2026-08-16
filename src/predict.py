"""
predict.py

Loads the trained pipeline and scores one email or a CSV of emails.
Prints a risk percentage, risk band, and the signals that fired.

Usage:
    python src/predict.py
    python src/predict.py --subject "..." --body "..." --sender "..."
    python src/predict.py --csv data/emails.csv --json
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

from feature_engineering import (
    FeatureBuilder,
    extract_urls,
    load_email_csv,
    structured_risk_boost,
    triggered_reasons,
)
from rules import rule_override_risk

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "phishing_model.joblib"
BUILDER_PATH = MODELS_DIR / "feature_builder.joblib"

SAFE_MAX = 30
HIGH_MIN = 70

DEMO_EMAIL = {
    "subject": "URGENT: Your PayPal account will be suspended",
    "body": (
        "We detected unusual activity. Verify your password immediately or "
        "your account will be closed. Click http://192.168.1.50/paypal to "
        "confirm your identity."
    ),
    "sender": "PayPal Security <security@paypa1.com>",
    "reply_to": "verify@accounts-secure.xyz",
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
            "Train first: python src/train_model.py"
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
    return float(model.predict(X)[0])


def risk_band(risk_pct: float) -> str:
    if risk_pct >= HIGH_MIN:
        return "HIGH RISK"
    if risk_pct >= SAFE_MAX:
        return "SUSPICIOUS"
    return "SAFE"


def score_email(fields, model=None, fb=None):
    """
    Score a dict of email fields.

    Combines the ML probability, a structured-signal boost, and Layer 1
    smoking-gun rules. Returns risk_pct, band, label, reasons, proba.
    """
    if model is None or fb is None:
        model, fb = load_pipeline()

    row = dict(fields)
    if not row.get("urls"):
        row["urls"] = " ".join(extract_urls(row.get("body", "") or ""))

    X = fb.transform_fields(row)
    p = float(np.clip(_phishing_probability(model, X), 0.0, 1.0))
    boost = structured_risk_boost(row)
    if boost > 0:
        p = float(np.clip(p + (1.0 - p) * boost, 0.0, 1.0))

    floor, rule_reasons = rule_override_risk(row)
    if floor is not None:
        p = max(p, floor)

    risk_pct = round(p * 100.0, 1)
    reasons = triggered_reasons(row) + rule_reasons
    # Preserve order, drop duplicates
    seen = set()
    unique_reasons = []
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            unique_reasons.append(reason)

    return {
        "risk_pct": risk_pct,
        "band": risk_band(risk_pct),
        "label": "phishing" if risk_pct >= 50 else "legit",
        "reasons": unique_reasons,
        "proba": p,
        "boost": boost,
    }


def _print_result(result):
    print(f"PHISHING RISK: {result['risk_pct']:.1f}%  [{result['band']}]  ({result['label']})")
    if result["reasons"]:
        print("Reasons:")
        for reason in result["reasons"]:
            print(f"  ✓ {reason}")
    else:
        print("Reasons: (no structured red flags fired)")


def score_csv(path: Path, model, fb):
    df = load_email_csv(path)
    results = []
    for row in df.to_dict(orient="records"):
        result = score_email(row, model=model, fb=fb)
        result["subject"] = str(row.get("subject", ""))[:80]
        result["sender"] = str(row.get("sender", ""))[:60]
        results.append(result)
    return results


if __name__ == "__main__":
    _ensure_src_on_path()
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default=None)
    parser.add_argument("--body", default=None)
    parser.add_argument("--sender", default=None)
    parser.add_argument("--reply_to", default=None)
    parser.add_argument("--urls", default=None)
    parser.add_argument("--spf", default=None)
    parser.add_argument("--dkim", default=None)
    parser.add_argument("--dmarc", default=None)
    parser.add_argument("--num_attachments", type=int, default=None)
    parser.add_argument("--attachment_types", default=None)
    parser.add_argument("--csv", type=Path, help="Score every row in a CSV")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text")
    args = parser.parse_args()

    model, fb = load_pipeline()

    if args.csv:
        results = score_csv(args.csv, model, fb)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            summary = {"SAFE": 0, "SUSPICIOUS": 0, "HIGH RISK": 0}
            for result in results:
                summary[result["band"]] += 1
                print(
                    f"[{result['band']:10}] {result['risk_pct']:5.1f}%  "
                    f"from={result['sender']:<40}  {result['subject']}"
                )
            print("\n--- Summary ---")
            for band, count in summary.items():
                print(f"{band}: {count}")
    else:
        override_keys = (
            "subject", "body", "sender", "reply_to", "urls",
            "spf", "dkim", "dmarc", "num_attachments", "attachment_types",
        )
        if all(getattr(args, key) is None for key in override_keys):
            fields = dict(DEMO_EMAIL)
        else:
            fields = {
                "subject": args.subject or "",
                "body": args.body or "",
                "sender": args.sender or "",
                "reply_to": args.reply_to or "",
                "urls": args.urls or "",
                "spf": args.spf or "",
                "dkim": args.dkim or "",
                "dmarc": args.dmarc or "",
                "num_attachments": args.num_attachments or 0,
                "attachment_types": args.attachment_types or "",
            }
        result = score_email(fields, model=model, fb=fb)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            _print_result(result)
