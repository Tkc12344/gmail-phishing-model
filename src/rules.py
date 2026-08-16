"""
Layer 1 — high-precision rule combos.

These fire only on smoking-gun combinations (unknown infra + credential
ask, lookalike sender + auth failure, etc.). They can force a high-risk
band before the model score is used, and they add extra reasons.
"""

from feature_engineering import STRUCTURED_FEATURE_NAMES, structured_feature_row

SMOKING_GUN_REASONS = {
    "ip_plus_credentials": "Rule: credential request plus a raw-IP link",
    "lookalike_plus_auth": "Rule: brand-lookalike sender that fails authentication",
    "spoof_plus_credentials": "Rule: spoofed brand display name asking for credentials",
    "risky_attachment_plus_urgency": "Rule: risky attachment with urgent language",
    "obfuscated_login": "Rule: obfuscated or HTTP login link",
}


def _flags(fields: dict) -> dict:
    vec = structured_feature_row(fields)
    return dict(zip(STRUCTURED_FEATURE_NAMES, vec))


def rule_hits(fields: dict) -> list:
    """Return the smoking-gun rule reason strings that fired."""
    f = _flags(fields)
    hits = []
    if f["has_ip_url"] and f["has_credential_request"]:
        hits.append(SMOKING_GUN_REASONS["ip_plus_credentials"])
    if (f["sender_lookalike"] or f["display_name_spoof"] or f["digit_in_domain"]) and f["auth_fail_count"] >= 2:
        hits.append(SMOKING_GUN_REASONS["lookalike_plus_auth"])
    if f["display_name_spoof"] and f["has_credential_request"]:
        hits.append(SMOKING_GUN_REASONS["spoof_plus_credentials"])
    if f["has_risky_attachment"] and f["has_urgency"]:
        hits.append(SMOKING_GUN_REASONS["risky_attachment_plus_urgency"])
    if (f["has_at_obfuscation"] or f["http_not_https"] or f["has_punycode"]) and (
        f["has_credential_request"] or f["has_urgency"]
    ):
        hits.append(SMOKING_GUN_REASONS["obfuscated_login"])
    return hits


def rule_override_risk(fields: dict):
    """
    If a smoking-gun combo fired, return a floor probability (0-1).
    Otherwise return None and let the model decide.
    """
    hits = rule_hits(fields)
    if not hits:
        return None, []
    # One combo → at least suspicious; two or more → high risk.
    floor = 0.78 if len(hits) >= 2 else 0.62
    return floor, hits
