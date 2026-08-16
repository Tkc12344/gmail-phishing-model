"""
feature_engineering.py

Turns raw email fields into the feature matrix the classifier trains on, and
into the per-signal flags that predict.py translates into plain-English
reasons.

Expected columns (extras are ignored; missing ones default to empty/0):
  subject, body, sender, urls, spf, dkim, dmarc,
  num_attachments, attachment_types, label
"""

import re
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

URL_RE = re.compile(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", re.IGNORECASE)
EMAIL_RE = re.compile(r"[\w.+\-]+@([\w.\-]+\.[a-z]{2,})", re.IGNORECASE)
IP_IN_URL_RE = re.compile(r"(?:https?://)?(\d{1,3}(?:\.\d{1,3}){3})")

URGENCY_PHRASES = (
    "urgent", "immediately", "act now", "right now", "expires", "final notice",
    "account will be", "suspended", "locked", "disabled", "verify now",
    "confirm now", "unusual activity", "unauthorized", "limited time",
    "within 24 hours", "within 24hrs", "failure to", "will be closed",
    "will be terminated", "click below", "click here",
)

CREDENTIAL_PHRASES = (
    "password", "passcode", "login", "log in", "sign in", "username",
    "verify your identity", "confirm your identity", "social security",
    "credit card", "bank account", "routing number", "pin number",
    "one-time code", "otp", "credentials", "account details",
)

SUSPICIOUS_TLDS = {
    "xyz", "top", "click", "tk", "ml", "ga", "cf", "gq", "zip", "review",
    "country", "stream", "gdn", "work", "link", "rest", "cam", "icu",
}

SHORTENER_HOSTS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "cutt.ly", "rebrand.ly", "tiny.cc",
}

RISKY_EXTS = {
    "exe", "scr", "js", "jse", "vbs", "vbe", "bat", "cmd", "com", "pif",
    "jar", "msi", "iso", "img", "docm", "xlsm", "pptm", "hta", "ps1",
}

BRAND_DOMAINS = {
    "paypal": ["paypal.com"],
    "google": ["google.com", "gmail.com", "googlemail.com", "accounts.google.com"],
    "microsoft": ["microsoft.com", "live.com", "outlook.com", "office.com"],
    "apple": ["apple.com", "icloud.com"],
    "amazon": ["amazon.com", "amazon.co.uk"],
    "netflix": ["netflix.com"],
    "facebook": ["facebook.com", "fb.com", "meta.com"],
    "instagram": ["instagram.com"],
    "bankofamerica": ["bankofamerica.com"],
    "chase": ["chase.com"],
    "wells fargo": ["wellsfargo.com"],
}

LEET_TABLE = str.maketrans({
    "0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a",
})

AUTH_FAIL = {"fail", "softfail", "none", "neutral", "permerror", "temperror"}

STRUCTURED_FEATURE_NAMES = [
    "urgency_hits",
    "has_urgency",
    "credential_hits",
    "has_credential_request",
    "all_caps_ratio",
    "exclamation_density",
    "url_count",
    "has_ip_url",
    "suspicious_tld_count",
    "has_shortener",
    "has_at_obfuscation",
    "max_dot_count",
    "sender_lookalike",
    "hyphenated_domain",
    "digit_in_domain",
    "spf_fail",
    "dkim_fail",
    "dmarc_fail",
    "auth_fail_count",
    "num_attachments",
    "has_risky_attachment",
]


def extract_urls(text: str):
    """Return http(s)/www URLs found in a body (used by the Gmail scanner)."""
    if not text:
        return []
    return URL_RE.findall(text)


def _as_text(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def _urls_from_row(row):
    listed = _as_text(row.get("urls", "")).split()
    listed = [u for u in listed if u]
    if listed:
        return listed
    return extract_urls(_as_text(row.get("body", "")))


def _sender_domain(sender: str):
    sender = _as_text(sender).lower()
    m = EMAIL_RE.search(sender)
    if m:
        return m.group(1).lower()
    if "@" in sender:
        return sender.rsplit("@", 1)[-1].strip("> ").lower()
    return sender.strip().lower()


def _host(url: str):
    raw = url if "://" in url else "http://" + url
    try:
        return (urlparse(raw).hostname or "").lower()
    except ValueError:
        return ""


def _count_phrases(text: str, phrases):
    return sum(1 for p in phrases if p in text)


def _auth_fail(value: str):
    return 1.0 if _as_text(value).lower().strip() in AUTH_FAIL else 0.0


def _has_at_obfuscation(urls):
    """True when a URL uses userinfo@host to hide the real destination."""
    for url in urls:
        rest = url.split("://", 1)[-1]
        host_part = rest.split("/", 1)[0]
        if "@" in host_part:
            return 1.0
    return 0.0


def _lookalike_brand(domain: str):
    if not domain:
        return 0.0
    normalized = domain.translate(LEET_TABLE)
    for brand, legit in BRAND_DOMAINS.items():
        brand_key = brand.replace(" ", "")
        if domain in legit or domain.endswith("." + legit[0]):
            return 0.0
        in_domain = brand_key in domain.replace("-", "") or brand_key in normalized.replace("-", "")
        if in_domain and domain not in legit:
            return 1.0
    return 0.0


def structured_feature_row(row):
    """Numeric feature vector for one email (dict or Series)."""
    if hasattr(row, "to_dict"):
        row = row.to_dict()

    subject = _as_text(row.get("subject", ""))
    body = _as_text(row.get("body", ""))
    combined = (subject + " " + body).lower()
    letters = [c for c in (subject + body) if c.isalpha()]
    all_caps_ratio = (
        sum(1 for c in letters if c.isupper()) / len(letters) if letters else 0.0
    )
    length = max(len(subject + body), 1)
    exclamation_density = (subject + body).count("!") / length

    urls = _urls_from_row(row)
    hosts = [_host(u) for u in urls]
    tld_hits = 0
    for host in hosts:
        parts = host.rsplit(".", 1)
        if len(parts) == 2 and parts[1] in SUSPICIOUS_TLDS:
            tld_hits += 1

    domain = _sender_domain(row.get("sender", ""))
    hyphenated = 0.0
    if "-" in domain:
        for brand in BRAND_DOMAINS:
            if brand.replace(" ", "") in domain.replace("-", ""):
                hyphenated = 1.0
                break
        if not hyphenated and domain.count("-") >= 1 and len(domain) > 8:
            hyphenated = 1.0

    spf_fail = _auth_fail(row.get("spf", ""))
    dkim_fail = _auth_fail(row.get("dkim", ""))
    dmarc_fail = _auth_fail(row.get("dmarc", ""))

    try:
        n_attach = float(row.get("num_attachments", 0) or 0)
    except (TypeError, ValueError):
        n_attach = 0.0
    attach_types = _as_text(row.get("attachment_types", "")).lower().split()
    risky = 1.0 if any(ext.strip(".") in RISKY_EXTS for ext in attach_types) else 0.0

    urgency_hits = _count_phrases(combined, URGENCY_PHRASES)
    cred_hits = _count_phrases(combined, CREDENTIAL_PHRASES)

    return np.array([
        float(urgency_hits),
        1.0 if urgency_hits else 0.0,
        float(cred_hits),
        1.0 if cred_hits else 0.0,
        all_caps_ratio,
        exclamation_density,
        float(len(urls)),
        1.0 if any(IP_IN_URL_RE.search(u) for u in urls) else 0.0,
        float(tld_hits),
        1.0 if any(h in SHORTENER_HOSTS for h in hosts) else 0.0,
        _has_at_obfuscation(urls),
        float(max((u.count(".") for u in urls), default=0)),
        _lookalike_brand(domain),
        hyphenated,
        1.0 if re.search(r"\d", domain) else 0.0,
        spf_fail,
        dkim_fail,
        dmarc_fail,
        spf_fail + dkim_fail + dmarc_fail,
        n_attach,
        risky,
    ], dtype=float)


def triggered_reasons(fields: dict):
    """Plain-English reasons for whichever structured signals fired."""
    vec = structured_feature_row(fields)
    by_name = dict(zip(STRUCTURED_FEATURE_NAMES, vec))
    checks = [
        ("has_urgency", "Uses urgent / threatening language"),
        ("has_credential_request", "Asks for passwords or other credentials"),
        ("all_caps_ratio", "Uses unusually heavy ALL-CAPS in the text"),
        ("exclamation_density", "Uses a high density of exclamation marks"),
        ("has_ip_url", "Link uses a raw IP address instead of a domain"),
        ("suspicious_tld_count", "Link uses a suspicious top-level domain"),
        ("has_shortener", "Link is hidden behind a URL shortener"),
        ("has_at_obfuscation", "Link uses @ to obfuscate the real host"),
        ("max_dot_count", "Link contains an unusually high number of dots"),
        ("sender_lookalike", "Sender domain lookalike of a well-known brand"),
        ("hyphenated_domain", "Sender domain is a hyphenated brand lookalike"),
        ("digit_in_domain", "Sender domain uses digit substitution (e.g. paypa1)"),
        ("auth_fail_count", "Fails SPF/DKIM/DMARC authentication"),
        ("has_risky_attachment", "Includes a risky attachment type (.exe, .docm, …)"),
    ]
    reasons = []
    for key, sentence in checks:
        value = by_name[key]
        if key == "all_caps_ratio" and value < 0.35:
            continue
        if key == "exclamation_density" and value < 0.004:
            continue
        if key == "max_dot_count" and value < 4:
            continue
        if value > 0:
            reasons.append(sentence)
    return reasons


def fields_to_frame(fields: dict) -> pd.DataFrame:
    row = {
        "subject": _as_text(fields.get("subject", "")),
        "body": _as_text(fields.get("body", "")),
        "sender": _as_text(fields.get("sender", "")),
        "urls": _as_text(fields.get("urls", "")),
        "spf": _as_text(fields.get("spf", "")),
        "dkim": _as_text(fields.get("dkim", "")),
        "dmarc": _as_text(fields.get("dmarc", "")),
        "num_attachments": fields.get("num_attachments", 0) or 0,
        "attachment_types": _as_text(fields.get("attachment_types", "")),
    }
    return pd.DataFrame([row])


class FeatureBuilder:
    """TF-IDF on subject+body, concatenated with the structured signal vector."""

    def __init__(self, max_features=2000):
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=1,
            lowercase=True,
        )
        self.structured_names = list(STRUCTURED_FEATURE_NAMES)
        self._fitted = False

    def _combined_text(self, df: pd.DataFrame):
        subject = df["subject"].fillna("").astype(str) if "subject" in df else ""
        body = df["body"].fillna("").astype(str) if "body" in df else ""
        return subject + " " + body

    def fit(self, df: pd.DataFrame):
        self.tfidf.fit(self._combined_text(df))
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame):
        if not self._fitted:
            raise RuntimeError("FeatureBuilder.fit() must be called before transform().")
        text_x = self.tfidf.transform(self._combined_text(df))
        struct = np.vstack([structured_feature_row(row) for row in df.to_dict(orient="records")])
        return hstack([text_x, csr_matrix(struct)])

    def fit_transform(self, df: pd.DataFrame):
        return self.fit(df).transform(df)

    def transform_fields(self, fields: dict):
        return self.transform(fields_to_frame(fields))
