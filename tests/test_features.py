import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from feature_engineering import (  # noqa: E402
    STRUCTURED_FEATURE_NAMES,
    extract_urls,
    normalize_email_frame,
    structured_feature_row,
    structured_risk_boost,
    triggered_reasons,
)
from rules import rule_hits  # noqa: E402
import pandas as pd  # noqa: E402


class FeatureTests(unittest.TestCase):
    def test_extract_urls(self):
        text = "See http://evil.xyz/a and https://ok.com/b"
        urls = extract_urls(text)
        self.assertEqual(len(urls), 2)

    def test_ip_url_and_lookalike(self):
        row = {
            "subject": "Verify now",
            "body": "Enter your password at http://192.168.1.50/paypal",
            "sender": "security@paypa1.com",
            "urls": "http://192.168.1.50/paypal",
            "spf": "fail",
            "dkim": "fail",
            "dmarc": "fail",
        }
        flags = dict(zip(STRUCTURED_FEATURE_NAMES, structured_feature_row(row)))
        self.assertEqual(flags["has_ip_url"], 1.0)
        self.assertEqual(flags["sender_lookalike"], 1.0)
        self.assertGreater(flags["auth_fail_count"], 0)
        reasons = triggered_reasons(row)
        self.assertTrue(any("IP address" in r for r in reasons))

    def test_display_name_spoof(self):
        row = {
            "subject": "Account",
            "body": "Please sign in",
            "sender": "PayPal Security <help@random-host.xyz>",
            "spf": "fail",
        }
        flags = dict(zip(STRUCTURED_FEATURE_NAMES, structured_feature_row(row)))
        self.assertEqual(flags["display_name_spoof"], 1.0)

    def test_keyword_leak_stripped(self):
        df = pd.DataFrame({
            "text": ["Hello\n\nKeywords: password otp\n\nBye"],
            "label": [1],
        })
        out = normalize_email_frame(df)
        self.assertNotIn("Keywords", out.loc[0, "body"])

    def test_legit_google_has_no_boost(self):
        row = {
            "subject": "Security alert from Google",
            "body": "New sign-in. Details: https://myaccount.google.com/notifications",
            "sender": "no-reply@accounts.google.com",
            "urls": "https://myaccount.google.com/notifications",
            "spf": "pass",
            "dkim": "pass",
            "dmarc": "pass",
        }
        self.assertEqual(structured_risk_boost(row), 0.0)
        self.assertEqual(rule_hits(row), [])

    def test_smoking_gun_rule(self):
        row = {
            "subject": "Verify",
            "body": "Enter your password at http://10.0.0.8/login",
            "sender": "it@evil.xyz",
            "urls": "http://10.0.0.8/login",
            "spf": "fail",
        }
        hits = rule_hits(row)
        self.assertTrue(any("credential request plus a raw-IP" in h for h in hits))


if __name__ == "__main__":
    unittest.main()
