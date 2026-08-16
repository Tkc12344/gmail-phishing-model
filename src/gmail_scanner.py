"""
gmail_scanner.py

Connects to Gmail, scores inbox messages with the trained model, and takes
tiered protective action.

DEFAULT MODE IS DRY RUN. Nothing is modified until you pass --live.

Actions taken (in --live mode):
  risk >= 70%  -> label "Phishing/High-Risk", remove from INBOX (archive)
  30% <= risk < 70% -> label "Phishing/Suspicious", left in inbox
  risk < 30%   -> no action

Nothing is ever deleted. OAuth scopes do not allow sending mail.

Usage:
    python src/gmail_scanner.py --max_results 20 --dry-run
    python src/gmail_scanner.py --max_results 20 --live
    python src/gmail_scanner.py --query "in:inbox is:unread" --dry-run
"""

import argparse
import base64
import json
import re

from googleapiclient.discovery import build

from feature_engineering import extract_urls
from gmail_auth import get_credentials
from predict import load_pipeline, score_email

HIGH_RISK_LABEL = "Phishing/High-Risk"
SUSPICIOUS_LABEL = "Phishing/Suspicious"


def get_service():
    creds = get_credentials()
    return build("gmail", "v1", credentials=creds)


def find_label(service, name):
    labels = service.users().labels().list(userId="me").execute().get("labels", [])
    for lbl in labels:
        if lbl["name"] == name:
            return lbl["id"]
    return None


def ensure_label(service, name):
    existing = find_label(service, name)
    if existing:
        return existing
    created = service.users().labels().create(
        userId="me",
        body={
            "name": name,
            "labelListVisibility": "labelShow",
            "messageListVisibility": "show",
        },
    ).execute()
    return created["id"]


def _get_header(headers, name):
    for h in headers:
        if h["name"].lower() == name.lower():
            return h["value"]
    return ""


def _parse_auth_results(auth_header: str):
    """Pulls spf=/dkim=/dmarc= pass/fail out of Authentication-Results."""
    result = {"spf": "", "dkim": "", "dmarc": ""}
    for key in result:
        match = re.search(rf"{key}=(\w+)", auth_header, re.IGNORECASE)
        if match:
            result[key] = match.group(1).lower()
    return result


def _decode_body(payload):
    """Walk MIME parts; prefer text/plain, fall back to stripped text/html."""
    def walk(part):
        mime = part.get("mimeType", "")
        data = part.get("body", {}).get("data")
        if mime == "text/plain" and data:
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
        for sub in part.get("parts", []) or []:
            found = walk(sub)
            if found:
                return found
        if mime == "text/html" and data:
            html = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
            return re.sub("<[^<]+?>", " ", html)
        return None

    return walk(payload) or ""


def fetch_message_fields(service, msg_id):
    msg = service.users().messages().get(
        userId="me", id=msg_id, format="full"
    ).execute()
    headers = msg["payload"].get("headers", [])

    subject = _get_header(headers, "Subject")
    sender = _get_header(headers, "From")
    reply_to = _get_header(headers, "Reply-To")
    auth_results = _get_header(headers, "Authentication-Results")
    auth = _parse_auth_results(auth_results)
    body = _decode_body(msg["payload"])
    urls = " ".join(extract_urls(body))

    attachments = []

    def collect_attachments(part):
        filename = part.get("filename", "")
        if filename:
            attachments.append(filename.split(".")[-1].lower())
        for sub in part.get("parts", []) or []:
            collect_attachments(sub)

    collect_attachments(msg["payload"])

    return {
        "subject": subject,
        "body": body,
        "sender": sender,
        "reply_to": reply_to,
        "urls": urls,
        "spf": auth.get("spf", ""),
        "dkim": auth.get("dkim", ""),
        "dmarc": auth.get("dmarc", ""),
        "num_attachments": len(attachments),
        "attachment_types": " ".join(attachments),
    }, msg


def apply_action(service, msg_id, risk_pct, dry_run, label_ids):
    if risk_pct >= 70:
        tier, label_id = "HIGH RISK", label_ids[HIGH_RISK_LABEL]
        body = {"addLabelIds": [label_id], "removeLabelIds": ["INBOX"]}
    elif risk_pct >= 30:
        tier, label_id = "SUSPICIOUS", label_ids[SUSPICIOUS_LABEL]
        body = {"addLabelIds": [label_id]}
    else:
        return "SAFE", None

    if not dry_run:
        service.users().messages().modify(userId="me", id=msg_id, body=body).execute()
    return tier, body


def _already_labeled(msg, known_ids) -> bool:
    if not known_ids:
        return False
    return bool(set(msg.get("labelIds", [])) & known_ids)


def scan_inbox(max_results=20, dry_run=True, query="in:inbox", as_json=False, skip_labeled=False):
    service = get_service()
    model, fb = load_pipeline()

    label_ids = {}
    if not dry_run:
        label_ids = {
            HIGH_RISK_LABEL: ensure_label(service, HIGH_RISK_LABEL),
            SUSPICIOUS_LABEL: ensure_label(service, SUSPICIOUS_LABEL),
        }
    existing_custom = {
        find_label(service, HIGH_RISK_LABEL),
        find_label(service, SUSPICIOUS_LABEL),
    }
    existing_custom.discard(None)

    resp = service.users().messages().list(
        userId="me", q=query, maxResults=max_results
    ).execute()
    messages = resp.get("messages", [])
    mode = "DRY RUN — no changes will be made" if dry_run else "LIVE MODE"
    if not as_json:
        print(f"Scanning {len(messages)} messages ({mode})\n")

    summary = {"SAFE": 0, "SUSPICIOUS": 0, "HIGH RISK": 0}
    rows = []

    for item in messages:
        fields, raw_msg = fetch_message_fields(service, item["id"])
        if skip_labeled and _already_labeled(raw_msg, existing_custom):
            continue
        result = score_email(fields, model=model, fb=fb)
        if dry_run:
            tier = result["band"]
            action = None
        else:
            tier, action = apply_action(
                service, item["id"], result["risk_pct"], dry_run, label_ids
            )
        summary[tier] += 1
        row = {
            "id": item["id"],
            "tier": tier,
            "risk_pct": result["risk_pct"],
            "sender": fields["sender"],
            "subject": fields["subject"],
            "reasons": result["reasons"],
            "action": action,
        }
        rows.append(row)
        if not as_json:
            print(
                f"[{tier:10}] {result['risk_pct']:5.1f}%  "
                f"from={fields['sender'][:40]:<40} subj={fields['subject'][:50]}"
            )
            if tier != "SAFE":
                for reason in result["reasons"]:
                    print(f"              ✓ {reason}")

    if as_json:
        print(json.dumps({"mode": mode, "summary": summary, "messages": rows}, indent=2))
    else:
        print("\n--- Summary ---")
        for tier, count in summary.items():
            print(f"{tier}: {count}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_results", type=int, default=20)
    parser.add_argument(
        "--query",
        default="in:inbox",
        help="Gmail search query, e.g. 'in:inbox is:unread'",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--skip-labeled",
        action="store_true",
        help="Skip messages that already have a Phishing/* label id in labelIds",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="(default) Score and print only, no mailbox changes",
    )
    mode.add_argument(
        "--live",
        action="store_true",
        help="Actually apply labels / archive high-risk mail",
    )
    args = parser.parse_args()

    scan_inbox(
        max_results=args.max_results,
        dry_run=not args.live,
        query=args.query,
        as_json=args.json,
        skip_labeled=args.skip_labeled,
    )
