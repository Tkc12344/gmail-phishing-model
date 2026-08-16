"""
Build a structured training CSV with URL / sender / auth / attachment fields.

The emails are synthetic templates for defensive model training — not a
phishing kit. Run:

    python src/generate_dataset.py
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "emails.csv"

FIRST = ["Avery", "Jordan", "Sam", "Riley", "Taylor", "Casey", "Quinn", "Morgan"]
LAST = ["Kim", "Patel", "Garcia", "Nguyen", "Brooks", "Okoye", "Singh", "Walsh"]
TEAMS = ["engineering", "finance", "design", "ops", "sales"]
COMPANIES = ["acme.com", "northwind.dev", "harborlabs.io", "brightline.co"]

COLUMNS = [
    "subject", "body", "sender", "reply_to", "urls",
    "spf", "dkim", "dmarc", "num_attachments", "attachment_types",
    "phishing_type", "label",
]


def _name():
    return f"{random.choice(FIRST)} {random.choice(LAST)}"


def _row(**kwargs):
    base = {k: "" for k in COLUMNS}
    base["num_attachments"] = 0
    base["attachment_types"] = ""
    base["reply_to"] = kwargs.get("sender", "")
    base.update(kwargs)
    return base


def phishing_rows():
    rows = []

    rows.append(_row(
        subject="URGENT: Your PayPal account will be suspended",
        body="We detected unusual activity. Verify your password immediately or your account will be closed. Click http://192.168.1.50/paypal to confirm your identity.",
        sender="PayPal Security <security@paypa1.com>",
        reply_to="verify@accounts-secure.xyz",
        urls="http://192.168.1.50/paypal",
        spf="fail", dkim="fail", dmarc="fail",
        phishing_type="credential_harvesting", label="phishing",
    ))
    rows.append(_row(
        subject="Your Microsoft password expires today",
        body="Sign in and confirm your credentials now at http://microsoft-secure.xyz/login or your mailbox will be disabled.",
        sender="account@microsoft-secure.xyz",
        urls="http://microsoft-secure.xyz/login",
        spf="fail", dkim="fail", dmarc="none",
        phishing_type="credential_harvesting", label="phishing",
    ))
    rows.append(_row(
        subject="Final notice: bank account locked",
        body="Unauthorized transfers detected. Act now and verify your bank account at http://bit.ly/bank-verify within 24 hours.",
        sender="alerts@bankofamerica-verify.tk",
        urls="http://bit.ly/bank-verify",
        spf="fail", dkim="softfail", dmarc="fail",
        phishing_type="financial_scam", label="phishing",
    ))
    rows.append(_row(
        subject="Netflix: update your payment or lose access",
        body="Your membership will be terminated. Enter your credit card at http://netflix-update.top/billing immediately.",
        sender="billing@netflix-update.top",
        urls="http://netflix-update.top/billing",
        spf="fail", dkim="fail", dmarc="fail",
        phishing_type="financial_scam", label="phishing",
    ))
    rows.append(_row(
        subject="Apple ID locked!!!",
        body="Your Apple ID has been locked. Click http://apple.com.secure-id.gq/unlock and enter your password right now.",
        sender="support@apple.com.secure-id.gq",
        urls="http://apple.com.secure-id.gq/unlock",
        spf="fail", dkim="none", dmarc="fail",
        phishing_type="credential_harvesting", label="phishing",
    ))
    rows.append(_row(
        subject="Amazon invoice attached",
        body="Open the attached invoice.exe to review a pending charge. Confirm your password if prompted.",
        sender="invoices@amaz0n.com",
        urls="http://amaz0n-billing.click/invoice",
        spf="fail", dkim="fail", dmarc="fail",
        num_attachments=1, attachment_types="exe",
        phishing_type="malware_attachment", label="phishing",
    ))
    rows.append(_row(
        subject="Google Docs: document shared with you",
        body="View the file at http://google.com@evil-share.xyz/doc and sign in with your username and password.",
        sender="sharing@accounts-google.ml",
        urls="http://google.com@evil-share.xyz/doc",
        spf="fail", dkim="fail", dmarc="fail",
        phishing_type="credential_harvesting", label="phishing",
    ))
    rows.append(_row(
        subject="YOU HAVE WON $1,000,000!!!",
        body="CLICK HERE NOW to claim your prize. Send your login and bank account details to receive the transfer. http://prize-claim.icu/win",
        sender="winner@prize-claim.icu",
        urls="http://prize-claim.icu/win",
        spf="none", dkim="none", dmarc="none",
        phishing_type="prize", label="phishing",
    ))
    rows.append(_row(
        subject="IT: malware detected on your workstation",
        body="The help desk found a virus. Install the security patch from http://desk-support.work/patch and enter your domain password.",
        sender="helpdesk@it-support-desk.work",
        urls="http://desk-support.work/patch",
        spf="fail", dkim="fail", dmarc="fail",
        num_attachments=1, attachment_types="js",
        phishing_type="tech_support", label="phishing",
    ))
    rows.append(_row(
        subject="Wire needed for the children's hospital gala",
        body="We talked at the conference. I am stranded and the hotel needs payment verification. Can you send a temporary loan today? I will repay next week.",
        sender="Alex Khan <traveler@mail-relay.gq>",
        reply_to="urgent-help@proton-temp.xyz",
        urls="",
        spf="fail", dkim="none", dmarc="none",
        phishing_type="romance_advance_fee", label="phishing",
    ))
    rows.append(_row(
        subject="Updated payroll file — action required",
        body="Please open payroll-update.docm and enable macros so we can process this week's payments. Failure to respond may delay deposits.",
        sender="payroll@acme-hr-secure.click",
        urls="http://acme-hr-secure.click/payroll",
        spf="fail", dkim="fail", dmarc="fail",
        num_attachments=1, attachment_types="docm",
        phishing_type="malware_attachment", label="phishing",
    ))
    rows.append(_row(
        subject="GitHub: new SSH key added",
        body="A new SSH key was added to your account. If this was not you, sign in at http://github-security.xyz/sessions and reset your password immediately.",
        sender="GitHub <noreply@githvb.com>",
        urls="http://github-security.xyz/sessions",
        spf="fail", dkim="fail", dmarc="fail",
        phishing_type="credential_harvesting", label="phishing",
    ))
    rows.append(_row(
        subject="IRS notice CP14 — balance due",
        body="Your case will be referred for collection. Pay the amount due and confirm your social security and bank account at http://irs-payments.top/cp14.",
        sender="notices@irs-treasury.top",
        urls="http://irs-payments.top/cp14",
        spf="fail", dkim="fail", dmarc="fail",
        phishing_type="authority_scam", label="phishing",
    ))
    rows.append(_row(
        subject="Shared OneDrive folder",
        body="You have a new encrypted folder. Open http://login.microsoftonline.com.session-verify.tk/office to view it.",
        sender="sharepoint@office365-mail.tk",
        urls="http://login.microsoftonline.com.session-verify.tk/office",
        spf="fail", dkim="none", dmarc="fail",
        phishing_type="credential_harvesting", label="phishing",
    ))
    rows.append(_row(
        subject="Delivery exception — extra customs fee",
        body="Your package is on hold. Pay the customs fee with a credit card at http://parcel-hold.icu/pay to avoid return to sender.",
        sender="tracking@dhl-express-pay.icu",
        urls="http://parcel-hold.icu/pay",
        spf="fail", dkim="fail", dmarc="none",
        phishing_type="financial_scam", label="phishing",
    ))
    rows.append(_row(
        subject="Reset your iCloud password",
        body="We received a request to reset your Apple ID. Confirm at http://xn--pple-43d.com/unlock using your current password and one-time code.",
        sender="appleid@icloud-verify.xyz",
        urls="http://xn--pple-43d.com/unlock",
        spf="fail", dkim="fail", dmarc="fail",
        phishing_type="credential_harvesting", label="phishing",
    ))

    # Variations so TF-IDF sees more than one wording per pattern.
    brands = [
        ("PayPal", "paypal", "paypa1-secure.xyz", "account will be limited"),
        ("Amazon", "amazon", "amaz0n-help.click", "order cannot be shipped"),
        ("Microsoft 365", "microsoft", "office365-alert.top", "mailbox will be disabled"),
        ("Chase", "chase", "chase-verify.tk", "card will be frozen"),
        ("Netflix", "netflix", "netflix-billing.icu", "stream will be interrupted"),
    ]
    for brand, key, host, threat in brands:
        rows.append(_row(
            subject=f"{brand} security alert",
            body=f"Unusual sign-in on your {brand} account. {threat.capitalize()}. Verify your username and password at http://{host}/login within 24 hours.",
            sender=f"{brand} Alerts <no-reply@{host}>",
            urls=f"http://{host}/login",
            spf="fail", dkim="fail", dmarc="fail",
            phishing_type="credential_harvesting", label="phishing",
        ))
        rows.append(_row(
            subject=f"Invoice from {brand}",
            body=f"Please review the attached statement and confirm payment details. If the charge looks wrong, sign in at http://bit.ly/{key}-bill.",
            sender=f"billing@{key}-statements.work",
            urls=f"http://bit.ly/{key}-bill",
            spf="softfail", dkim="fail", dmarc="none",
            num_attachments=1, attachment_types="html",
            phishing_type="financial_scam", label="phishing",
        ))

    for i in range(1, 21):
        ip = f"10.4.{i}.20"
        rows.append(_row(
            subject=f"Mailbox quota exceeded ({i})",
            body=f"Your inbox is over quota. Click here to keep receiving mail and enter your password: http://{ip}/quota/login",
            sender=f"mailadmin@quota-fix{i}.xyz",
            urls=f"http://{ip}/quota/login",
            spf="fail", dkim="none", dmarc="fail",
            phishing_type="credential_harvesting", label="phishing",
        ))

    for i in range(1, 16):
        rows.append(_row(
            subject=f"Vendor payment {1800 + i}",
            body=f"Bank details changed this morning. Wire the outstanding balance using the new account in the attached spreadsheet. Do not use the old routing number.",
            sender=f"accounts@vendor-pay{i}.top",
            reply_to=f"wires@offbook{i}.gq",
            urls=f"http://vendor-pay{i}.top/wire",
            spf="fail", dkim="fail", dmarc="fail",
            num_attachments=1, attachment_types="xlsm",
            phishing_type="invoice_fraud", label="phishing",
        ))

    return rows


def legit_rows():
    rows = []
    rows.append(_row(
        subject="Notes from today's standup",
        body="Here are the action items we agreed on: finish the API review, ping design about the empty state, and ship the copy tweak tomorrow.",
        sender="jordan@acme.com",
        urls="https://acme.com/wiki/standup",
        spf="pass", dkim="pass", dmarc="pass",
        phishing_type="legitimate", label="legit",
    ))
    rows.append(_row(
        subject="New sign-in on GitHub",
        body="A new sign-in from Chrome on macOS was recorded. If this was you, no action is needed. Review sessions at https://github.com/settings/sessions",
        sender="GitHub <noreply@github.com>",
        urls="https://github.com/settings/sessions",
        spf="pass", dkim="pass", dmarc="pass",
        phishing_type="legitimate", label="legit",
    ))
    rows.append(_row(
        subject="Weekly product newsletter",
        body="Three launches this week: usage analytics, the new export CSV, and a quieter notification default. Read more at https://news.example.com/week-32",
        sender="hello@news.example.com",
        urls="https://news.example.com/week-32",
        spf="pass", dkim="pass", dmarc="pass",
        phishing_type="legitimate", label="legit",
    ))
    rows.append(_row(
        subject="Calendar: design critique Thursday 3pm",
        body="Invited: you, Priya, and Sam. Location: Room 4B. Add a comment on the Figma file beforehand if you can.",
        sender="calendar@acme.com",
        urls="https://calendar.acme.com/event/8841",
        spf="pass", dkim="pass", dmarc="pass",
        phishing_type="legitimate", label="legit",
    ))
    rows.append(_row(
        subject="Your Amazon.com order of USB-C hub",
        body="Thanks for your order. It will arrive Thursday. Track it at https://www.amazon.com/gp/your-account/order-history",
        sender="auto-confirm@amazon.com",
        urls="https://www.amazon.com/gp/your-account/order-history",
        spf="pass", dkim="pass", dmarc="pass",
        phishing_type="legitimate", label="legit",
    ))
    rows.append(_row(
        subject="Security alert from Google",
        body="We noticed a new sign-in to your account. If this was you, you can ignore this message. Details: https://myaccount.google.com/notifications",
        sender="Google <no-reply@accounts.google.com>",
        urls="https://myaccount.google.com/notifications",
        spf="pass", dkim="pass", dmarc="pass",
        phishing_type="legitimate", label="legit",
    ))
    rows.append(_row(
        subject="Q3 budget spreadsheet",
        body="Attached is the latest budget workbook as a PDF. Please add comments before Friday's planning meeting.",
        sender="finance@acme.com",
        urls="",
        spf="pass", dkim="pass", dmarc="pass",
        num_attachments=1, attachment_types="pdf",
        phishing_type="legitimate", label="legit",
    ))
    rows.append(_row(
        subject="Invoice 1842 from Acme Supplies",
        body="Please find invoice 1842 for April office supplies. Payment terms net 30. Contact billing@acme-supplies.com with questions.",
        sender="vendor@acme-supplies.com",
        urls="https://acme-supplies.com/invoices/1842",
        spf="pass", dkim="pass", dmarc="pass",
        num_attachments=1, attachment_types="pdf",
        phishing_type="legitimate", label="legit",
    ))
    rows.append(_row(
        subject="Reset your password",
        body="You requested a password reset. Use this official link if it was you. The link expires in 30 minutes: https://accounts.google.com/signin/recovery",
        sender="no-reply@accounts.google.com",
        urls="https://accounts.google.com/signin/recovery",
        spf="pass", dkim="pass", dmarc="pass",
        phishing_type="legitimate", label="legit",
    ))
    rows.append(_row(
        subject="Your Apple ID was used to sign in",
        body="If this was you, no further action is needed. Manage devices at https://appleid.apple.com. We will never ask for your password by email.",
        sender="Apple <appleid@id.apple.com>",
        urls="https://appleid.apple.com",
        spf="pass", dkim="pass", dmarc="pass",
        phishing_type="legitimate", label="legit",
    ))
    rows.append(_row(
        subject="PayPal: you sent a payment",
        body="You sent $42.00 to Harbor Labs. View the transaction at https://www.paypal.com/activity",
        sender="service@paypal.com",
        urls="https://www.paypal.com/activity",
        spf="pass", dkim="pass", dmarc="pass",
        phishing_type="legitimate", label="legit",
    ))
    rows.append(_row(
        subject="Microsoft account security code",
        body="Your security code is 184293. If you did not request this, you can ignore the email. Do not forward the code.",
        sender="account-security-noreply@accountprotection.microsoft.com",
        urls="https://account.microsoft.com/security",
        spf="pass", dkim="pass", dmarc="pass",
        phishing_type="legitimate", label="legit",
    ))

    for i in range(1, 31):
        person = _name()
        company = random.choice(COMPANIES)
        team = random.choice(TEAMS)
        first = person.split()[0].lower()
        rows.append(_row(
            subject=f"{team.title()} sync notes {i}",
            body=f"Hi team — {person} captured notes from the {team} sync. Please review the ticket list and add estimates before Thursday. Doc: https://{company}/docs/sync-{i}",
            sender=f"{first}@{company}",
            urls=f"https://{company}/docs/sync-{i}",
            spf="pass", dkim="pass", dmarc="pass",
            phishing_type="legitimate", label="legit",
        ))

    for i in range(1, 16):
        rows.append(_row(
            subject=f"Receipt for order #{10000 + i}",
            body=f"Thanks for your purchase. Your receipt is attached as a PDF. Track shipping at https://www.amazon.com/gp/your-account/order-history?order={10000 + i}",
            sender="auto-confirm@amazon.com",
            urls=f"https://www.amazon.com/gp/your-account/order-history?order={10000 + i}",
            spf="pass", dkim="pass", dmarc="pass",
            num_attachments=1, attachment_types="pdf",
            phishing_type="legitimate", label="legit",
        ))

    for i in range(1, 12):
        rows.append(_row(
            subject=f"Password change confirmation ({i})",
            body="Your password was changed successfully. If you did not do this, visit https://myaccount.google.com/security and sign in from a trusted device.",
            sender="no-reply@accounts.google.com",
            urls="https://myaccount.google.com/security",
            spf="pass", dkim="pass", dmarc="pass",
            phishing_type="legitimate", label="legit",
        ))

    return rows


def main():
    random.seed(42)
    rows = phishing_rows() + legit_rows()
    random.shuffle(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    n_phish = sum(1 for r in rows if r["label"] == "phishing")
    print(f"Wrote {len(rows)} rows to {OUT}  (phishing={n_phish} legit={len(rows) - n_phish})")


if __name__ == "__main__":
    main()
