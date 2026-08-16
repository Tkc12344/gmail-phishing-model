"""
gmail_auth.py

Handles OAuth2 login to a Gmail account so the scanner can read messages and
apply labels. Run once interactively to generate token.json; after that it
refreshes automatically.

SETUP (one-time, in Google Cloud Console):
  1. Create/select a project at https://console.cloud.google.com
  2. Enable the "Gmail API" for that project.
  3. Go to "APIs & Services" > "Credentials" > "Create Credentials"
     > "OAuth client ID" > Application type: "Desktop app".
  4. Download the JSON and save it as credentials.json in this project's
     root folder. Do NOT commit this file or the generated token.json.
  5. On first run, a browser window will open asking you to log in and
     grant access to the scopes below.

SCOPES: we request the minimum needed —
  - gmail.readonly   : read message content/headers
  - gmail.modify     : apply/remove labels (needed for quarantine actions)
We deliberately do NOT request gmail.send or full mailbox delete access.
"""

from pathlib import Path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
]

ROOT = Path(__file__).resolve().parent.parent
CREDENTIALS_FILE = ROOT / "credentials.json"
TOKEN_FILE = ROOT / "token.json"


def get_credentials():
    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDENTIALS_FILE.exists():
                raise FileNotFoundError(
                    f"Missing {CREDENTIALS_FILE}. See the setup instructions "
                    "at the top of gmail_auth.py."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_FILE), SCOPES
            )
            creds = flow.run_local_server(port=0)
        TOKEN_FILE.write_text(creds.to_json())

    return creds
