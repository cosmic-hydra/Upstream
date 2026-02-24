"""
Communication platform connectors for enterprise sales context.

Connectors for Gmail, Outlook, Slack, WhatsApp Business,
Zoom (call transcripts), and Microsoft Teams retrieve messages,
threads, and meeting transcripts and expose them as normalised
``Message`` objects for ingestion into the sales LLM's RAG pipeline.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Normalised message / communication record."""

    message_id: str
    platform: str  # "gmail" | "outlook" | "slack" | "whatsapp" | "zoom" | "teams"
    sender: str
    recipients: List[str]
    subject: str = ""
    body: str = ""
    timestamp: str = ""
    thread_id: Optional[str] = None
    attachments: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseCommunicationConnector(ABC):
    """Abstract base for all communication platform connectors."""

    platform: str = "base"

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self._authenticated = False

    @abstractmethod
    def authenticate(self) -> None:
        """Establish an authenticated session with the platform."""

    @abstractmethod
    def get_messages(
        self,
        limit: int = 100,
        query: Optional[str] = None,
    ) -> List[Message]:
        """Retrieve recent messages, optionally filtered by *query*."""


# ---------------------------------------------------------------------------
# Gmail
# ---------------------------------------------------------------------------


class GmailConnector(BaseCommunicationConnector):
    """
    Gmail connector via the Google API Python client.

    Expected config keys:
        credentials_file  – path to ``credentials.json`` (OAuth2 desktop app)
        token_file        – path to ``token.json`` (auto-refreshed)
        scopes            – list of Gmail scopes (optional)
    """

    platform = "gmail"
    _DEFAULT_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

    def authenticate(self) -> None:
        try:
            from google.oauth2.credentials import Credentials  # type: ignore
            from google.auth.transport.requests import Request  # type: ignore
            from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore
            from googleapiclient.discovery import build  # type: ignore
            import os

            scopes = self.config.get("scopes", self._DEFAULT_SCOPES)
            token_file = self.config.get("token_file", "gmail_token.json")
            creds = None
            if os.path.exists(token_file):
                creds = Credentials.from_authorized_user_file(token_file, scopes)
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.config["credentials_file"], scopes
                    )
                    creds = flow.run_local_server(port=0)
                with open(token_file, "w") as token:
                    token.write(creds.to_json())
            self._service = build("gmail", "v1", credentials=creds)
            self._authenticated = True
            logger.info("Gmail authentication successful.")
        except ImportError:
            raise RuntimeError(
                "google-api-python-client and google-auth-oauthlib are required "
                "for Gmail integration. Install with: "
                "pip install google-api-python-client google-auth-oauthlib"
            )

    def get_messages(
        self,
        limit: int = 100,
        query: Optional[str] = None,
    ) -> List[Message]:
        if not self._authenticated:
            self.authenticate()
        q = query or "in:inbox"
        result = (
            self._service.users()
            .messages()
            .list(userId="me", q=q, maxResults=limit)
            .execute()
        )
        messages = []
        for item in result.get("messages", []):
            msg = (
                self._service.users()
                .messages()
                .get(userId="me", id=item["id"], format="full")
                .execute()
            )
            headers = {
                h["name"].lower(): h["value"]
                for h in msg.get("payload", {}).get("headers", [])
            }
            body = ""
            parts = msg.get("payload", {}).get("parts", [])
            for part in parts:
                if part.get("mimeType") == "text/plain":
                    import base64

                    raw = part.get("body", {}).get("data", "")
                    if raw:
                        body = base64.urlsafe_b64decode(raw + "==").decode(
                            "utf-8", errors="replace"
                        )
                    break
            recipients = [
                r.strip()
                for r in (headers.get("to") or "").split(",")
                if r.strip()
            ]
            messages.append(
                Message(
                    message_id=item["id"],
                    platform="gmail",
                    sender=headers.get("from", ""),
                    recipients=recipients,
                    subject=headers.get("subject", ""),
                    body=body,
                    timestamp=headers.get("date", ""),
                    thread_id=msg.get("threadId"),
                )
            )
        return messages


# ---------------------------------------------------------------------------
# Outlook (Microsoft Graph)
# ---------------------------------------------------------------------------


class OutlookConnector(BaseCommunicationConnector):
    """
    Outlook connector via the Microsoft Graph REST API (requests + MSAL).

    Expected config keys:
        tenant_id, client_id, client_secret, user_email
    """

    platform = "outlook"
    _SCOPE = ["https://graph.microsoft.com/.default"]
    _GRAPH_BASE = "https://graph.microsoft.com/v1.0"

    def authenticate(self) -> None:
        try:
            import msal  # type: ignore
            import requests  # type: ignore

            self._requests = requests
            app = msal.ConfidentialClientApplication(
                client_id=self.config["client_id"],
                client_credential=self.config["client_secret"],
                authority=f"https://login.microsoftonline.com/{self.config['tenant_id']}",
            )
            token_result = app.acquire_token_for_client(scopes=self._SCOPE)
            if "access_token" not in token_result:
                raise RuntimeError(
                    f"MSAL authentication failed: {token_result.get('error_description')}"
                )
            self._access_token = token_result["access_token"]
            self._user_email = self.config["user_email"]
            self._authenticated = True
            logger.info("Outlook (Graph) authentication successful.")
        except ImportError:
            raise RuntimeError(
                "msal and requests are required for Outlook integration. "
                "Install with: pip install msal requests"
            )

    def get_messages(
        self,
        limit: int = 100,
        query: Optional[str] = None,
    ) -> List[Message]:
        if not self._authenticated:
            self.authenticate()
        url = f"{self._GRAPH_BASE}/users/{self._user_email}/messages"
        params: Dict[str, Any] = {"$top": limit, "$select": "id,subject,from,toRecipients,body,receivedDateTime,conversationId"}
        if query:
            params["$search"] = f'"{query}"'
        headers = {"Authorization": f"Bearer {self._access_token}"}
        resp = self._requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()
        messages = []
        for item in data.get("value", []):
            recipients = [
                r["emailAddress"]["address"]
                for r in item.get("toRecipients", [])
            ]
            messages.append(
                Message(
                    message_id=item["id"],
                    platform="outlook",
                    sender=item.get("from", {}).get("emailAddress", {}).get("address", ""),
                    recipients=recipients,
                    subject=item.get("subject", ""),
                    body=item.get("body", {}).get("content", ""),
                    timestamp=item.get("receivedDateTime", ""),
                    thread_id=item.get("conversationId"),
                )
            )
        return messages


# ---------------------------------------------------------------------------
# Slack
# ---------------------------------------------------------------------------


class SlackConnector(BaseCommunicationConnector):
    """
    Slack connector via the slack-sdk library.

    Expected config keys:
        bot_token   – Bot User OAuth Token (xoxb-...)
        channel_ids – list of channel IDs to ingest (optional)
    """

    platform = "slack"

    def authenticate(self) -> None:
        try:
            from slack_sdk import WebClient  # type: ignore
            from slack_sdk.errors import SlackApiError  # type: ignore

            self._client = WebClient(token=self.config["bot_token"])
            self._SlackApiError = SlackApiError
            # Verify credentials
            self._client.auth_test()
            self._authenticated = True
            logger.info("Slack authentication successful.")
        except ImportError:
            raise RuntimeError(
                "slack-sdk is required for Slack integration. "
                "Install with: pip install slack-sdk"
            )

    def get_messages(
        self,
        limit: int = 100,
        query: Optional[str] = None,
    ) -> List[Message]:
        if not self._authenticated:
            self.authenticate()
        channel_ids: List[str] = self.config.get("channel_ids", [])
        if not channel_ids:
            # Auto-discover public channels
            resp = self._client.conversations_list(types="public_channel", limit=200)
            channel_ids = [c["id"] for c in resp.get("channels", [])]
        messages: List[Message] = []
        per_channel = max(1, limit // max(len(channel_ids), 1))
        for cid in channel_ids:
            try:
                kwargs: Dict[str, Any] = {"channel": cid, "limit": per_channel}
                if query:
                    # Slack search requires a different API endpoint
                    search_resp = self._client.search_messages(
                        query=f"{query} in:<#{cid}>", count=per_channel
                    )
                    items = search_resp.get("messages", {}).get("matches", [])
                    for item in items:
                        messages.append(
                            Message(
                                message_id=item.get("ts", ""),
                                platform="slack",
                                sender=item.get("user", ""),
                                recipients=[],
                                body=item.get("text", ""),
                                timestamp=item.get("ts", ""),
                                thread_id=item.get("thread_ts"),
                                extra={"channel": cid},
                            )
                        )
                else:
                    hist = self._client.conversations_history(**kwargs)
                    for item in hist.get("messages", []):
                        messages.append(
                            Message(
                                message_id=item.get("ts", ""),
                                platform="slack",
                                sender=item.get("user", ""),
                                recipients=[],
                                body=item.get("text", ""),
                                timestamp=item.get("ts", ""),
                                thread_id=item.get("thread_ts"),
                                extra={"channel": cid},
                            )
                        )
            except self._SlackApiError as exc:
                logger.warning("Slack channel %s error: %s", cid, exc)
        return messages[:limit]


# ---------------------------------------------------------------------------
# WhatsApp Business
# ---------------------------------------------------------------------------


class WhatsAppBusinessConnector(BaseCommunicationConnector):
    """
    WhatsApp Business connector via the Meta Cloud API (requests).

    Expected config keys:
        access_token, phone_number_id, webhook_verify_token (optional)
    """

    platform = "whatsapp"
    _API_BASE = "https://graph.facebook.com/v19.0"

    def authenticate(self) -> None:
        try:
            import requests  # type: ignore

            self._requests = requests
            self._access_token = self.config["access_token"]
            self._phone_number_id = self.config["phone_number_id"]
            self._authenticated = True
            logger.info("WhatsApp Business authentication successful.")
        except ImportError:
            raise RuntimeError(
                "requests is required for WhatsApp integration. "
                "Install with: pip install requests"
            )

    def get_messages(
        self,
        limit: int = 100,
        query: Optional[str] = None,
    ) -> List[Message]:
        """
        Note: The WhatsApp Business Cloud API is push-based (webhooks).
        This method returns an empty list and logs a guidance message.
        Implement a webhook receiver to capture incoming messages in real time.
        """
        if not self._authenticated:
            self.authenticate()
        logger.info(
            "WhatsApp Business uses webhooks for incoming messages. "
            "Set up a webhook receiver at /webhook/whatsapp to ingest messages."
        )
        return []


# ---------------------------------------------------------------------------
# Zoom (meeting transcripts)
# ---------------------------------------------------------------------------


class ZoomConnector(BaseCommunicationConnector):
    """
    Zoom connector for meeting recordings and AI summaries via the Zoom API v2.

    Expected config keys:
        account_id, client_id, client_secret
    """

    platform = "zoom"
    _TOKEN_URL = "https://zoom.us/oauth/token"
    _API_BASE = "https://api.zoom.us/v2"

    def authenticate(self) -> None:
        try:
            import requests  # type: ignore
            from base64 import b64encode

            self._requests = requests
            credentials = b64encode(
                f"{self.config['client_id']}:{self.config['client_secret']}".encode()
            ).decode()
            resp = requests.post(
                self._TOKEN_URL,
                headers={"Authorization": f"Basic {credentials}"},
                params={
                    "grant_type": "account_credentials",
                    "account_id": self.config["account_id"],
                },
            )
            resp.raise_for_status()
            self._access_token = resp.json()["access_token"]
            self._authenticated = True
            logger.info("Zoom authentication successful.")
        except ImportError:
            raise RuntimeError(
                "requests is required for Zoom integration. "
                "Install with: pip install requests"
            )

    def get_messages(
        self,
        limit: int = 100,
        query: Optional[str] = None,
    ) -> List[Message]:
        """Returns Zoom cloud recordings as Message objects (transcript as body)."""
        if not self._authenticated:
            self.authenticate()
        headers = {"Authorization": f"Bearer {self._access_token}"}
        resp = self._requests.get(
            f"{self._API_BASE}/users/me/recordings",
            headers=headers,
            params={"page_size": min(limit, 300)},
        )
        resp.raise_for_status()
        data = resp.json()
        messages = []
        for meeting in data.get("meetings", []):
            transcript_url = ""
            for rf in meeting.get("recording_files", []):
                if rf.get("file_type") == "TRANSCRIPT":
                    transcript_url = rf.get("download_url", "")
                    break
            body = ""
            if transcript_url:
                try:
                    tr = self._requests.get(
                        transcript_url,
                        headers=headers,
                        params={"access_token": self._access_token},
                    )
                    body = tr.text
                except Exception as exc:
                    logger.warning("Zoom transcript download error: %s", exc)
            messages.append(
                Message(
                    message_id=meeting.get("uuid", meeting.get("id", "")),
                    platform="zoom",
                    sender=meeting.get("host_email", ""),
                    recipients=[],
                    subject=meeting.get("topic", ""),
                    body=body,
                    timestamp=meeting.get("start_time", ""),
                    extra={"duration": meeting.get("duration")},
                )
            )
        return messages[:limit]


# ---------------------------------------------------------------------------
# Microsoft Teams
# ---------------------------------------------------------------------------


class TeamsConnector(BaseCommunicationConnector):
    """
    Microsoft Teams connector via the Microsoft Graph API (MSAL + requests).

    Expected config keys:
        tenant_id, client_id, client_secret, user_email
    """

    platform = "teams"
    _SCOPE = ["https://graph.microsoft.com/.default"]
    _GRAPH_BASE = "https://graph.microsoft.com/v1.0"

    def authenticate(self) -> None:
        try:
            import msal  # type: ignore
            import requests  # type: ignore

            self._requests = requests
            app = msal.ConfidentialClientApplication(
                client_id=self.config["client_id"],
                client_credential=self.config["client_secret"],
                authority=f"https://login.microsoftonline.com/{self.config['tenant_id']}",
            )
            token_result = app.acquire_token_for_client(scopes=self._SCOPE)
            if "access_token" not in token_result:
                raise RuntimeError(
                    f"Teams/MSAL authentication failed: {token_result.get('error_description')}"
                )
            self._access_token = token_result["access_token"]
            self._user_email = self.config["user_email"]
            self._authenticated = True
            logger.info("Microsoft Teams authentication successful.")
        except ImportError:
            raise RuntimeError(
                "msal and requests are required for Teams integration. "
                "Install with: pip install msal requests"
            )

    def get_messages(
        self,
        limit: int = 100,
        query: Optional[str] = None,
    ) -> List[Message]:
        if not self._authenticated:
            self.authenticate()
        headers = {"Authorization": f"Bearer {self._access_token}"}
        # List joined teams for the user
        teams_resp = self._requests.get(
            f"{self._GRAPH_BASE}/users/{self._user_email}/joinedTeams",
            headers=headers,
        )
        teams_resp.raise_for_status()
        teams = teams_resp.json().get("value", [])
        messages: List[Message] = []
        per_team = max(1, limit // max(len(teams), 1))
        for team in teams:
            team_id = team["id"]
            ch_resp = self._requests.get(
                f"{self._GRAPH_BASE}/teams/{team_id}/channels",
                headers=headers,
            )
            if ch_resp.status_code != 200:
                continue
            channels = ch_resp.json().get("value", [])
            for channel in channels:
                ch_id = channel["id"]
                msg_resp = self._requests.get(
                    f"{self._GRAPH_BASE}/teams/{team_id}/channels/{ch_id}/messages",
                    headers=headers,
                    params={"$top": per_team},
                )
                if msg_resp.status_code != 200:
                    continue
                for item in msg_resp.json().get("value", []):
                    body_content = item.get("body", {}).get("content", "")
                    sender_addr = (
                        item.get("from", {})
                        .get("user", {})
                        .get("userPrincipalName", "")
                    )
                    messages.append(
                        Message(
                            message_id=item["id"],
                            platform="teams",
                            sender=sender_addr,
                            recipients=[],
                            body=body_content,
                            timestamp=item.get("createdDateTime", ""),
                            thread_id=item.get("replyToId"),
                            extra={"team_id": team_id, "channel_id": ch_id},
                        )
                    )
        return messages[:limit]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_CONNECTOR_REGISTRY: Dict[str, type] = {
    "gmail": GmailConnector,
    "outlook": OutlookConnector,
    "slack": SlackConnector,
    "whatsapp": WhatsAppBusinessConnector,
    "zoom": ZoomConnector,
    "teams": TeamsConnector,
}


def get_communication_connector(
    name: str, config: Dict[str, Any]
) -> BaseCommunicationConnector:
    """Return an instantiated communication connector by name.

    Args:
        name: One of ``"gmail"``, ``"outlook"``, ``"slack"``,
              ``"whatsapp"``, ``"zoom"``, ``"teams"``.
        config: Platform-specific authentication configuration.

    Raises:
        ValueError: If *name* is not a registered connector.
    """
    name = name.lower()
    if name not in _CONNECTOR_REGISTRY:
        raise ValueError(
            f"Unknown communication connector '{name}'. "
            f"Available: {list(_CONNECTOR_REGISTRY)}"
        )
    return _CONNECTOR_REGISTRY[name](config)
