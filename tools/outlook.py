"""Outlook mail and calendar through Microsoft Graph: sign-in, tokens, and read-only calls.

Sign-in is the OAuth device code flow (MSAL): Orion prints a code, you enter it
at microsoft.com/devicelogin on any browser. Nothing listens on a port.

Permissions are read-only (``SCOPES``). Every request is a GET, so reading an
email doesn't even mark it as read.

Tokens: one MSAL cache per Orion person, kept together in ``data/outlook.enc``
(AES-256-GCM with the data key). The wipe shreds that file first (sign-out hook).
A read-only app can't revoke its refresh token at Microsoft; removing the app in
the account's consent page does (see the user manual).

Graph content (subjects, senders, bodies, event titles) is returned to the
caller and never logged. The audit log gets the kind of call and a count only.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx

log = logging.getLogger(__name__)

# Delegated, read-only. MSAL adds openid, profile and offline_access by itself
# (and refuses them if passed). tests/test_outlook.py fails if anything else appears.
SCOPES = ["Mail.Read", "Calendars.Read"]
GRAPH = "https://graph.microsoft.com/v1.0"
LOGIN = "https://login.microsoftonline.com"
MAIL_FIELDS = "id,subject,from,receivedDateTime,isRead,bodyPreview"
EVENT_FIELDS = "id,subject,start,end,location,isAllDay,isCancelled,showAs,organizer"


class OutlookError(RuntimeError):
    """A failure Orion can say out loud (``spoken``)."""

    def __init__(self, spoken: str, detail: str = "") -> None:
        super().__init__(detail or spoken)
        self.spoken = spoken


class NotConnected(OutlookError):
    pass


@dataclass
class Email:
    id: str
    sender: str
    address: str
    subject: str
    received: datetime
    is_read: bool
    preview: str = ""


@dataclass
class Event:
    id: str
    subject: str
    start: datetime
    end: datetime
    location: str = ""
    all_day: bool = False
    show_as: str = "busy"         # free | tentative | busy | oof | workingElsewhere | unknown
    organizer: str = ""

    @property
    def blocks_time(self) -> bool:
        return self.show_as not in ("free", "workingElsewhere")


# ---- tokens -----------------------------------------------------------------------
class TokenVault:
    """Each person's serialized MSAL cache, inside one encrypted file. ``file=None``: memory only."""

    def __init__(self, file) -> None:
        self.file = file              # storage.encrypted_file.EncryptedJSONFile or None
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, str]] = {}
        if file is not None and file.exists():
            self._data = (file.load() or {}).get("accounts", {})

    def get(self, user_id: str) -> dict[str, str] | None:
        return self._data.get(user_id)

    def put(self, user_id: str, username: str, cache: str) -> None:
        with self._lock:
            self._data[user_id] = {"username": username, "cache": cache}
            self._save()

    def update_cache(self, user_id: str, cache: str) -> None:
        with self._lock:
            if user_id in self._data:
                self._data[user_id]["cache"] = cache
                self._save()

    def remove(self, user_id: str) -> bool:
        with self._lock:
            gone = self._data.pop(user_id, None) is not None
            if gone:
                self._save()
            return gone

    def connected(self) -> dict[str, str]:
        """user id -> Microsoft account name (an email address)."""
        return {uid: v.get("username", "") for uid, v in self._data.items()}

    def _save(self) -> None:
        if self.file is not None:
            self.file.save({"version": 1, "accounts": self._data})


class OutlookAuth:
    """MSAL public client per person; access tokens refreshed silently from the cache."""

    def __init__(self, client_id: str, tenant: str, vault: TokenVault, msal_module=None) -> None:
        self.client_id = client_id
        self.authority = f"{LOGIN}/{tenant or 'common'}"
        self.vault = vault
        self._msal = msal_module
        self._apps: dict[str, tuple[Any, Any]] = {}     # user id -> (app, cache); built lazily

    @property
    def msal(self):
        if self._msal is None:
            import msal

            self._msal = msal
        return self._msal

    def _app(self, user_id: str):
        if user_id not in self._apps:
            cache = self.msal.SerializableTokenCache()
            saved = self.vault.get(user_id)
            if saved and saved.get("cache"):
                cache.deserialize(saved["cache"])
            # Building the app contacts Microsoft for the authority's metadata, so only on demand.
            app = self.msal.PublicClientApplication(self.client_id, authority=self.authority,
                                                    token_cache=cache)
            self._apps[user_id] = (app, cache)
        return self._apps[user_id]

    def is_connected(self, user_id: str) -> bool:
        return self.vault.get(user_id) is not None

    def connect(self, user_id: str, show: Callable[[str], None] = print) -> str:
        """Device code sign-in. Prints Microsoft's instructions; returns the account name."""
        self._apps.pop(user_id, None)
        app, cache = self._app(user_id)
        flow = app.initiate_device_flow(scopes=SCOPES)
        if "user_code" not in flow:
            raise OutlookError("Microsoft didn't start the sign-in.", flow.get("error_description", ""))
        show(flow["message"])                 # "To sign in, use a web browser to open ... and enter the code ..."
        result = app.acquire_token_by_device_flow(flow)    # polls until done, declined, or expired
        if "access_token" not in result:
            raise OutlookError("The sign-in didn't finish.", result.get("error_description", result.get("error", "")))
        claims = result.get("id_token_claims") or {}
        username = claims.get("preferred_username") or claims.get("email") or ""
        if not username:
            accounts = app.get_accounts()
            username = accounts[0].get("username", "") if accounts else ""
        self.vault.put(user_id, username, cache.serialize())
        return username

    def token(self, user_id: str) -> str:
        """A valid access token for this person, refreshed silently when needed."""
        if not self.is_connected(user_id):
            raise NotConnected("Outlook isn't connected for you.")
        try:
            app, cache = self._app(user_id)
            accounts = app.get_accounts()
            result = app.acquire_token_silent(SCOPES, account=accounts[0]) if accounts else None
        except Exception as exc:              # MSAL raises requests errors when offline
            raise OutlookError("I can't reach Microsoft right now. Is the internet up?", str(exc)) from exc
        if cache.has_state_changed:
            self.vault.update_cache(user_id, cache.serialize())
            cache.has_state_changed = False
        if not result or "access_token" not in result:
            raise OutlookError("Your Outlook sign-in has expired. Connect it again from the terminal "
                               "with the outlook connect command.",
                               (result or {}).get("error_description", "no cached account"))
        return result["access_token"]

    def disconnect(self, user_id: str) -> bool:
        """Forget this person's tokens on this computer."""
        entry = self._apps.pop(user_id, None)
        if entry is not None:
            app, _ = entry
            try:
                for account in app.get_accounts():
                    app.remove_account(account)
            except Exception:
                pass
        return self.vault.remove(user_id)


# ---- Graph ------------------------------------------------------------------------
def _when(value: dict | str | None) -> datetime:
    """Graph timestamps: '2026-10-30T14:00:00Z' or {'dateTime': '...0000000', 'timeZone': 'UTC'}."""
    if isinstance(value, dict):
        value = value.get("dateTime", "")
    text = (value or "").rstrip("Z")
    if "." in text:
        head, frac = text.split(".", 1)
        text = f"{head}.{frac[:6]}"
    t = datetime.fromisoformat(text) if text else datetime.fromtimestamp(0)
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _email(row: dict) -> Email:
    addr = (row.get("from") or {}).get("emailAddress") or {}
    return Email(id=row.get("id", ""), sender=addr.get("name") or addr.get("address") or "someone",
                 address=addr.get("address", ""), subject=(row.get("subject") or "").strip(),
                 received=_when(row.get("receivedDateTime")), is_read=bool(row.get("isRead")),
                 preview=row.get("bodyPreview") or "")


def _event(row: dict) -> Event:
    loc = (row.get("location") or {}).get("displayName") or ""
    org = ((row.get("organizer") or {}).get("emailAddress") or {}).get("name") or ""
    return Event(id=row.get("id", ""), subject=(row.get("subject") or "").strip() or "an untitled event",
                 start=_when(row.get("start")), end=_when(row.get("end")), location=loc.strip(),
                 all_day=bool(row.get("isAllDay")), show_as=row.get("showAs") or "busy", organizer=org)


class GraphClient:
    """GETs against Microsoft Graph for one person. Read-only by construction: no other verb exists."""

    def __init__(self, auth: OutlookAuth, audit=None, timeout_s: float = 15.0,
                 client: httpx.Client | None = None) -> None:
        self.auth = auth
        self.audit = audit
        self._client = client or httpx.Client(timeout=timeout_s, base_url=GRAPH)

    def _get(self, user_id: str, kind: str, path: str, params: dict | None = None,
             headers: dict | None = None) -> dict:
        token = self.auth.token(user_id)
        h = {"Authorization": f"Bearer {token}", "Accept": "application/json", **(headers or {})}
        error = None
        try:
            r = self._client.get(path, params=params, headers=h)
        except httpx.HTTPError as exc:
            self._record(kind, user_id, error="network")
            raise OutlookError("I can't reach Outlook right now. Is the internet up?", type(exc).__name__) from exc
        if r.status_code == 401:
            error = OutlookError("Outlook turned down my sign-in. Connect it again from the terminal "
                                 "with the outlook connect command.")
        elif r.status_code == 403:
            error = OutlookError("Outlook says I don't have permission to read that.")
        elif r.status_code == 429:
            error = OutlookError("Outlook asked me to slow down. Try again in a minute.")
        elif r.status_code == 404:
            error = OutlookError("I couldn't find that in Outlook any more.")
        elif r.status_code >= 400:
            error = OutlookError("Outlook isn't answering properly right now.")
        if error is not None:
            self._record(kind, user_id, error=r.status_code)
            raise error
        try:
            data = r.json()
        except ValueError as exc:
            self._record(kind, user_id, error="bad json")
            raise OutlookError("Outlook sent me something I couldn't read.") from exc
        self._record(kind, user_id, items=len(data.get("value", [])) if "value" in data else 1)
        return data

    def _record(self, kind: str, user_id: str, **detail) -> None:
        if self.audit is not None:
            try:
                self.audit.record("outlook", call=kind, person=user_id, **detail)
            except Exception:
                log.exception("Couldn't write the audit record")

    # ---- mail ----------------------------------------------------------------------
    def unread_count(self, user_id: str) -> int:
        return int(self._get(user_id, "inbox", "/me/mailFolders/inbox",
                             {"$select": "unreadItemCount"}).get("unreadItemCount", 0))

    def unread(self, user_id: str, top: int = 10) -> list[Email]:
        # Graph needs the $orderby property in the $filter too, first, or it rejects the query.
        data = self._get(user_id, "unread", "/me/mailFolders/inbox/messages", {
            "$filter": "receivedDateTime ge 1900-01-01T00:00:00Z and isRead eq false",
            "$orderby": "receivedDateTime desc", "$top": str(top), "$select": MAIL_FIELDS})
        return [_email(x) for x in data.get("value", [])]

    def latest(self, user_id: str, top: int = 5) -> list[Email]:
        data = self._get(user_id, "latest", "/me/mailFolders/inbox/messages", {
            "$orderby": "receivedDateTime desc", "$top": str(top), "$select": MAIL_FIELDS})
        return [_email(x) for x in data.get("value", [])]

    def from_sender(self, user_id: str, sender: str, top: int = 5) -> list[Email]:
        # $search can't be combined with $orderby; results are sorted here instead.
        term = sender.replace('"', "").replace("\\", "").strip()
        data = self._get(user_id, "from", "/me/messages", {
            "$search": f'"from:{term}"', "$top": str(top), "$select": MAIL_FIELDS})
        return sorted((_email(x) for x in data.get("value", [])), key=lambda e: e.received, reverse=True)

    def body(self, user_id: str, message_id: str) -> str:
        """Plain text of just this message (uniqueBody leaves out the quoted thread below it)."""
        data = self._get(user_id, "body", f"/me/messages/{message_id}", {"$select": "uniqueBody,body"},
                         {"Prefer": 'outlook.body-content-type="text"'})
        unique = (data.get("uniqueBody") or {}).get("content") or ""
        return unique.strip() or ((data.get("body") or {}).get("content") or "").strip()

    # ---- calendar ------------------------------------------------------------------
    def events(self, user_id: str, start: datetime, end: datetime, top: int = 50) -> list[Event]:
        """Events overlapping [start, end), recurring ones expanded, cancelled ones dropped."""
        data = self._get(user_id, "calendar", "/me/calendarView", {
            "startDateTime": _iso(start), "endDateTime": _iso(end), "$orderby": "start/dateTime",
            "$top": str(top), "$select": EVENT_FIELDS}, {"Prefer": 'outlook.timezone="UTC"'})
        out = []
        for row in data.get("value", []):
            if row.get("isCancelled"):
                continue
            ev = _event(row)
            if ev.all_day:
                # All-day events are dates, not instants: midnight to midnight where you are.
                tz = start.tzinfo
                ev.start = datetime.combine(ev.start.date(), datetime.min.time(), tzinfo=tz)
                ev.end = datetime.combine(ev.end.date(), datetime.min.time(), tzinfo=tz)
            out.append(ev)
        return sorted(out, key=lambda e: e.start)


# ---- wiring -----------------------------------------------------------------------
def token_file(data_dir: Path) -> Path:
    return data_dir / "outlook.enc"


def open_vault(data_dir: Path, data_key: bytes | None) -> TokenVault:
    from storage.encrypted_file import EncryptedJSONFile

    if data_key is None:
        return TokenVault(None)
    return TokenVault(EncryptedJSONFile(token_file(data_dir), data_key, "outlook"))


def sign_out_hook(data_dir: Path) -> Callable[[], None]:
    """For the wipe: shred every Outlook token on this computer. Silent, offline."""
    def hook() -> None:
        from security.wipe import shred

        path = token_file(data_dir)
        if path.exists():
            shred(path)
    return hook
