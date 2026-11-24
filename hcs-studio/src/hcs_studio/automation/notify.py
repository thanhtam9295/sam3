"""Notifications of automatic runs (Phase 19.4): email (SMTP), Teams/Slack incoming webhooks,
and desktop messages (shown by the app).

A message names the barcode, assay, QC verdict, key numbers (Z′, mean cell count) and the
report. Each channel lists the kinds it wants (``on``); failures (``failed``, ``qc_failed``,
``unrouted``) also go to the rule's owners and the admin list. Passwords and webhook URLs are
keychain references (``secret:<name>``); a channel whose secret is missing is skipped with a
warning, never sent in clear.
"""

from __future__ import annotations

import json
import logging
import smtplib
import ssl
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from email.message import EmailMessage
from typing import Any

from hcs_studio.automation.settings import Channel

log = logging.getLogger("hcs_studio.automation")

FAILURES = ("failed", "qc_failed", "unrouted")


@dataclass
class Notice:
    kind: str  # completed | qc_failed | failed | needs_layout | unrouted
    barcode: str
    title: str
    lines: list[str] = field(default_factory=list)
    numbers: dict[str, float | None] = field(default_factory=dict)
    report: str = ""
    #: extra recipients (the routing rule's owners)
    owners: list[str] = field(default_factory=list)

    @property
    def subject(self) -> str:
        return f"[HCS Studio] {self.barcode}: {self.title}"

    def text(self) -> str:
        body = [self.title, "", *list(self.lines)]
        if self.numbers:
            body += [""] + [
                f"{k}: {'–' if v is None else f'{v:.3g}'}" for k, v in self.numbers.items()
            ]
        if self.report:
            body += ["", f"Report: {self.report}"]
        return "\n".join(body)


def _secret(reference: str) -> str | None:
    from hcs_studio.secrets import get_secret

    if not reference:
        return None
    if not reference.startswith("secret:"):
        raise ValueError("Passwords and webhooks must be keychain references (secret:<name>)")
    return get_secret(reference.removeprefix("secret:"))


def send_email(channel: Channel, notice: Notice, recipients: list[str]) -> None:
    message = EmailMessage()
    message["Subject"] = notice.subject
    message["From"] = channel.sender
    message["To"] = ", ".join(recipients)
    message.set_content(notice.text())
    with smtplib.SMTP(channel.smtp_host, channel.smtp_port, timeout=30) as smtp:
        if channel.smtp_tls:
            smtp.starttls(context=ssl.create_default_context())
        if channel.username:
            password = _secret(channel.password)
            if password is None:
                raise RuntimeError(f"no SMTP password stored for {channel.password}")
            smtp.login(channel.username, password)
        smtp.send_message(message)


def send_webhook(channel: Channel, notice: Notice) -> None:
    url = _secret(channel.webhook)
    if not url:
        raise RuntimeError(f"no webhook stored for {channel.webhook or channel.name}")
    body = {"text": f"**{notice.subject}**\n\n{notice.text()}"}
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15):
        pass


class Notifier:
    """Sends notices to the configured channels; ``desktop`` receives desktop notices."""

    def __init__(
        self,
        channels: list[Channel],
        admins: list[str] | None = None,
        desktop: Callable[[Notice], None] | None = None,
    ) -> None:
        self.channels = channels
        self.admins = list(admins or [])
        self.desktop = desktop
        #: (channel, kind, status, detail) of every attempt (tests, the Automation tab)
        self.sent: list[tuple[str, str, str, str]] = []

    def recipients(self, channel: Channel, notice: Notice) -> list[str]:
        out = list(channel.to) + list(notice.owners)
        if notice.kind in FAILURES:
            out += self.admins
        return list(dict.fromkeys(a for a in out if a))

    def notify(self, notice: Notice) -> None:
        for channel in self.channels:
            if notice.kind not in channel.on:
                continue
            try:
                if channel.kind == "email":
                    recipients = self.recipients(channel, notice)
                    if not recipients:
                        continue
                    send_email(channel, notice, recipients)
                    detail = ", ".join(recipients)
                elif channel.kind in ("teams", "slack"):
                    send_webhook(channel, notice)
                    detail = channel.kind
                else:
                    if self.desktop is not None:
                        self.desktop(notice)
                    detail = "desktop"
                self.sent.append((channel.name, notice.kind, "sent", detail))
            except Exception as exc:
                log.warning("Notification %s via %s failed: %s", notice.kind, channel.name, exc)
                self.sent.append((channel.name, notice.kind, "failed", str(exc)))

    def summary(self) -> list[dict[str, Any]]:
        return [{"channel": c, "kind": k, "status": s, "detail": d} for c, k, s, d in self.sent]
