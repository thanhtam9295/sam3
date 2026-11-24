"""Roles and the permission matrix (Phase 26).

Four roles, each including the ones below it: **viewer** (looks only) < **analyst** (runs and
analyses, edits layouts, reviews QC with reasons) < **reviewer** (signs runs as reviewed or
approved, changes model status) < **admin** (settings, accounts, QC and routing rules,
unlocking). A state change is a service event; its type decides the role it needs — the most
specific matching prefix of ``PERMISSIONS`` wins.
"""

from __future__ import annotations

ROLES = ("viewer", "analyst", "reviewer", "admin")
RANK = {role: i for i, role in enumerate(ROLES)}

#: event-type prefix → the lowest role allowed ("" = every other change).
PERMISSIONS: dict[str, str] = {
    "": "analyst",
    "run.sign": "reviewer",
    "run.unlock": "admin",
    "models.status": "reviewer",
    "settings": "admin",
    "accounts": "admin",
    "qc.rules": "admin",
    "automation.rules": "admin",
    #: asking the lab assistant (read-only; audited, counted against the user's budget)
    "assistant": "viewer",
}


def required_role(event_type: str) -> str:
    matches = [
        p for p in PERMISSIONS if event_type == p or event_type.startswith(p + ".") or p == ""
    ]
    return PERMISSIONS[max(matches, key=len)]


def allowed(role: str, event_type: str) -> bool:
    if role not in RANK:
        return False
    return RANK[role] >= RANK[required_role(event_type)]


def matrix() -> list[dict[str, str]]:
    """The permission matrix for the docs and the accounts dialog."""
    return [{"action": p or "(any other change)", "lowest role": r} for p, r in PERMISSIONS.items()]
