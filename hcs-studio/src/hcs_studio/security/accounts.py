"""Accounts in local mode (Phase 26): OS identity, an optional app password, roles.

Who acts is the person logged in to the computer. Without any account in the registry the
computer has one user, who is an admin (a single-user installation keeps working as before).
Once accounts exist, the OS user name selects the account and its role; an OS user without an
account may only look (viewer).

Passwords (optional; required in regulated mode) are hashed with argon2id (``argon2-cffi``).
Failed attempts lock the account for ``lockout_minutes`` after ``lockout_attempts`` failures; a
signed-in session ends after ``session_timeout_min`` of inactivity. Signatures (Phase 26b) can
require the password again (re-authentication).
"""

from __future__ import annotations

import getpass
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from hcs_studio.security.roles import ROLES


class AuthError(PermissionError):
    """Wrong password, locked or inactive account."""


@dataclass(frozen=True)
class AccountInfo:
    username: str
    display_name: str
    role: str
    active: bool
    has_password: bool
    locked_until: datetime | None = None
    last_login: datetime | None = None


def os_username() -> str:
    try:
        return getpass.getuser()
    except Exception:  # no login name (services, containers)
        return "unknown"


def _hasher() -> Any:
    from argon2 import PasswordHasher

    return PasswordHasher()


def hash_password(password: str) -> str:
    if len(password) < 8:
        raise ValueError("Passwords need at least 8 characters")
    return str(_hasher().hash(password))


def _settings() -> Any:
    from hcs_studio.config import load_settings

    return load_settings().security


def _info(row: Any) -> AccountInfo:
    return AccountInfo(
        row.username,
        row.display_name,
        row.role,
        row.active,
        bool(row.password_hash),
        row.locked_until,
        row.last_login,
    )


def list_accounts() -> list[AccountInfo]:
    from sqlalchemy import select

    from hcs_studio.db import session_scope
    from hcs_studio.db.security import Account

    with session_scope() as session:
        return [_info(r) for r in session.scalars(select(Account).order_by(Account.username))]


def get_account(username: str) -> AccountInfo | None:
    from sqlalchemy import select

    from hcs_studio.db import session_scope
    from hcs_studio.db.security import Account

    with session_scope() as session:
        row = session.scalar(select(Account).where(Account.username == username))
        return _info(row) if row is not None else None


def _row(session: Any, username: str) -> Any:
    from sqlalchemy import select

    from hcs_studio.db.security import Account

    row = session.scalar(select(Account).where(Account.username == username))
    if row is None:
        raise ValueError(f"No account {username!r}")
    return row


def create_account(
    username: str, role: str, display_name: str = "", password: str | None = None
) -> AccountInfo:
    from hcs_studio.db import session_scope
    from hcs_studio.db.security import Account

    if role not in ROLES:
        raise ValueError(f"Unknown role {role!r} ({', '.join(ROLES)})")
    if not username.strip():
        raise ValueError("A username is needed")
    if get_account(username) is not None:
        raise ValueError(f"Account {username!r} exists")
    with session_scope() as session:
        row = Account(
            username=username,
            display_name=display_name or username,
            role=role,
            password_hash=hash_password(password) if password else "",
        )
        session.add(row)
        session.flush()
        return _info(row)


def update_account(
    username: str,
    *,
    role: str | None = None,
    display_name: str | None = None,
    active: bool | None = None,
    password: str | None = None,
) -> AccountInfo:
    from hcs_studio.db import session_scope

    if role is not None and role not in ROLES:
        raise ValueError(f"Unknown role {role!r} ({', '.join(ROLES)})")
    with session_scope() as session:
        row = _row(session, username)
        if role is not None:
            row.role = role
        if display_name is not None:
            row.display_name = display_name
        if active is not None:
            row.active = active
        if password is not None:
            row.password_hash = hash_password(password) if password else ""
            row.failed_attempts, row.locked_until = 0, None
        return _info(row)


def authenticate(username: str, password: str) -> AccountInfo:
    """Check a password (with lockout). Raises :class:`AuthError`."""
    from argon2.exceptions import VerificationError, VerifyMismatchError

    from hcs_studio.db import session_scope

    settings = _settings()
    now = datetime.now(UTC)
    with session_scope() as session:
        try:
            row = _row(session, username)
        except ValueError:
            raise AuthError("Unknown user or wrong password") from None
        if not row.active:
            raise AuthError(f"The account {username} is disabled")
        locked = row.locked_until
        if locked is not None and _aware(locked) > now:
            raise AuthError(f"The account {username} is locked until {_aware(locked):%H:%M} UTC")
        if not row.password_hash:
            raise AuthError(f"The account {username} has no password; an admin must set one")
        try:
            _hasher().verify(row.password_hash, password)
        except (VerifyMismatchError, VerificationError):
            row.failed_attempts += 1
            if row.failed_attempts >= settings.lockout_attempts:
                row.locked_until = now + timedelta(minutes=settings.lockout_minutes)
                row.failed_attempts = 0
            # the failed attempt is counted even though we raise: commit it first
            session.commit()
            raise AuthError("Unknown user or wrong password") from None
        row.failed_attempts, row.locked_until, row.last_login = 0, None, now
        if _hasher().check_needs_rehash(row.password_hash):
            row.password_hash = _hasher().hash(password)
        return _info(row)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def role_of(username: str) -> str:
    """The role of an OS user: their account's, admin when no accounts exist, else viewer."""
    try:
        accounts = list_accounts()
    except Exception:  # no registry (read-only installs): a single user
        return "admin"
    if not accounts:
        return "admin"
    for account in accounts:
        if account.username == username:
            return account.role if account.active else "viewer"
    return "viewer"


@dataclass
class Session:
    """A signed-in user in this process (GUI): expires after inactivity."""

    username: str
    role: str
    display_name: str
    last_active: datetime

    def expired(self, timeout_min: float) -> bool:
        return datetime.now(UTC) - self.last_active > timedelta(minutes=timeout_min)

    def touch(self) -> None:
        self.last_active = datetime.now(UTC)


_SESSION: Session | None = None


def sign_in(username: str, password: str) -> Session:
    global _SESSION
    info = authenticate(username, password)
    _SESSION = Session(info.username, info.role, info.display_name, datetime.now(UTC))
    return _SESSION


def adopt_session(username: str, role: str, display_name: str = "") -> Session:
    """Connected mode: the account signed in to the server acts here too (the server checked
    the password and enforces the role)."""
    global _SESSION
    _SESSION = Session(username, role, display_name or username, datetime.now(UTC))
    return _SESSION


def sign_out() -> None:
    global _SESSION
    _SESSION = None


def current_session() -> Session | None:
    """The signed-in session, or None when there is none or it timed out."""
    global _SESSION
    if _SESSION is None:
        return None
    if _SESSION.expired(_settings().session_timeout_min):
        _SESSION = None
        return None
    return _SESSION
