"""Secrets (API keys, SMTP and webhook credentials, database passwords) in the OS keychain.

They are stored with :mod:`keyring` under the service ``hcs-studio`` and never written to
settings files, logs or run folders. A machine without a keychain backend (a headless Linux
server) can provide them as environment variables ``HCS_SECRET_<NAME>`` (name upper-cased,
``/`` and ``-`` → ``_``). Every value read is registered with :class:`RedactingFilter`, which
replaces it by ``***`` in log records.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import threading

SERVICE = "hcs-studio"
_NAME = re.compile(r"^[a-z0-9][a-z0-9_./-]*$")
_KNOWN: set[str] = set()
_LOCK = threading.Lock()


class SecretError(RuntimeError):
    pass


def _check(name: str) -> str:
    name = name.strip().lower()
    if not _NAME.fullmatch(name):
        raise ValueError(f"Invalid secret name {name!r} (letters, digits, _ . / -)")
    return name


def env_name(name: str) -> str:
    return "HCS_SECRET_" + re.sub(r"[^A-Z0-9]", "_", _check(name).upper())


def _remember(value: str) -> None:
    if len(value) >= 4:  # very short values would redact ordinary text
        with _LOCK:
            _KNOWN.add(value)


def get_secret(name: str) -> str | None:
    """The secret, from the environment (``HCS_SECRET_<NAME>``) or the keychain; None if unset."""
    value = os.environ.get(env_name(name))
    if value is None:
        try:
            import keyring

            value = keyring.get_password(SERVICE, _check(name))
        except Exception as exc:  # no backend, locked keychain …
            raise SecretError(
                f"The OS keychain is not available ({type(exc).__name__}); set "
                f"{env_name(name)} instead"
            ) from exc
    if value:
        _remember(value)
    return value or None


def set_secret(name: str, value: str) -> None:
    if not value:
        raise ValueError("Empty secret")
    import keyring

    try:
        keyring.set_password(SERVICE, _check(name), value)
    except Exception as exc:
        raise SecretError(
            f"Could not store the secret in the OS keychain ({type(exc).__name__}: {exc})"
        ) from exc
    _remember(value)


def delete_secret(name: str) -> None:
    import keyring
    from keyring.errors import PasswordDeleteError

    with contextlib.suppress(PasswordDeleteError):
        keyring.delete_password(SERVICE, _check(name))


def resolve(value: str) -> str:
    """A settings value: ``secret:<name>`` is looked up, anything else returned as is."""
    from hcs_studio.config import SECRET_PREFIX

    if value.startswith(SECRET_PREFIX):
        secret = get_secret(value[len(SECRET_PREFIX) :])
        if secret is None:
            raise SecretError(f"Secret {value[len(SECRET_PREFIX) :]!r} is not set")
        return secret
    return value


def redact(text: str) -> str:
    with _LOCK:
        known = sorted(_KNOWN, key=len, reverse=True)
    for value in known:
        text = text.replace(value, "***")
    return text


class RedactingFilter(logging.Filter):
    """Replace every secret read in this process by ``***`` in log messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        if _KNOWN:
            message = record.getMessage()
            cleaned = redact(message)
            if cleaned != message:
                record.msg, record.args = cleaned, None
        return True


def install_log_redaction(logger: logging.Logger | None = None) -> None:
    """Add the filter to every handler of ``logger`` (default: the root logger)."""
    target = logger or logging.getLogger()
    for handler in target.handlers:
        if not any(isinstance(f, RedactingFilter) for f in handler.filters):
            handler.addFilter(RedactingFilter())
