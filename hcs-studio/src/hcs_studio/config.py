"""Application settings: one versioned file, ``<settings>/settings.json``.

The file holds data roots, the database location and other non-secret configuration. Its
``config_version`` lets later versions migrate old files step by step (:data:`MIGRATIONS`).
Secrets never go here: a value such as ``"secret:smtp/password"`` is a reference to the OS
keychain (:mod:`hcs_studio.secrets`).

Older per-feature files (``sam3.json``, ``ilastik.json``) keep working; their content is
imported once into ``settings.json`` under ``imported`` for reference.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from hcs_studio.automation.settings import AutomationSettings
from hcs_studio.io.sidecar import config_dir

CONFIG_VERSION = 1
SECRET_PREFIX = "secret:"


class DataRootConfig(BaseModel):
    """A named location that differs between computers (``SCREENING`` = ``\\\\lab\\screening``
    on Windows, ``/mnt/screening`` on Linux)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    windows: str = ""
    linux: str = ""
    macos: str = ""
    description: str = ""

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 2 or not value.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "Data root names have at least 2 characters: letters, digits, '-' and '_'"
            )
        if value.upper() == "ABSOLUTE":
            raise ValueError("'ABSOLUTE' is reserved for paths outside every data root")
        return value.upper()


class SecuritySettings(BaseModel):
    """Accounts, sign-off and the audit trail (Phase 26). Research defaults; ``regulated_mode``
    turns on the strict settings (password, re-authentication, four-eyes approval)."""

    model_config = ConfigDict(extra="forbid")

    regulated_mode: bool = False
    require_password: bool = False
    reauth_for_signoff: bool = False
    session_timeout_min: float = Field(default=30.0, gt=0)
    lockout_attempts: int = Field(default=5, ge=1)
    lockout_minutes: float = Field(default=15.0, gt=0)
    #: Write the audit chain head every N records (0 = only on demand).
    audit_head_every: int = Field(default=50, ge=0)
    #: Extra folders (e.g. on another computer) that keep copies of the chain head.
    audit_head_copies: list[str] = Field(default_factory=list)

    @property
    def password_required(self) -> bool:
        return self.require_password or self.regulated_mode

    @property
    def reauth_required(self) -> bool:
        return self.reauth_for_signoff or self.regulated_mode


class ServerSettings(BaseModel):
    """Server mode (Phase 27). On the server: where runs are written and how long API sessions
    last. On a desktop: the server to connect to (empty = work locally)."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(default="", description="https://hcs.example.org (desktop: connect to it)")
    verify_tls: bool = True
    #: Server: the results share for runs submitted through the API (a data-root path).
    results_root: str = ""
    #: Server: API sessions end after this many hours (and after the idle timeout).
    session_hours: float = Field(default=12.0, gt=0)


class DatabaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["local", "server"] = "local"
    #: local: SQLite file (empty = <settings>/registry.sqlite); server: a PostgreSQL URL
    #: without the password (the password is the secret ``database/password``).
    url: str = ""


#: Claude models the assistant may use (PROMPT_PLATFORM.md §28.1).
ASSISTANT_MODELS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-fable-5-2")


class AssistantSettings(BaseModel):
    """The Claude lab assistant (Phase 28). Off until an admin enables it; the API key is a
    keychain reference (local mode) or lives only on the server (server mode)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    #: routine questions
    model: str = "claude-opus-5-5"
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    #: "use a stronger model" for one question
    strong_model: str = "claude-fable-5-2"
    strong_effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    api_key: str = "secret:anthropic/api_key"
    #: weighted tokens per user and calendar month (UTC); 0 = no limit
    monthly_token_budget: int = Field(default=2_000_000, ge=0)
    user_budgets: dict[str, int] = Field(default_factory=dict)
    #: compound names replaced by codes unless a question opts out
    compound_ids_only: bool = False
    #: whether users may enable image thumbnails for a question (each image is confirmed)
    allow_images: bool = False
    max_rows: int = Field(default=200, ge=1, le=5000)
    query_timeout_s: float = Field(default=5.0, gt=0, le=120)
    #: model turns per question before the assistant stops
    max_turns: int = Field(default=12, ge=1, le=50)
    max_tokens: int = Field(default=16000, ge=1024, le=64000)
    #: re-run a declined request on a fallback model (server-side fallback)
    refusal_fallback: bool = True
    #: strict tool schemas (turn off where structured outputs are unavailable, e.g. CMEK)
    strict_tools: bool = True

    @field_validator("model", "strong_model")
    @classmethod
    def _model(cls, value: str) -> str:
        if not value.startswith("claude-"):
            raise ValueError("A Claude model ID, e.g. claude-opus-5-5")
        return value


class AppSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_version: int = CONFIG_VERSION
    data_roots: list[DataRootConfig] = Field(default_factory=list)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    automation: AutomationSettings = Field(default_factory=AutomationSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    assistant: AssistantSettings = Field(default_factory=AssistantSettings)
    #: Earlier per-feature settings files, imported once (read-only copy).
    imported: dict[str, Any] = Field(default_factory=dict)

    @field_validator("data_roots")
    @classmethod
    def _unique(cls, roots: list[DataRootConfig]) -> list[DataRootConfig]:
        names = [r.name for r in roots]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"Data root names must be unique: {', '.join(duplicates)}")
        return roots

    def root(self, name: str) -> DataRootConfig | None:
        return next((r for r in self.data_roots if r.name == name.upper()), None)


#: config_version → function upgrading a raw dict from that version to the next one.
MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {}


def settings_path() -> Path:
    return config_dir() / "settings.json"


def migrate(data: dict[str, Any]) -> dict[str, Any]:
    """Upgrade a settings dict to :data:`CONFIG_VERSION` (refuses newer files)."""
    version = int(data.get("config_version", 1))
    if version > CONFIG_VERSION:
        raise ValueError(
            f"settings.json is version {version}; this HCS Studio understands up to "
            f"{CONFIG_VERSION}. Update HCS Studio."
        )
    while version < CONFIG_VERSION:
        data = MIGRATIONS[version](dict(data))
        version += 1
        data["config_version"] = version
    return data


def _import_legacy(settings: AppSettings) -> bool:
    changed = False
    for name in ("sam3", "ilastik"):
        path = config_dir() / f"{name}.json"
        if name in settings.imported or not path.exists():
            continue
        try:
            settings.imported[name] = json.loads(path.read_text(encoding="utf-8"))
            changed = True
        except (OSError, ValueError):
            continue
    return changed


def load_settings() -> AppSettings:
    """The settings (defaults when there is no file yet)."""
    path = settings_path()
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        settings = AppSettings.model_validate(migrate(data))
    else:
        settings = AppSettings()
    if _import_legacy(settings) and path.parent.exists():
        save_settings(settings)
    return settings


def save_settings(settings: AppSettings) -> Path:
    check_no_secrets(settings.model_dump(mode="json"))
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(settings.model_dump(mode="json"), indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


_SECRET_KEYS = ("password", "token", "api_key", "apikey", "secret", "webhook")


def check_no_secrets(data: Any, where: str = "") -> None:
    """Refuse plain-text secrets in settings: keys that look secret must hold a
    ``secret:`` reference (or be empty)."""
    if isinstance(data, dict):
        for key, value in data.items():
            path = f"{where}.{key}" if where else str(key)
            if (
                isinstance(value, str)
                and value
                and any(k in str(key).lower() for k in _SECRET_KEYS)
                and not value.startswith(SECRET_PREFIX)
            ):
                raise ValueError(
                    f"settings {path}: store secrets in the keychain and write "
                    f"'{SECRET_PREFIX}<name>' here"
                )
            check_no_secrets(value, path)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            check_no_secrets(item, f"{where}[{i}]")
