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


class DatabaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["local", "server"] = "local"
    #: local: SQLite file (empty = <settings>/registry.sqlite); server: a PostgreSQL URL
    #: without the password (the password is the secret ``database/password``).
    url: str = ""


class AppSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_version: int = CONFIG_VERSION
    data_roots: list[DataRootConfig] = Field(default_factory=list)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    automation: AutomationSettings = Field(default_factory=AutomationSettings)
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
