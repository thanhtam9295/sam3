"""Platform foundations (PROMPT_PLATFORM.md §2): settings, secrets, data roots."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import keyring
import pytest
from keyring.backend import KeyringBackend

from hcs_studio import config, secrets
from hcs_studio.config import AppSettings, DataRootConfig, load_settings, save_settings
from hcs_studio.paths import ABSOLUTE, RootedPath, resolve, resolve_text, to_rooted


class MemoryKeyring(KeyringBackend):
    priority = 1

    def __init__(self) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        self.store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self.store.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.store[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        from keyring.errors import PasswordDeleteError

        if (service, username) not in self.store:
            raise PasswordDeleteError(username)
        del self.store[(service, username)]


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def memory_keyring() -> Iterator[MemoryKeyring]:
    previous = keyring.get_keyring()
    backend = MemoryKeyring()
    keyring.set_keyring(backend)
    yield backend
    keyring.set_keyring(previous)


# ------------------------------------------------------------------- settings


def test_settings_round_trip_versions_and_legacy_import(home: Path) -> None:
    assert load_settings() == AppSettings()
    (home / "sam3.json").write_text('{"python": "/envs/sam3/bin/python"}', encoding="utf-8")
    settings = AppSettings(
        data_roots=[DataRootConfig(name="screening", windows=r"\\lab\screening", linux="/mnt/s")]
    )
    save_settings(settings)
    loaded = load_settings()
    assert loaded.root("SCREENING") is not None and loaded.data_roots[0].name == "SCREENING"
    assert loaded.imported["sam3"] == {"python": "/envs/sam3/bin/python"}  # imported once
    raw = json.loads((home / "settings.json").read_text(encoding="utf-8"))
    assert raw["config_version"] == config.CONFIG_VERSION
    raw["config_version"] = config.CONFIG_VERSION + 1
    (home / "settings.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="Update HCS Studio"):
        load_settings()


def test_settings_validation_and_no_plain_secrets(home: Path) -> None:
    with pytest.raises(ValueError, match="unique"):
        AppSettings(data_roots=[DataRootConfig(name="AA"), DataRootConfig(name="aa")])
    with pytest.raises(ValueError, match="at least 2"):
        DataRootConfig(name="C")  # would look like a drive letter
    with pytest.raises(ValueError, match="reserved"):
        DataRootConfig(name="absolute")
    config.check_no_secrets({"smtp": {"password": "secret:smtp/password"}})
    with pytest.raises(ValueError, match="keychain"):
        config.check_no_secrets({"smtp": {"password": "hunter22"}})
    with pytest.raises(ValueError, match="keychain"):
        config.check_no_secrets({"hooks": [{"webhook_url": "https://x"}]})


def test_settings_migrations_run_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "CONFIG_VERSION", 3)
    monkeypatch.setattr(
        config,
        "MIGRATIONS",
        {1: lambda d: {**d, "a": 1}, 2: lambda d: {**d, "b": d["a"] + 1}},
    )
    assert config.migrate({"config_version": 1}) == {"config_version": 3, "a": 1, "b": 2}


# -------------------------------------------------------------------- secrets


def test_secrets_keychain_environment_and_redaction(
    memory_keyring: MemoryKeyring, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    assert secrets.get_secret("smtp/password") is None
    secrets.set_secret("smtp/password", "correct horse")
    assert memory_keyring.store[("hcs-studio", "smtp/password")] == "correct horse"
    assert secrets.resolve("secret:smtp/password") == "correct horse"
    assert secrets.resolve("plain value") == "plain value"
    monkeypatch.setenv("HCS_SECRET_ASSISTANT_API_KEY", "sk-from-env")
    assert secrets.get_secret("assistant/api-key") == "sk-from-env"  # environment wins
    with pytest.raises(secrets.SecretError, match="not set"):
        secrets.resolve("secret:missing")
    logger = logging.getLogger("hcs_test_secrets")
    handler = logging.Handler()
    records: list[str] = []
    handler.emit = lambda record: records.append(record.getMessage())  # type: ignore[method-assign]
    logger.addHandler(handler)
    secrets.install_log_redaction(logger)
    logger.warning("login with %s failed", "correct horse")
    assert records == ["login with *** failed"]
    secrets.delete_secret("smtp/password")
    assert secrets.get_secret("smtp/password") is None
    with pytest.raises(ValueError, match="Invalid secret name"):
        secrets.get_secret("bad name!")


# ----------------------------------------------------------------- data roots

ROOTS = [
    DataRootConfig(name="SCREENING", windows=r"\\labserver\screening", linux="/mnt/screening"),
    DataRootConfig(name="SCREENING_RAW", windows=r"\\labserver\screening\raw", linux="/mnt/raw"),
    DataRootConfig(name="RESULTS", windows=r"D:\Results", linux="/data/results"),
]


def test_rooted_paths_round_trip_between_windows_and_linux() -> None:
    win = to_rooted(r"\\LabServer\Screening\raw\plate 01\Index.idx.xml", ROOTS, "win32")
    assert win == RootedPath("SCREENING_RAW", "plate 01/Index.idx.xml")  # longest root wins
    assert resolve_text(win, ROOTS, "linux") == "/mnt/raw/plate 01/Index.idx.xml"
    lin = to_rooted("/data/results/run1/run_manifest.json", ROOTS, "linux")
    assert lin == RootedPath("RESULTS", "run1/run_manifest.json")
    assert resolve_text(lin, ROOTS, "win32") == r"D:\Results\run1\run_manifest.json"
    assert RootedPath.parse(str(lin)) == lin
    with pytest.raises(ValueError, match="Not a rooted path"):
        RootedPath.parse(r"C:\data\x")  # a drive letter is not a root


def test_paths_outside_roots_and_unknown_roots() -> None:
    outside = to_rooted("/home/me/plate", ROOTS, "linux")
    assert outside.root == ABSOLUTE and outside.is_absolute_fallback
    assert resolve(outside, ROOTS, "linux") == Path("/home/me/plate")
    assert to_rooted("/mnt/screening-old/x", ROOTS, "linux").root == ABSOLUTE  # not a prefix
    with pytest.raises(KeyError, match="not defined"):
        resolve(RootedPath("ARCHIVE", "x"), ROOTS, "linux")
    with pytest.raises(KeyError, match="no macos location"):
        resolve(RootedPath("RESULTS", "x"), ROOTS, "darwin")
