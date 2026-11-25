"""Phase 28: the assistant in server mode (the key stays on the server; desktops use
``RemoteAssistant``), the admin-only settings, the command line and the Assistant tab."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from hcs_studio.assistant import Privacy, Question
from hcs_studio.assistant.session import LocalAssistant, RemoteAssistant
from hcs_studio.config import AssistantSettings


class Scripted:
    """A stand-in for ``anthropic.Anthropic``: a query, then an answer (repeats)."""

    def __init__(self, image: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.image = image
        self.messages = self
        self.beta = self

    def create(self, **body: Any) -> dict[str, Any]:
        self.calls.append(json.loads(json.dumps(body, default=str)))
        usage = {"input_tokens": 50, "output_tokens": 10}
        last = body["messages"][-1]
        if last["role"] == "user" and isinstance(last["content"], str):
            if self.image:
                call = {
                    "type": "tool_use",
                    "id": "i",
                    "name": "get_image_thumbnail",
                    "input": {"run_id": 1, "well": "B02", "field": 1},
                }
            else:
                call = {
                    "type": "tool_use",
                    "id": "q",
                    "name": "run_query",
                    "input": {"sql": "SELECT COUNT(*) AS n FROM plates", "purpose": "count"},
                }
            return {"content": [call], "stop_reason": "tool_use", "usage": usage}
        content = last["content"][0]["content"]
        text = "There is 1 plate (Drug A on it)." if isinstance(content, str) else "Seen."
        return {
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "usage": usage,
        }


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from hcs_studio.db import session_scope
    from hcs_studio.db.engine import dispose_engines
    from hcs_studio.db.scanner import scan
    from hcs_studio.synthetic import registry_runs

    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HCS_DATABASE_URL", raising=False)
    dispose_engines()
    registry_runs(tmp_path / "runs", n_runs=1)
    with session_scope() as session:
        scan(session, [tmp_path / "runs"])
    yield tmp_path
    dispose_engines()


def _enable(**changes: Any) -> None:
    from hcs_studio.config import load_settings, save_settings

    settings = load_settings()
    settings.assistant = AssistantSettings(**{"enabled": True, **changes})
    save_settings(settings)


def test_server_mode(home: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from hcs_studio.security import accounts
    from hcs_studio.server.api import create_app
    from hcs_studio.server.client import ServerClient, ServerError
    from hcs_studio.services import recent_events

    accounts.create_account("admin", "admin", password="admin password")
    accounts.create_account("vera", "viewer", "Vera Viewer", password="vera password")
    app = create_app()
    fake = Scripted()
    app.state.assistant.client_factory = lambda settings: fake
    client = ServerClient(http=TestClient(app))
    client.login("vera", "vera password")
    remote = RemoteAssistant(client)
    assert remote.status()["enabled"] is False
    with pytest.raises(ServerError) as off:
        remote.ask(Question("How many plates?"))
    assert off.value.status == 403 and not fake.calls  # off until an admin enables it
    _enable()
    answer = remote.ask(Question("How many plates?", privacy=Privacy(compound_ids_only=True)))
    assert answer.status == "answered" and answer.steps[0].tool == "run_query"
    assert answer.steps[0].sql == ["SELECT COUNT(*) AS n FROM plates"]
    assert remote.conversation_id
    # the server sent codes, the user sees names
    sent = json.dumps(fake.calls)
    assert "Drug A" not in sent
    first = remote.conversation_id
    remote.ask(Question("And now?", privacy=Privacy(compound_ids_only=True)))
    assert remote.conversation_id == first  # the same conversation continues
    remote.ask(Question("Names please"))  # other privacy settings → a new conversation
    assert remote.conversation_id != first
    preview = remote.preview(Question("What is sent?"))
    assert preview["messages"][-1] == {"role": "user", "content": "What is sent?"}
    status = remote.status()
    assert status["usage"]["requests"] == 6 and status["usage"]["user"] == "vera"
    asked = [e for e in recent_events(50) if e["type"] == "assistant.question"]
    assert asked and asked[0]["actor"] == "vera"
    # another account cannot use vera's conversation
    other = ServerClient(http=TestClient(app))
    other.login("admin", "admin password")
    with pytest.raises(ServerError) as foreign:
        other.post(f"/api/assistant/{remote.conversation_id}/confirm", {"approved": True})
    assert foreign.value.status == 404
    remote.close()


def test_image_confirmation_through_the_server(home: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from PIL import Image

    from hcs_studio.server.api import create_app
    from hcs_studio.server.client import ServerClient

    runs = sorted((home / "runs").glob("REG-*"))
    previews = runs[0] / "wells" / "B02" / "previews"
    previews.mkdir(parents=True)
    Image.new("RGB", (64, 64), (0, 200, 0)).save(previews / "f001.png")
    _enable(allow_images=True)
    app = create_app()
    fake = Scripted(image=True)
    app.state.assistant.client_factory = lambda settings: fake
    client = ServerClient(http=TestClient(app))
    from hcs_studio.security import accounts

    accounts.create_account("anna", "analyst", password="anna password")
    client.login("anna", "anna password")
    remote = RemoteAssistant(client)
    paused = remote.ask(Question("Show B02", images=True))
    assert paused.status == "needs_confirmation" and paused.pending == {
        "run_id": 1,
        "well": "B02",
        "field": 1,
    }
    assert len(fake.calls) == 1
    answer = remote.resume(True)
    assert answer.status == "answered"
    blocks = fake.calls[1]["messages"][-1]["content"][0]["content"]
    assert any(b["type"] == "image" for b in blocks)


def test_admin_only_settings_and_cli(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    from hcs_studio.app import main
    from hcs_studio.config import load_settings
    from hcs_studio.security import accounts
    from hcs_studio.services import context
    from hcs_studio.services.assistant import configure_assistant

    accounts.create_account("anna", "analyst", password="anna password")
    accounts.adopt_session("anna", "analyst", "Anna")
    with pytest.raises(PermissionError):
        configure_assistant(context(), enabled=True)
    assert main(["assistant", "configure", "--enable"]) == 2
    accounts.sign_out()
    accounts.create_account("root-admin", "admin", password="admin password")
    accounts.adopt_session("root-admin", "admin", "Admin")
    assert main(["assistant", "configure", "--enable", "--budget", "5000", "--ids-only", "on"]) == 0
    settings = load_settings().assistant
    assert settings.enabled and settings.monthly_token_budget == 5000 and settings.compound_ids_only
    # the key goes to the keychain, never to settings.json
    import keyring

    stored: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(keyring, "set_password", lambda s, n, v: stored.__setitem__((s, n), v))
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "sk-ant-cli-test")
    assert main(["assistant", "set-key"]) == 0
    assert stored == {("hcs-studio", "anthropic/api_key"): "sk-ant-cli-test"}
    settings_text = (home / "home" / "settings.json").read_text(encoding="utf-8")
    assert "sk-ant-cli-test" not in settings_text
    capsys.readouterr()
    assert main(["assistant", "preview", "Which drug is most potent?"]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert "Compound names are replaced by codes" in preview["system"][0]["text"]
    accounts.sign_out()


def test_cli_ask_shows_work(home: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    from hcs_studio import assistant as package
    from hcs_studio.app import main

    _enable()
    monkeypatch.setattr(package, "make_client", lambda settings: Scripted())
    assert main(["assistant", "ask", "How many plates?"]) == 0
    out = capsys.readouterr().out
    assert "There is 1 plate" in out and "SQL: SELECT COUNT(*) AS n FROM plates" in out


@pytest.mark.gui
def test_assistant_tab(home: Path, qtbot: Any, inline_tasks: Any) -> None:
    pytest.importorskip("PySide6")
    from PIL import Image

    from hcs_studio.gui.assistant_panel import AssistantPanel

    runs = sorted((home / "runs").glob("REG-*"))
    previews = runs[0] / "wells" / "B02" / "previews"
    previews.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), (0, 0, 200)).save(previews / "f001.png")
    _enable(allow_images=True)
    fake = Scripted()
    panel = AssistantPanel(LocalAssistant(client=fake))
    qtbot.addWidget(panel)
    panel.refresh_status()
    assert panel.ask_button.isEnabled() and "Tokens this month" in panel.status_label.text()
    panel.question.setPlainText("How many plates?")
    panel.ask()
    assert panel.answers and panel.answers[-1].status == "answered"
    assert "There is 1 plate" in panel.answer_view.toPlainText()
    item = panel.work.topLevelItem(0)
    sql_item = item.child(0) if item is not None else None
    assert item is not None and sql_item is not None
    assert item.text(0) == "1. run_query" and sql_item.text(1).startswith("SELECT COUNT")
    # the data preview opens without sending anything
    calls = len(fake.calls)
    panel.preview()
    assert panel.preview_dialog.isVisible() and len(fake.calls) == calls
    # an image needs the user's confirmation; declined → not sent
    panel.assistant = LocalAssistant(client=Scripted(image=True))
    asked: list[dict[str, Any]] = []

    def decline(request: dict[str, Any]) -> bool:
        asked.append(request)
        return False

    panel.confirm = decline
    panel.images.setEnabled(True)
    panel.images.setChecked(True)
    panel.question.setPlainText("Show B02")
    panel.ask()
    assert asked == [{"run_id": 1, "well": "B02", "field": 1}]
    assert panel.answers[-1].steps[-1].summary == "image declined"
    # UI actions: a chart drawn locally, highlights forwarded to the main window
    forwarded: list[dict[str, Any]] = []
    panel.ui_action.connect(forwarded.append)
    panel.handle_action({"action": "highlight_wells", "run_id": 1, "wells": ["B02"]})
    panel.handle_action(
        {
            "action": "chart",
            "title": "EC50",
            "kind": "bar",
            "x": "compound",
            "y": "ec50",
            "group": None,
            "log_x": False,
            "columns": ["compound", "ec50"],
            "rows": [["Drug A", 1e-6], ["Drug B", 3e-7]],
            "sql": "SELECT …",
        }
    )
    assert forwarded == [{"action": "highlight_wells", "run_id": 1, "wells": ["B02"]}]
    assert panel.charts and panel.charts[-1].isVisible()
    panel.assistant.close()


@pytest.mark.gui
def test_assistant_settings_dialog(home: Path, qtbot: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("PySide6")
    import keyring

    from hcs_studio.config import load_settings
    from hcs_studio.gui.assistant_panel import AssistantSettingsDialog

    stored: dict[str, str] = {}
    monkeypatch.setattr(keyring, "set_password", lambda s, n, v: stored.__setitem__(n, v))
    dialog = AssistantSettingsDialog()
    qtbot.addWidget(dialog)
    dialog.enabled.setChecked(True)
    dialog.budget.setValue(300_000)
    dialog.key.setText("sk-ant-dialog-test")
    dialog.save()
    settings = load_settings().assistant
    assert settings.enabled and settings.monthly_token_budget == 300_000
    assert stored == {"anthropic/api_key": "sk-ant-dialog-test"} and dialog.key.text() == ""


def test_local_assistant_new_snapshot_per_privacy(home: Path) -> None:
    _enable()
    fake = Scripted()
    assistant = LocalAssistant(client=fake)
    assistant.ask(Question("one"))
    first = assistant.conversation
    assistant.ask(Question("two"))
    assert assistant.conversation is first
    assistant.ask(Question("three", privacy=Privacy(compound_ids_only=True)))
    assert assistant.conversation is not first
    assert assistant.reveal("CPD-0001") != "CPD-0001"
    assistant.close()
