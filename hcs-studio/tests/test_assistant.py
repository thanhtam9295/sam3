"""Phase 28: the lab assistant against a mocked Claude client — curated views and privacy,
read-only SQL, the tool loop with "show work", the image confirmation, budget, audit, and a
payload log proving nothing beyond the curated views leaves the app."""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hcs_studio.assistant import (
    AssistantDisabled,
    BudgetExceeded,
    PayloadLog,
    Privacy,
    Question,
    open_conversation,
)
from hcs_studio.assistant.conversation import Conversation, unsupported_numbers
from hcs_studio.assistant.sql import QueryRejected, QueryTimeout, ReadOnlyDatabase
from hcs_studio.assistant.tools import TOOL_NAMES, tool_definitions
from hcs_studio.assistant.views import VIEWS, build_snapshot, scrub
from hcs_studio.config import AssistantSettings
from hcs_studio.services import context

SECRETS = {
    "owner": "Dr Hidden Owner",
    "operator": "Olga Operator",
    "notes": "CONFIDENTIAL lab note",
    "plate_notes": "plate note do-not-send",
    "smiles": "C1=CC=CC=C1SECRET",
    "supplier": "SecretSupplier Inc",
}


@pytest.fixture(scope="module")
def registry(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    """Two synthetic dose-response/combination runs, indexed, plus planted sensitive data."""
    from sqlalchemy import select

    from hcs_studio.db import session_scope
    from hcs_studio.db.catalog import Compound, Experiment, Plate, Project, Run
    from hcs_studio.db.engine import dispose_engines
    from hcs_studio.db.scanner import scan
    from hcs_studio.synthetic import registry_runs

    home = tmp_path_factory.mktemp("assistant-home")
    patch = pytest.MonkeyPatch()
    patch.setenv("HCS_STUDIO_HOME", str(home))
    patch.delenv("HCS_DATABASE_URL", raising=False)
    dispose_engines()
    runs = registry_runs(home / "runs", n_runs=2)
    manifest_path = runs[0] / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pipeline"]["steps"] = [
        {"module": "segment_classical", "params": {"channel": 1}},
        {"module": "measure_objects", "params": {"texture": False}},
        {"module": "classify_objects", "params": {"model_path": str(home / "models" / "m.joblib")}},
    ]
    manifest["operator"] = SECRETS["operator"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with session_scope() as session:
        scan(session, [home / "runs"])
        project = Project(name="Combination project", owner=SECRETS["owner"])
        experiment = Experiment(
            name="Combo screen",
            assay="viability",
            owner=SECRETS["owner"],
            notes=SECRETS["notes"],
            purpose="find synergy",
            project=project,
        )
        session.add_all([project, experiment])
        session.flush()
        for plate in session.scalars(select(Plate)):
            plate.experiment_id = experiment.id
            plate.notes = SECRETS["plate_notes"]
        for run in session.scalars(select(Run)):
            run.actor = SECRETS["operator"]
        for compound in session.scalars(select(Compound)):
            compound.smiles = SECRETS["smiles"]
            compound.supplier = SECRETS["supplier"]
            compound.moa = "kinase inhibitor" if compound.name == "Drug A" else "other"
    yield {"home": home, "runs": runs}
    dispose_engines()
    patch.undo()


def settings(**changes: Any) -> AssistantSettings:
    return AssistantSettings(**{"enabled": True, "allow_images": True, **changes})


# --- a scripted stand-in for anthropic.Anthropic


def text(value: str) -> dict[str, Any]:
    return {"type": "text", "text": value}


def tool_use(call_id: str, name: str, data: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool_use", "id": call_id, "name": name, "input": data}


def response(
    content: list[dict[str, Any]], stop: str, model: str = "claude-opus-5-5"
) -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop,
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 100,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 2000,
        },
    }


class FakeMessages:
    def __init__(self, script: Callable[[dict[str, Any], int], dict[str, Any]]) -> None:
        self.script = script
        self.calls: list[dict[str, Any]] = []

    def create(self, **body: Any) -> dict[str, Any]:
        self.calls.append(copy.deepcopy(body))
        return self.script(body, len(self.calls))


class FakeClient:
    def __init__(self, script: Callable[[dict[str, Any], int], dict[str, Any]]) -> None:
        self.messages = FakeMessages(script)
        self.beta = SimpleNamespace(messages=self.messages)


def scripted(*turns: dict[str, Any]) -> FakeClient:
    """A client answering with ``turns`` in order (the last one repeats)."""
    return FakeClient(lambda body, n: turns[min(n, len(turns)) - 1])


def last_tool_results(body: dict[str, Any]) -> list[dict[str, Any]]:
    content = body["messages"][-1]["content"]
    return [c for c in content if isinstance(c, dict) and c.get("type") == "tool_result"]


# --- curated views and privacy


def test_snapshot_holds_only_curated_views(registry: dict[str, Any], tmp_path: Path) -> None:
    import sqlite3

    snapshot = build_snapshot(tmp_path / "v.sqlite")
    db = sqlite3.connect(snapshot.path)
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert tables == {v.name for v in VIEWS} | {"_snapshot"}
    dump = "\n".join(db.iterdump())
    for value in SECRETS.values():
        assert value not in dump
    assert str(registry["home"]) not in dump  # no paths
    assert snapshot.counts["combination_scores"] > 0 and snapshot.counts["wells"] == 616
    # the run folders are known locally (manifest tool) but never written to the snapshot
    assert all(str(registry["home"]) in f or "runs" in f for f in snapshot.run_folders.values())
    db.close()
    with_notes = build_snapshot(tmp_path / "n.sqlite", Privacy(include_notes=True))
    dump = "\n".join(sqlite3.connect(with_notes.path).iterdump())
    assert SECRETS["notes"] in dump and SECRETS["plate_notes"] in dump
    assert SECRETS["owner"] not in dump and SECRETS["smiles"] not in dump


def test_compound_ids_only(registry: dict[str, Any], tmp_path: Path) -> None:
    import sqlite3

    snapshot = build_snapshot(tmp_path / "c.sqlite", Privacy(compound_ids_only=True), seed=3)
    dump = "\n".join(sqlite3.connect(snapshot.path).iterdump())
    for name in ("Drug A", "Drug B", "Drug C", "Drug D"):
        assert name not in dump
    codes = snapshot.pseudonyms.code
    assert set(codes) == {"Drug A", "Drug B", "Drug C", "Drug D"}
    assert snapshot.pseudonyms.reveal(f"{codes['Drug B']} is potent") == "Drug B is potent"
    assert scrub({"text": "Drug A at 1 µM"}, snapshot.pseudonyms) == {
        "text": f"{codes['Drug A']} at 1 µM"
    }


def test_scrub_removes_paths_and_people() -> None:
    value = {
        "plate": {"path": "/mnt/x", "barcode": "B1", "operator": "Olga"},
        "params": {
            "model": "registry:nuclei@3",
            "model_path": "C:\\models\\a.joblib",
            "other": "/home/me/file.txt",
            "threshold": 0.5,
        },
        "steps": ["\\\\server\\share\\x", "fine text"],
    }
    assert scrub(value) == {
        "plate": {"barcode": "B1"},
        "params": {"model": "registry:nuclei@3", "other": "[path]", "threshold": 0.5},
        "steps": ["[path]", "fine text"],
    }


# --- read-only SQL


def test_run_query_rejects_writes_and_other_tables(
    registry: dict[str, Any], tmp_path: Path
) -> None:
    snapshot = build_snapshot(tmp_path / "v.sqlite")
    db = ReadOnlyDatabase(snapshot.path, max_rows=5, timeout_s=1.0)
    for statement in (
        "DELETE FROM runs",
        "UPDATE runs SET status = 'x'",
        "INSERT INTO runs (run_id) VALUES (99)",
        "DROP TABLE runs",
        "CREATE TABLE x (a)",
        "ATTACH DATABASE 'other.sqlite' AS other",
        "PRAGMA writable_schema = 1",
        "SELECT 1; DELETE FROM runs",
        "WITH x AS (SELECT 1) DELETE FROM runs",
        "SELECT * FROM sqlite_master",
        "SELECT * FROM _snapshot",
        "SELECT load_extension('evil')",
        "VACUUM",
    ):
        with pytest.raises(QueryRejected):
            db.query(statement)
    # the connection itself refuses writes, whatever the statement looks like
    import sqlite3

    with pytest.raises(sqlite3.DatabaseError):
        db.db.set_authorizer(None)
        db.db.execute("DELETE FROM runs")
    db.db.set_authorizer(db._authorize)
    assert db.query("SELECT COUNT(*) AS n FROM runs").rows == [[2]]
    many = db.query("SELECT well FROM wells")
    assert len(many.rows) == 5 and many.truncated
    with pytest.raises(QueryTimeout):
        db.query(
            "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) SELECT COUNT(*) FROM n"
        )
    db.close()


# --- the conversation loop


def test_question_with_a_query_shows_its_work(registry: dict[str, Any]) -> None:
    from hcs_studio.assistant.usage import usage_of
    from hcs_studio.services import recent_events

    sql = (
        "SELECT cell_line, drug_a, drug_b, ROUND(mean, 2) AS zip FROM combination_scores "
        "WHERE model = 'zip' AND mean > 10 AND run_id = 1"
    )
    client = scripted(
        response(
            [
                {"type": "thinking", "thinking": "", "signature": "sig"},
                tool_use("t1", "run_query", {"sql": sql, "purpose": "ZIP above 10"}),
            ],
            "tool_use",
        ),
        response([text("In run 1, Drug A + Drug B in MCF7 had a ZIP score of 10.95.")], "end_turn"),
    )
    ctx = context()
    before = usage_of(ctx.actor.id)["requests"]
    conversation = open_conversation(ctx, client=client, settings=settings())
    answer = conversation.ask("Which combinations had a ZIP score above 10 in run 1?")
    assert answer.status == "answered" and "10.95" in answer.text
    assert [s.tool for s in answer.steps] == ["run_query"]
    assert answer.steps[0].sql == [sql] and "1 rows" in answer.steps[0].summary
    assert answer.unsupported_numbers == []
    second = client.messages.calls[1]
    result = last_tool_results(second)[0]
    assert result["tool_use_id"] == "t1"
    assert json.loads(result["content"])["rows"] == [["MCF7", "Drug A", "Drug B", 10.95]]
    # the thinking block is passed back unchanged with the assistant turn
    assert second["messages"][1]["content"][0]["type"] == "thinking"
    # prompt caching: stable system prompt + tools with a breakpoint, automatic caching on
    first = client.messages.calls[0]
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert first["cache_control"] == {"type": "ephemeral"}
    assert first["system"] == second["system"] and first["tools"] == second["tools"]
    assert "combination_scores" in first["system"][0]["text"]
    # Opus: server-side refusal fallback, effort from the settings, no thinking parameter
    assert first["betas"] == ["server-side-fallback-2026-07-01"] and first["fallbacks"] == "default"
    assert first["output_config"] == {"effort": "medium"} and "thinking" not in first
    assert usage_of(ctx.actor.id)["requests"] == before + 2
    kinds = [e["type"] for e in recent_events(10) if e["object_id"] == conversation.id]
    assert sorted(kinds) == ["assistant.answer", "assistant.question", "assistant.tool"]
    events = {e["type"]: e for e in recent_events(10) if e["object_id"] == conversation.id}
    assert events["assistant.question"]["after"]["question"].startswith("Which combinations")
    assert events["assistant.tool"]["after"]["sql"] == [sql]
    assert "10.95" in events["assistant.answer"]["after"]["answer"]
    conversation.close()


def test_unsupported_numbers_are_flagged(registry: dict[str, Any]) -> None:
    client = scripted(
        response(
            [
                tool_use(
                    "t1", "run_query", {"sql": "SELECT COUNT(*) AS n FROM runs", "purpose": "runs"}
                )
            ],
            "tool_use",
        ),
        response([text("There are 2 runs; the mean Z′ was 0.73.")], "end_turn"),
    )
    conversation = open_conversation(context(), client=client, settings=settings())
    answer = conversation.ask("How many runs?")
    assert answer.unsupported_numbers == ["0.73"]
    conversation.close()


def test_grounding_check_allows_rounding_and_units() -> None:
    evidence = ['{"rows":[[9.97104e-07, 0.5123, -6.0013]]}']
    assert unsupported_numbers("EC50 0.997 µM (997 nM), 51.2 %, log −6.00", evidence) == []
    assert unsupported_numbers("EC50 3.5 µM", evidence) == ["3.5"]
    assert unsupported_numbers("plates 1 to 4 of 384 wells", [], "384 wells") == []


def test_tool_errors_go_back_to_the_model(registry: dict[str, Any]) -> None:
    client = scripted(
        response(
            [
                tool_use("t1", "run_query", {"sql": "DELETE FROM runs", "purpose": "x"}),
                tool_use("t2", "get_plate_layout", {"plate_id": 999}),
                tool_use("t3", "nonexistent_tool", {}),
            ],
            "tool_use",
        ),
        response([text("I cannot change data.")], "end_turn"),
    )
    conversation = open_conversation(context(), client=client, settings=settings())
    answer = conversation.ask("Delete the runs")
    results = last_tool_results(client.messages.calls[1])
    assert [r.get("is_error") for r in results] == [True, True, True]
    assert "Only SELECT" in results[0]["content"]
    assert all(s.is_error for s in answer.steps)
    from hcs_studio.db import session_scope
    from hcs_studio.db.catalog import Run

    with session_scope() as session:
        assert session.query(Run).count() == 2
    conversation.close()


def test_data_tools(registry: dict[str, Any]) -> None:
    calls = [
        tool_use("a", "get_schema", {}),
        tool_use("b", "get_plate_layout", {"plate_id": 1}),
        tool_use("c", "get_run_manifest", {"run_id": 1}),
        tool_use(
            "d",
            "get_dose_response",
            {
                "run_id": 1,
                "compound": "Drug A",
                "cell_line": "MCF7",
                "readout": "nuclei_count",
                "include_points": True,
            },
        ),
        tool_use(
            "e",
            "get_combination_scores",
            {
                "run_id": None,
                "cell_line": "MCF7",
                "model": "zip",
                "drug": "Drug B",
                "min_mean": 5.0,
            },
        ),
        tool_use("f", "get_qc_summary", {"run_id": 1}),
        tool_use("g", "list_readouts", {"run_id": 1}),
        tool_use(
            "h",
            "make_chart",
            {
                "title": "EC50s",
                "kind": "bar",
                "x": "compound",
                "y": "ec50_m",
                "group": "cell_line",
                "log_x": False,
                "sql": "SELECT compound, cell_line, ec50_m FROM dose_response WHERE run_id = 1",
            },
        ),
        tool_use(
            "i",
            "highlight_wells",
            {"run_id": 1, "plate_id": None, "wells": ["b02", "B03"], "note": "synergy"},
        ),
        tool_use("j", "open_comparison", {"run_id": 1, "wells": ["B02"]}),
    ]
    client = scripted(response(calls, "tool_use"), response([text("Done.")], "end_turn"))
    conversation = open_conversation(context(), client=client, settings=settings())
    answer = conversation.ask("Tell me about run 1")
    assert [s.tool for s in answer.steps] == [c["name"] for c in calls]
    assert not any(s.is_error for s in answer.steps), [s.summary for s in answer.steps]
    results = {r["tool_use_id"]: r["content"] for r in last_tool_results(client.messages.calls[1])}
    layout = json.loads(results["b"])
    assert layout["wells_annotated"] == 308 and layout["roles"]["negative_control"] >= 16
    assert layout["compounds"]["Drug A"]["max_m"] == pytest.approx(3e-5)
    manifest = json.loads(results["c"])
    assert manifest["plate"]["barcode"] == "REG-001" and manifest["status"] == "completed"
    assert "path" not in json.dumps(manifest["plate"])
    drc = json.loads(results["d"])
    assert drc["n_fits"] == 1 and drc["fits"][0]["ec50_m"] == pytest.approx(1e-6, rel=0.1)
    assert len(drc["points"]["rows"]) >= 20
    combos = json.loads(results["e"])
    assert {(s["drug_a"], s["drug_b"]) for s in combos["scores"]} == {("Drug A", "Drug B")}
    qc = json.loads(results["f"])
    assert qc["plate_qc"] and qc["plate_qc"][0]["z_prime"] is not None
    readouts = json.loads(results["g"])["readouts"]
    assert {r["readout"] for r in readouts} == {"nuclei_count", "pct_caspase_positive"}
    chart = json.loads(results["h"])
    assert chart == {"chart": "EC50s", "points": 6, "truncated": False}  # data stays local
    actions = {a["action"]: a for a in answer.ui}
    assert len(actions["chart"]["rows"]) == 6
    assert actions["highlight_wells"]["wells"] == ["B02", "B03"]
    assert actions["open_comparison"] == {
        "action": "open_comparison",
        "run_id": 1,
        "wells": ["B02"],
    }
    conversation.close()


def test_pipeline_proposal_and_methods_draft(registry: dict[str, Any]) -> None:
    manifest_path = registry["runs"][0] / "run_manifest.json"
    before = manifest_path.read_bytes()
    steps = json.loads(before)["pipeline"]["steps"]
    draft = {
        "paragraphs": [{"text": "Images were analysed.", "facts_used": ["pipeline"]}],
        "missing": ["objective"],
    }

    def script(body: dict[str, Any], n: int) -> dict[str, Any]:
        if "format" in body.get("output_config", {}):  # the structured-output draft call
            assert body["output_config"]["format"]["type"] == "json_schema"
            return response([text(json.dumps(draft))], "end_turn")
        if n == 1:
            return response(
                [
                    tool_use(
                        "p",
                        "propose_pipeline_change",
                        {
                            "run_id": 1,
                            "summary": "tighter threshold",
                            "changes": [
                                {
                                    "step": 1,
                                    "parameter": "not_a_parameter",
                                    "new_value": "1",
                                    "reason": "x",
                                },
                                {"step": 99, "parameter": "x", "new_value": "1", "reason": "x"},
                            ],
                        },
                    ),
                    tool_use("m", "draft_methods_text", {"run_id": 1, "style": "short"}),
                ],
                "tool_use",
            )
        return response([text("Proposed.")], "end_turn")

    client = FakeClient(script)
    conversation = open_conversation(context(), client=client, settings=settings())
    answer = conversation.ask("Suggest a change and draft methods")
    results = {r["tool_use_id"]: r for r in last_tool_results(client.messages.calls[-1])}
    assert len(steps) == 3 and results["p"].get("is_error")  # both changes invalid
    assert "no parameter" in results["p"]["content"] and "no step 99" in results["p"]["content"]
    assert json.loads(results["m"]["content"]) == draft
    assert manifest_path.read_bytes() == before  # nothing applied
    assert any(a["action"] == "methods_draft" for a in answer.ui)
    conversation.close()


def test_pipeline_proposal_diff(registry: dict[str, Any], tmp_path: Path) -> None:
    """A valid proposal returns old → new values; the run is untouched."""
    from hcs_studio.assistant.tools import ToolBox

    snapshot = build_snapshot(tmp_path / "v.sqlite")
    box = ToolBox(snapshot)
    manifest = json.loads((registry["runs"][0] / "run_manifest.json").read_text(encoding="utf-8"))
    steps = manifest["pipeline"]["steps"]
    module = steps[0]["module"]
    from hcs_studio.processing.base import MODULES, load_builtin_modules

    load_builtin_modules()
    parameter = next(iter(MODULES.get(module).Params.model_fields))
    outcome = box.run(
        "propose_pipeline_change",
        {
            "run_id": 1,
            "summary": "s",
            "changes": [{"step": 1, "parameter": parameter, "new_value": "null", "reason": "r"}],
        },
    )
    assert not outcome.is_error, outcome.content
    proposal = outcome.ui[0]
    assert proposal["action"] == "pipeline_proposal" and proposal["changes"][0]["module"] == module
    assert proposal["changes"][0]["new"] is None
    box.close()


# --- images need an explicit opt-in and a confirmation


def test_image_tool_needs_opt_in_and_confirmation(registry: dict[str, Any]) -> None:
    preview = registry["runs"][0] / "wells" / "B02" / "previews"
    preview.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.new("RGB", (600, 400), (200, 10, 10)).save(preview / "f001.png")
    call = tool_use("img", "get_image_thumbnail", {"run_id": 1, "well": "B02", "field": 1})
    # not offered by default, and refused if called anyway
    client = scripted(response([call], "tool_use"), response([text("No image.")], "end_turn"))
    conversation = open_conversation(context(), client=client, settings=settings())
    answer = conversation.ask("Show me B02")
    assert "get_image_thumbnail" not in [t["name"] for t in client.messages.calls[0]["tools"]]
    assert last_tool_results(client.messages.calls[1])[0]["is_error"]
    assert answer.status == "answered"
    conversation.close()
    # admins may forbid images entirely
    client = scripted(response([text("x")], "end_turn"))
    conversation = open_conversation(
        context(), client=client, settings=settings(allow_images=False)
    )
    conversation.ask(Question("Show me B02", images=True))
    assert "get_image_thumbnail" not in [t["name"] for t in client.messages.calls[0]["tools"]]
    conversation.close()
    # enabled for the question: the loop pauses until the user decides
    for approved in (False, True):
        client = scripted(response([call], "tool_use"), response([text("Red.")], "end_turn"))
        conversation = open_conversation(context(), client=client, settings=settings())
        paused = conversation.ask(Question("Show me B02", images=True))
        assert paused.status == "needs_confirmation"
        assert paused.pending == {"run_id": 1, "well": "B02", "field": 1}
        assert len(client.messages.calls) == 1  # nothing sent while waiting
        with pytest.raises(RuntimeError):
            conversation.ask("another question")
        answer = conversation.resume(approved)
        result = last_tool_results(client.messages.calls[1])[0]
        if approved:
            blocks = result["content"]
            image = next(b for b in blocks if b["type"] == "image")
            import base64
            import io

            sent = Image.open(io.BytesIO(base64.b64decode(image["source"]["data"])))
            assert sent.mode == "RGB" and max(sent.size) <= 256  # downsampled 8-bit
        else:
            assert "declined" in result["content"]
        assert answer.status == "answered"
        conversation.close()


# --- governance


def test_off_until_enabled_and_budget(registry: dict[str, Any]) -> None:
    from hcs_studio.assistant.usage import usage_of

    with pytest.raises(AssistantDisabled):
        open_conversation(context(), client=scripted(), settings=AssistantSettings())
    ctx = context()
    used = usage_of(ctx.actor.id)["weighted"]
    tight = settings(user_budgets={ctx.actor.id: used + 1})
    client = scripted(response([text("ok")], "end_turn"))
    conversation = open_conversation(ctx, client=client, settings=tight)
    conversation.ask("first")  # allowed: under the budget before the request
    with pytest.raises(BudgetExceeded):
        conversation.ask("second")
    conversation.close()


def test_strong_model_per_question(registry: dict[str, Any]) -> None:
    client = scripted(response([text("ok")], "end_turn", model="claude-fable-5-2"))
    conversation = open_conversation(context(), client=client, settings=settings())
    answer = conversation.ask(Question("hard question", strong=True))
    body = client.messages.calls[0]
    assert body["model"] == "claude-fable-5-2" and body["output_config"] == {"effort": "high"}
    assert answer.model == "claude-fable-5-2"
    conversation.close()
    client = scripted(response([text("ok")], "end_turn", model="claude-haiku-5-5"))
    conversation = open_conversation(
        context(), client=client, settings=settings(model="claude-haiku-5-5")
    )
    conversation.ask("easy")
    assert "betas" not in client.messages.calls[0]  # no server-side fallback for Haiku
    conversation.close()


def test_refusal_and_token_limit(registry: dict[str, Any]) -> None:
    client = scripted(response([], "refusal"))
    conversation = open_conversation(context(), client=client, settings=settings())
    assert conversation.ask("x").status == "refused"
    conversation.close()
    client = scripted(response([text("partial")], "max_tokens"))
    conversation = open_conversation(context(), client=client, settings=settings())
    answer = conversation.ask("x")
    assert answer.status == "incomplete" and answer.text == "partial"
    conversation.close()


def test_strict_tool_schemas() -> None:
    tools = tool_definitions(images=True)
    assert [t["name"] for t in tools] == list(TOOL_NAMES)
    for tool in tools:
        schema = tool["input_schema"]
        assert tool["strict"] is True and schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
    assert "strict" not in tool_definitions(strict=False)[0]


# --- the payload log: nothing beyond the curated views leaves the app


def _all_tools_script(body: dict[str, Any], n: int) -> dict[str, Any]:
    if "format" in body.get("output_config", {}):
        return response([text('{"paragraphs": [], "missing": []}')], "end_turn")
    if n == 1:
        return response(
            [
                tool_use("a", "get_schema", {}),
                tool_use("b", "get_plate_layout", {"plate_id": 1}),
                tool_use("c", "get_run_manifest", {"run_id": 1}),
                tool_use(
                    "d",
                    "get_dose_response",
                    {
                        "run_id": None,
                        "compound": None,
                        "cell_line": None,
                        "readout": None,
                        "include_points": False,
                    },
                ),
                tool_use(
                    "e",
                    "get_combination_scores",
                    {
                        "run_id": None,
                        "cell_line": None,
                        "model": None,
                        "drug": None,
                        "min_mean": None,
                    },
                ),
                tool_use("f", "get_qc_summary", {"run_id": 1}),
                tool_use("g", "list_readouts", {"run_id": None}),
                tool_use("h", "run_query", {"sql": "SELECT * FROM experiments", "purpose": "x"}),
                tool_use("i", "run_query", {"sql": "SELECT * FROM plates", "purpose": "x"}),
                tool_use("j", "run_query", {"sql": "SELECT * FROM compounds", "purpose": "x"}),
                tool_use("k", "draft_methods_text", {"run_id": 1, "style": "full"}),
            ],
            "tool_use",
        )
    return response([text("done")], "end_turn")


@pytest.mark.parametrize("ids_only", [False, True])
def test_payload_log_contains_only_curated_data(registry: dict[str, Any], ids_only: bool) -> None:
    log = PayloadLog()
    client = FakeClient(_all_tools_script)
    conversation = open_conversation(
        context(),
        Privacy(compound_ids_only=ids_only),
        client=client,
        settings=settings(),
        payload_log=log,
    )
    answer = conversation.ask("Summarise everything about run 1")
    assert not any(s.is_error for s in answer.steps), [s.summary for s in answer.steps]
    sent = log.text()
    assert len(log.entries) == len(client.messages.calls) == 3  # question, draft, answer
    for value in SECRETS.values():
        assert value not in sent, value
    home = str(registry["home"])
    assert home not in sent and home.replace("/", "\\\\") not in sent
    assert "registry.sqlite" not in sent and "run_manifest.json" not in sent
    import getpass

    user = getpass.getuser()
    if len(user) > 3:
        assert f'"{user}"' not in sent
    for name in ("Drug A", "Drug B", "Drug C", "Drug D"):
        assert (name in sent) is not ids_only
    # every table a query or tool touched is a curated view
    for call in client.messages.calls:
        for message in call["messages"]:
            if message["role"] != "assistant":
                continue
            for block in message["content"]:
                if block.get("name") == "run_query":
                    tables = {
                        w.strip('"').split()[0] for w in block["input"]["sql"].split("FROM")[1:]
                    }
                    assert tables <= {v.name for v in VIEWS}
    conversation.close()


def test_preview_shows_what_will_be_sent(registry: dict[str, Any]) -> None:
    conversation = open_conversation(
        context(), Privacy(compound_ids_only=True), client=scripted(), settings=settings()
    )
    preview = conversation.preview(Question("Which drug is most potent?"))
    assert preview["messages"] == [{"role": "user", "content": "Which drug is most potent?"}]
    assert "CPD-0001" in preview["system"][0]["text"] or "codes" in preview["system"][0]["text"]
    assert conversation.messages == []  # previewing sends and stores nothing
    conversation.close()


def test_api_key_comes_from_the_keychain(
    registry: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("anthropic")
    from hcs_studio.assistant import AssistantUnavailable, make_client

    monkeypatch.delenv("HCS_SECRET_ANTHROPIC_API_KEY", raising=False)
    import keyring

    monkeypatch.setattr(keyring, "get_password", lambda *a: None)
    with pytest.raises(AssistantUnavailable):
        make_client(settings())
    monkeypatch.setenv("HCS_SECRET_ANTHROPIC_API_KEY", "sk-ant-test-not-real")
    client = make_client(settings())
    assert client.api_key == "sk-ant-test-not-real"
    assert os.environ["HCS_SECRET_ANTHROPIC_API_KEY"] not in json.dumps(settings().model_dump())


def test_conversation_is_reusable_across_questions(
    registry: dict[str, Any], tmp_path: Path
) -> None:
    snapshot = build_snapshot(tmp_path / "v.sqlite")
    client = scripted(response([text("one")], "end_turn"))
    events: list[tuple[str, dict[str, Any]]] = []
    conversation = Conversation(
        client, snapshot, settings(), audit=lambda k, d: events.append((k, d)), count_usage=False
    )
    conversation.ask("first")
    conversation.ask("second")
    body = client.messages.calls[1]
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    assert [k for k, _ in events] == ["question", "answer", "question", "answer"]
    conversation.close()
