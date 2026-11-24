"""Phase 28.5: the evaluation set. The harness and every ground-truth query are checked with an
"oracle" stand-in for Claude (it runs the item's SQL through run_query and answers with the
numbers in convenient units). The live evaluation (≥ 90 % against SQL ground truth) runs only
with ``HCS_ASSISTANT_EVAL=1`` and a key (``HCS_SECRET_ANTHROPIC_API_KEY`` or the keychain): it
costs tokens."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from hcs_studio.assistant.conversation import Answer, Step
from hcs_studio.assistant.evaluation import (
    ITEMS,
    EvalItem,
    build_eval_registry,
    grade,
    number_found,
    run_evaluation,
)
from hcs_studio.config import AssistantSettings


@pytest.fixture(scope="module")
def eval_home(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    from hcs_studio.db.engine import dispose_engines

    home = tmp_path_factory.mktemp("assistant-eval")
    patch = pytest.MonkeyPatch()
    patch.setenv("HCS_STUDIO_HOME", str(home / "home"))
    patch.delenv("HCS_DATABASE_URL", raising=False)
    dispose_engines()
    build_eval_registry(home)
    yield home
    dispose_engines()
    patch.undo()


def _format(value: Any) -> str:
    if isinstance(value, float) and 0 < abs(value) < 1e-3:
        return f"{value * 1e6:.3g} µM"  # concentrations as a person would write them
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


class Oracle:
    """Answers each evaluation question with its ground-truth SQL (two model turns)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.messages = self
        self.beta = self

    def create(self, **body: Any) -> dict[str, Any]:
        self.calls.append(body)
        messages = body["messages"]
        question = messages[0]["content"]
        item = next(i for i in ITEMS if i.question == question)
        usage = {"input_tokens": 10, "output_tokens": 5}
        if len(messages) == 1:
            call = {
                "type": "tool_use",
                "id": "q",
                "name": "run_query",
                "input": {"sql": item.sql, "purpose": item.id},
            }
            return {"content": [call], "stop_reason": "tool_use", "usage": usage}
        result = json.loads(messages[-1]["content"][0]["content"])
        values = [v for row in result["rows"] for v in row if v is not None]
        text = "From the registry: " + ", ".join(_format(v) for v in values) + "."
        return {
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "usage": usage,
        }


def test_ground_truth_and_grading_with_an_oracle(eval_home: Path) -> None:
    assert len(ITEMS) >= 30 and len({i.id for i in ITEMS}) == len(ITEMS)
    settings = AssistantSettings(enabled=True)
    report = run_evaluation(Oracle(), settings, eval_home / "work", log=lambda m: None)
    failed = [(r.id, r.note, r.answer) for r in report.results if not r.passed]
    assert failed == []
    assert report.score == 1.0 and report.all_show_work
    for result in report.results:  # every ground truth is non-trivial
        assert result.expected, result.id
    by_id = {r.id: r for r in report.results}
    assert by_id["plates"].expected == [3] and by_id["runs"].expected == [3]
    assert sorted(by_id["zip_all"].expected) == ["Drug A", "Drug B"]
    assert by_id["most_potent"].expected == ["Drug D"]
    data = report.write(eval_home / "work" / "eval.json")
    assert json.loads(data.read_text())["score"] == 1.0


def test_grading_rules() -> None:
    item = EvalItem("x", "q", "SELECT 1", "numbers")
    step = Step("run_query", {}, "", [], False, 0.0)
    answer = Answer("c", "answered", text="EC50 = 0.298 µM", steps=[step])
    assert grade(item, [2.97735e-07], answer) == (True, "ok")
    assert not grade(item, [3.5e-07], answer)[0]
    no_work = Answer("c", "answered", text="EC50 = 0.298 µM")
    assert grade(item, [2.97735e-07], no_work) == (False, "no tool call (the answer shows no work)")
    assert number_found(0.731, "Z′ = 0.73", 0.02) and number_found(12.4, "CV 12.4 %", 0.02)
    assert number_found(0.124, "12.4 %", 0.02)
    names = EvalItem("y", "q", "SELECT 1", "names")
    assert grade(names, ["Drug D"], Answer("c", "answered", text="drug d", steps=[step]))[0]


@pytest.mark.skipif(
    os.environ.get("HCS_ASSISTANT_EVAL") != "1",
    reason="live evaluation against the Claude API (costs tokens): set HCS_ASSISTANT_EVAL=1",
)
def test_live_evaluation(eval_home: Path) -> None:
    """Done-when: ≥ 90 % of answers match the SQL ground truth, every answer shows its work."""
    pytest.importorskip("anthropic")
    from hcs_studio.assistant import make_client

    settings = AssistantSettings(
        enabled=True, model=os.environ.get("HCS_ASSISTANT_MODEL", "claude-opus-5-5")
    )
    report = run_evaluation(make_client(settings), settings, eval_home / "live")
    path = report.write(Path(os.environ.get("HCS_ASSISTANT_EVAL_REPORT", eval_home / "live.json")))
    print(f"score {report.score:.0%}; report {path}")
    assert report.all_show_work
    assert report.score >= 0.9, [(r.id, r.note) for r in report.results if not r.passed]
