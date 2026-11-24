"""The lab assistant's conversation loop (Phase 28) on the Claude Messages API.

The loop is HCS Studio's own (not the SDK's tool runner) because every request and every tool
call must pass through the app: requests are logged (:class:`PayloadLog`), each tool call is
audited and shown ("show work"), the image tool pauses for the user's confirmation, and token
usage counts against the user's monthly budget.

Prompt caching: the tool list and the system prompt (instructions + curated schema) are
stable, with a cache breakpoint on the system prompt; automatic caching covers the growing
conversation. Nothing per-request (time, user name, IDs) goes into the system prompt.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from hcs_studio.assistant.tools import (
    NeedsConfirmation,
    ToolBox,
    ToolOutcome,
    tool_definitions,
)
from hcs_studio.assistant.views import Privacy, Snapshot, schema_text

#: models that take the server-side refusal fallback (``fallbacks: "default"``)
FALLBACK_MODELS = (
    "claude-opus-5-5",
    "claude-fable-5-2",
    "claude-fable-5-1",
    "claude-opus-5",
    "claude-sonnet-5-5",
)
FALLBACK_BETA = "server-side-fallback-2026-07-01"

INSTRUCTIONS = """\
You are the lab assistant of HCS Studio, a high-content screening analysis application. You \
answer questions about the lab's results using the tools, which read a curated, read-only \
copy of the results registry.

Rules:
- Every number you state must come from a tool result in this conversation. Never estimate, \
recall or invent numbers. If the data needed is missing, say so plainly and name what is \
missing instead of guessing.
- Prefer one SQL query that aggregates over reading many rows. Check readout IDs in the \
readouts view before using them.
- Concentrations in the views are in mol/L (M); give them in convenient units (nM, µM) and \
always state the unit.
- Say which runs or plates (IDs, barcodes) your numbers come from, and mention QC problems \
(QC flags, failed image QC, few points) that weaken them.
- You cannot change anything. Pipeline changes are proposals (propose_pipeline_change) that \
the user applies; highlight_wells and open_comparison only change the user's view.
- Be brief: the answer first, then the supporting numbers, then caveats.

The curated views (SQLite) you can query:
"""


class AssistantDisabled(RuntimeError):
    """The assistant is off (an admin enables it in settings)."""


class AssistantUnavailable(RuntimeError):
    """No API key, or the SDK is not installed."""


@dataclass
class Question:
    text: str
    #: use the stronger model (``strong_model``) for this question
    strong: bool = False
    privacy: Privacy = field(default_factory=Privacy)
    #: offer the thumbnail tool for this question (each image still needs confirmation)
    images: bool = False


@dataclass
class Step:
    """One tool call, as shown under the answer ("show work")."""

    tool: str
    input: dict[str, Any]
    summary: str
    sql: list[str]
    is_error: bool
    elapsed_s: float


@dataclass
class Answer:
    conversation_id: str
    #: answered | needs_confirmation | refused | incomplete | error
    status: str
    text: str = ""
    #: ``text`` with compound codes replaced by names (for the screen only)
    display_text: str = ""
    steps: list[Step] = field(default_factory=list)
    ui: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    #: the thumbnail waiting for the user's confirmation
    pending: dict[str, Any] | None = None
    #: numbers in the answer that no tool result contains (grounding check)
    unsupported_numbers: list[str] = field(default_factory=list)
    message: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class PayloadLog:
    """Every request body sent to the API, as JSON text (the "data that will be sent" panel
    and the test proving nothing beyond the curated views leaves the app)."""

    def __init__(self, keep: int = 200) -> None:
        self.keep = keep
        self.entries: list[str] = []

    def __call__(self, request: dict[str, Any]) -> None:
        self.entries.append(json.dumps(_plain(request), ensure_ascii=False, default=str))
        del self.entries[: -self.keep]

    def text(self) -> str:
        return "\n".join(self.entries)


def _plain(value: Any) -> Any:
    """SDK objects (content blocks) → plain JSON values."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _usage(response: Any) -> dict[str, int]:
    usage = _get(response, "usage") or {}
    return {
        "input_tokens": int(_get(usage, "input_tokens", 0) or 0),
        "output_tokens": int(_get(usage, "output_tokens", 0) or 0),
        "cache_creation_tokens": int(_get(usage, "cache_creation_input_tokens", 0) or 0),
        "cache_read_tokens": int(_get(usage, "cache_read_input_tokens", 0) or 0),
    }


_NUMBER = re.compile(r"(?<![A-Za-z0-9_.])[-−]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?(?![A-Za-z]?\d)")


def _numbers(text: str) -> list[str]:
    return [m.group(0).replace("−", "-") for m in _NUMBER.finditer(text)]


def _as_float(text: str) -> float | None:
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def unsupported_numbers(answer: str, evidence: list[str], question: str = "") -> list[str]:
    """Numbers in ``answer`` that appear neither in the tool results nor the question, allowing
    for rounding and unit scaling (M → mM/µM/nM/pM, fractions → %). Small integers (≤ 10)
    are ignored (counts of things named in the text, list numbering)."""
    pool: list[float] = []
    for text in [*evidence, question]:
        for n in _numbers(text):
            value = _as_float(n)
            if value is not None:
                pool.append(value)
    scaled = [v * f for v in pool for f in (1, 1e3, 1e6, 1e9, 1e12, 100) if v != 0]
    import math

    out = []
    for n in _numbers(answer):
        value = _as_float(n)
        if value is None or (value.is_integer() and abs(value) <= 10):
            continue
        decimals = len(n.split(".")[1]) if "." in n else 0
        tolerance = 0.5 * 10**-decimals
        candidates = [*pool, *scaled]
        ok = any(abs(c - value) <= tolerance + 1e-12 for c in candidates)
        if not ok:  # rounded to significant digits
            digits = max(1, len(n.replace("-", "").replace(".", "").lstrip("0")))
            ok = any(
                c != 0 and math.isclose(float(f"{c:.{digits}g}"), value, rel_tol=1e-9)
                for c in candidates
            )
        if not ok and abs(value) > 0:  # log10 of a value (log EC50)
            ok = any(c > 0 and abs(math.log10(c) - value) <= tolerance for c in pool)
        if not ok:
            out.append(n)
    return out


class Conversation:
    """One conversation with the assistant (a series of questions with shared context).

    ``client`` is an ``anthropic.Anthropic`` (or a test double with ``.beta.messages.create``
    and ``.messages.create``). ``audit`` receives (kind, details) for every question, tool
    call and answer; the default writes service events (the Phase 26 audit trail).
    """

    def __init__(
        self,
        client: Any,
        snapshot: Snapshot,
        settings: Any,
        *,
        user: str = "",
        audit: Callable[[str, dict[str, Any]], None] | None = None,
        payload_log: PayloadLog | None = None,
        count_usage: bool = True,
        conversation_id: str | None = None,
    ) -> None:
        self.client = client
        self.snapshot = snapshot
        self.settings = settings
        self.user = user
        self.id = conversation_id or uuid.uuid4().hex
        self.audit = audit or (lambda kind, details: None)
        self.payload_log = payload_log or PayloadLog()
        self.count_usage = count_usage
        self.toolbox = ToolBox(
            snapshot,
            max_rows=settings.max_rows,
            timeout_s=settings.query_timeout_s,
            drafter=self._draft_methods,
        )
        self.messages: list[dict[str, Any]] = []
        self.system = [
            {
                "type": "text",
                "text": INSTRUCTIONS + schema_text(snapshot.privacy),
                "cache_control": {"type": "ephemeral"},
            }
        ]
        self._question: Question | None = None
        self._answer: Answer | None = None
        #: tool_use blocks of the last response still to run, and results so far
        self._pending_blocks: list[Any] = []
        self._results: list[dict[str, Any]] = []
        self._evidence: list[str] = []
        self._turns = 0

    # --- requests -------------------------------------------------------------------------

    def _model(self, question: Question) -> tuple[str, str]:
        s = self.settings
        return (s.strong_model, s.strong_effort) if question.strong else (s.model, s.effort)

    def request(self, question: Question) -> dict[str, Any]:
        """The next request body (also what the "data that will be sent" preview shows)."""
        model, effort = self._model(question)
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": self.settings.max_tokens,
            "system": self.system,
            "tools": tool_definitions(
                images=question.images and self.settings.allow_images,
                strict=self.settings.strict_tools,
            ),
            "messages": self.messages,
            "output_config": {"effort": effort},
            "cache_control": {"type": "ephemeral"},
        }
        if self.settings.refusal_fallback and model in FALLBACK_MODELS:
            body["betas"] = [FALLBACK_BETA]
            body["fallbacks"] = "default"
        return body

    def preview(self, question: Question) -> dict[str, Any]:
        """The request a question would send first (nothing is sent)."""
        messages = [*self.messages, {"role": "user", "content": question.text}]
        saved, self.messages = self.messages, messages
        try:
            return dict(_plain(self.request(question)))
        finally:
            self.messages = saved

    def _send(self, body: dict[str, Any]) -> Any:
        self.payload_log(body)
        if "betas" in body:
            return self.client.beta.messages.create(**body)
        return self.client.messages.create(**body)

    def _account(self, model: str, response: Any) -> dict[str, int]:
        usage = _usage(response)
        if self.count_usage and self.user:
            from hcs_studio.assistant.usage import record_usage

            record_usage(self.user, str(_get(response, "model", model) or model), usage)
        return usage

    # --- the loop -------------------------------------------------------------------------

    def ask(self, question: Question | str) -> Answer:
        """Ask a question; returns the answer, or a pause for an image confirmation."""
        if not getattr(self.settings, "enabled", False):
            raise AssistantDisabled(
                "The lab assistant is off. An admin can enable it in Settings → Assistant."
            )
        if isinstance(question, str):
            question = Question(question)
        if self._pending_blocks:
            raise RuntimeError("Answer the pending image confirmation first (resume)")
        if self.user:
            from hcs_studio.assistant.usage import check_budget

            check_budget(self.user, self.settings)
        self.toolbox.images_enabled = bool(question.images and self.settings.allow_images)
        self._turns = 0
        self._question = question
        model, _ = self._model(question)
        self._answer = Answer(conversation_id=self.id, status="answered", model=model)
        self._evidence = []
        self.audit(
            "question",
            {
                "conversation": self.id,
                "question": question.text,
                "model": model,
                "compound_ids_only": question.privacy.compound_ids_only,
                "include_notes": question.privacy.include_notes,
                "images": self.toolbox.images_enabled,
            },
        )
        self.messages.append({"role": "user", "content": question.text})
        return self._loop()

    def resume(self, approved: bool) -> Answer:
        """Continue after the user confirmed (or declined) sending an image."""
        if not self._pending_blocks or self._answer is None or self._answer.pending is None:
            raise RuntimeError("Nothing is waiting for a confirmation")
        request = self._answer.pending
        self.audit("thumbnail", {"conversation": self.id, **request, "approved": approved})
        key = (request["run_id"], request["well"], request["field"])
        block = self._pending_blocks[0]
        if approved:
            self.toolbox.confirmed.add(key)
        else:
            self._pending_blocks.pop(0)
            self._results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": _get(block, "id"),
                    "content": "The user declined to send this image. Answer without it.",
                }
            )
            self._record(block, ToolOutcome("declined", summary="image declined"), 0.0)
        self._answer.pending = None
        self._answer.status = "answered"
        return self._run_tools_then_loop()

    def _record(self, block: Any, outcome: ToolOutcome, elapsed: float) -> None:
        assert self._answer is not None
        name = str(_get(block, "name"))
        data = dict(_get(block, "input") or {})
        step = Step(name, data, outcome.summary, outcome.sql, outcome.is_error, elapsed)
        self._answer.steps.append(step)
        self._answer.ui.extend(outcome.ui)
        self.audit(
            "tool",
            {
                "conversation": self.id,
                "tool": name,
                "input": data,
                "summary": outcome.summary,
                "sql": outcome.sql,
                "is_error": outcome.is_error,
            },
        )

    def _run_tools_then_loop(self) -> Answer:
        assert self._answer is not None
        while self._pending_blocks:
            block = self._pending_blocks[0]
            started = time.monotonic()
            try:
                outcome = self.toolbox.run(
                    str(_get(block, "name")), dict(_get(block, "input") or {})
                )
            except NeedsConfirmation as pause:
                self._answer.status = "needs_confirmation"
                self._answer.pending = pause.request
                return self._answer
            self._pending_blocks.pop(0)
            self._record(block, outcome, time.monotonic() - started)
            if isinstance(outcome.content, str):
                self._evidence.append(outcome.content)
            result: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": _get(block, "id"),
                "content": outcome.content,
            }
            if outcome.is_error:
                result["is_error"] = True
            self._results.append(result)
        if self._results:
            self.messages.append({"role": "user", "content": self._results})
            self._results = []
        return self._loop()

    def _loop(self) -> Answer:
        assert self._answer is not None and self._question is not None
        answer, question = self._answer, self._question
        while self._turns < self.settings.max_turns:
            self._turns += 1
            body = self.request(question)
            response = self._send(body)
            usage = self._account(body["model"], response)
            for key, value in usage.items():
                answer.usage[key] = answer.usage.get(key, 0) + value
            answer.model = str(_get(response, "model", body["model"]) or body["model"])
            content = list(_get(response, "content") or [])
            stop = _get(response, "stop_reason")
            if stop == "refusal":
                answer.status = "refused"
                answer.message = "The model declined this request."
                return self._finish()
            self.messages.append({"role": "assistant", "content": content})
            if stop == "pause_turn":
                continue
            tool_blocks = [b for b in content if _get(b, "type") == "tool_use"]
            if stop == "tool_use" and tool_blocks:
                self._pending_blocks = tool_blocks
                self._results = []
                return self._run_tools_then_loop()
            answer.text = "\n\n".join(
                str(_get(b, "text")) for b in content if _get(b, "type") == "text"
            ).strip()
            if stop == "max_tokens":
                answer.status = "incomplete"
                answer.message = "The answer was cut off (token limit)."
            return self._finish()
        answer.status = "incomplete"
        answer.message = f"Stopped after {self.settings.max_turns} model turns."
        return self._finish()

    def _finish(self) -> Answer:
        assert self._answer is not None and self._question is not None
        answer = self._answer
        answer.unsupported_numbers = unsupported_numbers(
            answer.text, self._evidence, self._question.text
        )
        answer.display_text = self.snapshot.pseudonyms.reveal(answer.text)
        self.audit(
            "answer",
            {
                "conversation": self.id,
                "status": answer.status,
                "answer": answer.text,
                "tools": [s.tool for s in answer.steps],
                "usage": answer.usage,
                "model": answer.model,
                "unsupported_numbers": answer.unsupported_numbers,
            },
        )
        return answer

    # --- structured output: methods text --------------------------------------------------

    def _draft_methods(self, facts: dict[str, Any], style: str) -> dict[str, Any]:
        """A methods draft from curated manifest facts (structured output, same model)."""
        question = self._question or Question("")
        model, _ = self._model(question)
        schema = {
            "type": "object",
            "properties": {
                "paragraphs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "facts_used": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["text", "facts_used"],
                        "additionalProperties": False,
                    },
                },
                "missing": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["paragraphs", "missing"],
            "additionalProperties": False,
        }
        length = "one short paragraph" if style == "short" else "two to four paragraphs"
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": 4000,
            "system": "You write the methods section of a scientific paper from analysis "
            "records. Use only the facts given; never add instruments, settings or numbers that "
            "are not in them. List what a methods section would need but the facts lack under "
            "'missing'. For each paragraph, list the fact keys it uses.",
            "messages": [
                {
                    "role": "user",
                    "content": f"Write {length} describing the image analysis.\n\nFacts (JSON):\n"
                    + json.dumps(facts, ensure_ascii=False, default=str),
                }
            ],
            "output_config": {
                "effort": "low",
                "format": {"type": "json_schema", "schema": schema},
            },
        }
        response = self._send(body)
        usage = self._account(model, response)
        if self._answer is not None:
            for key, value in usage.items():
                self._answer.usage[key] = self._answer.usage.get(key, 0) + value
        if _get(response, "stop_reason") == "refusal":
            return {"paragraphs": [], "missing": ["the model declined to draft this text"]}
        text = next(
            (
                str(_get(b, "text"))
                for b in _get(response, "content") or []
                if _get(b, "type") == "text"
            ),
            "{}",
        )
        try:
            draft = json.loads(text)
        except ValueError:
            return {"paragraphs": [], "missing": ["the draft was not valid JSON"]}
        return dict(draft)

    def close(self) -> None:
        self.toolbox.close()


def tool_names(images: bool = False) -> list[str]:
    return [t["name"] for t in tool_definitions(images=images)]
