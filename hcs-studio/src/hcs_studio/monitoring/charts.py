"""Levey–Jennings control charts with configurable rules.

A chart is a time-ordered series of one value per run (Z′, a control mean, a control CV, the
log10 IC50 of a reference compound …). Its centre line and SD come from a **baseline**: the
first ``baseline_n`` points (default 20), or fixed values. Points are compared with
centre ± k·SD by the rules; every point a rule fires on is reported with the rule.

Rules (all configurable; names follow Westgard's notation):

- ``beyond`` — ``n`` of ``m`` consecutive points beyond ``k`` SD **on the same side**
  (1-of-1 beyond 3 SD = "1_3s"; 2 of 3 beyond 2 SD = "2of3_2s"),
- ``run`` — ``n`` consecutive points on the same side of the centre ("7x"),
- ``trend`` — ``n`` consecutive points each higher (or each lower) than the one before.

Event annotations (new reagent lot, cell vial, instrument service …) are attached to the
chart so a shift can be read against what changed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

RuleKind = Literal["beyond", "run", "trend"]


@dataclass(frozen=True)
class Rule:
    kind: RuleKind
    n: int = 1
    m: int = 1
    k: float = 3.0
    name: str = ""

    def __post_init__(self) -> None:
        if self.n < 1 or self.m < self.n:
            raise ValueError(f"Rule {self.label}: need 1 ≤ n ≤ m")
        if self.kind == "beyond" and not self.k > 0:
            raise ValueError("A 'beyond' rule needs k > 0 SD")
        if self.kind in ("run", "trend") and self.n < 2:
            raise ValueError("Run and trend rules need n ≥ 2")

    @property
    def label(self) -> str:
        if self.name:
            return self.name
        if self.kind == "beyond":
            return (
                f"{self.n} of {self.m} beyond {self.k:g} SD"
                if self.m > 1
                else f"beyond {self.k:g} SD"
            )
        if self.kind == "run":
            return f"{self.n} in a row on one side"
        return f"{self.n} in a row rising or falling"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "n": self.n, "m": self.m, "k": self.k, "name": self.name}

    @classmethod
    def parse(cls, text: str) -> Rule:
        """``1_3s``, ``2of3_2s``, ``4of5_1s``, ``7x`` (run), ``6t`` (trend)."""
        t = text.strip().lower()
        if t.endswith("x") and t[:-1].isdigit():
            return cls("run", int(t[:-1]), int(t[:-1]), name=text)
        if t.endswith("t") and t[:-1].isdigit():
            return cls("trend", int(t[:-1]), int(t[:-1]), name=text)
        if t.endswith("s") and "_" in t:
            count, k = t[:-1].split("_", 1)
            if "of" in count:
                n, m = (int(x) for x in count.split("of"))
            else:
                n = m = int(count)
            return cls("beyond", n, m, float(k), name=text)
        raise ValueError(f"Unknown rule {text!r} (e.g. 1_3s, 2of3_2s, 7x, 6t)")


DEFAULT_RULES = (Rule.parse("1_3s"), Rule.parse("2of3_2s"), Rule.parse("7x"))


@dataclass
class Violation:
    index: int
    rule: str
    value: float
    when: Any = None


@dataclass
class ControlChart:
    name: str
    points: pd.DataFrame  # when, value, label (+ run, barcode …), z
    centre: float
    sd: float
    baseline_n: int
    violations: list[Violation] = field(default_factory=list)
    events: pd.DataFrame = field(default_factory=pd.DataFrame)
    notes: list[str] = field(default_factory=list)

    def limits(self, k: float) -> tuple[float, float]:
        return self.centre - k * self.sd, self.centre + k * self.sd

    @property
    def flagged(self) -> list[int]:
        return sorted({v.index for v in self.violations})


def _side(z: float) -> int:
    return 0 if not math.isfinite(z) else (1 if z > 0 else (-1 if z < 0 else 0))


def evaluate(z: Sequence[float], rules: Sequence[Rule]) -> list[tuple[int, Rule]]:
    """(index, rule) for every point where a rule fires (the last point of the window)."""
    values = [float(v) for v in z]
    fired: list[tuple[int, Rule]] = []
    for rule in rules:
        for i in range(len(values)):
            if rule.kind == "beyond":
                if i + 1 < rule.n:
                    continue
                window = values[max(0, i + 1 - rule.m) : i + 1]
                # the newest point must itself be beyond the limit, and enough on its side
                side = _side(values[i])
                if side == 0 or abs(values[i]) <= rule.k:
                    continue
                hits = sum(1 for v in window if _side(v) == side and abs(v) > rule.k)
                if hits >= rule.n:
                    fired.append((i, rule))
            elif rule.kind == "run":
                if i + 1 < rule.n:
                    continue
                window = values[i + 1 - rule.n : i + 1]
                sides = {_side(v) for v in window}
                if len(sides) == 1 and 0 not in sides:
                    fired.append((i, rule))
            else:
                if i + 1 < rule.n:
                    continue
                window = values[i + 1 - rule.n : i + 1]
                if any(not math.isfinite(v) for v in window):
                    continue
                steps = np.diff(window)
                if (steps > 0).all() or (steps < 0).all():
                    fired.append((i, rule))
    return fired


def control_chart(
    points: pd.DataFrame,
    *,
    name: str = "",
    rules: Sequence[Rule] = DEFAULT_RULES,
    baseline_n: int = 20,
    centre: float | None = None,
    sd: float | None = None,
    events: pd.DataFrame | None = None,
) -> ControlChart:
    """``points``: one row per run with ``when`` (time) and ``value``; sorted here by time.
    ``centre``/``sd`` fix the limits; otherwise the first ``baseline_n`` finite points set
    them (mean and SD)."""
    if "when" not in points or "value" not in points:
        raise ValueError("Chart points need 'when' and 'value' columns")
    data = points.copy()
    data["_t"] = data["when"].map(_timestamp)
    data = data.sort_values("_t", kind="stable").drop(columns="_t").reset_index(drop=True)
    finite = data["value"].astype(float)
    finite = finite[np.isfinite(finite)]
    notes = []
    base = finite.iloc[:baseline_n]
    if centre is None or sd is None:
        if len(base) < 3:
            raise ValueError("A control chart needs at least 3 points for its limits")
        if len(base) < baseline_n:
            notes.append(
                f"Limits from {len(base)} points (fewer than the {baseline_n} asked for): "
                "they will move as runs are added"
            )
    centre = float(base.mean()) if centre is None else float(centre)
    sd = float(base.std(ddof=1)) if sd is None else float(sd)
    if not sd > 0:
        raise ValueError("The baseline has no spread (SD = 0); fix the limits by hand")
    data["z"] = (data["value"].astype(float) - centre) / sd
    values = data["value"].astype(float).to_numpy()
    violations = [
        Violation(i, rule.label, float(values[i]), data["when"].iat[i])
        for i, rule in evaluate(data["z"].tolist(), rules)
    ]
    return ControlChart(
        name, data, centre, sd, min(len(base), baseline_n), violations,
        events if events is not None else pd.DataFrame(), notes,
    )  # fmt: skip


def _timestamp(value: Any) -> float:
    if isinstance(value, datetime):
        return value.timestamp() if value.tzinfo else value.replace(tzinfo=None).timestamp()
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day).timestamp()
    if isinstance(value, pd.Timestamp):
        return float(value.timestamp())
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def events_for(
    events: pd.DataFrame, *, assay: str = "", instrument: str = "", cell_line: str = ""
) -> pd.DataFrame:
    """Event annotations that concern a chart: an empty field in the event matches anything."""
    if events.empty:
        return events
    keep = pd.Series(True, index=events.index)
    for column, wanted in (("assay", assay), ("instrument", instrument), ("cell_line", cell_line)):
        if wanted and column in events:
            keep &= events[column].fillna("").isin(["", wanted])
    return events[keep].reset_index(drop=True)
