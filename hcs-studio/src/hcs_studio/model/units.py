"""Concentration units: parsing, conversion to base units and display.

Concentrations are stored in base units of their kind:

* molar → mol/L (M)
* mass → g/L (so 1 ng/mL = 1e-6 g/L)
* percent → volume fraction (1 % = 0.01)

Molar and mass concentrations convert into each other only with a molecular weight
(g/mol). Percent never converts to the others.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

Kind = Literal["molar", "mass", "percent"]


@dataclass(frozen=True)
class Unit:
    symbol: str
    kind: Kind
    #: Base units per one of this unit (e.g. nM → 1e-9).
    factor: float


UNITS: dict[str, Unit] = {
    u.symbol: u
    for u in (
        Unit("M", "molar", 1.0),
        Unit("mM", "molar", 1e-3),
        Unit("µM", "molar", 1e-6),
        Unit("nM", "molar", 1e-9),
        Unit("pM", "molar", 1e-12),
        Unit("fM", "molar", 1e-15),
        Unit("g/L", "mass", 1.0),
        Unit("mg/mL", "mass", 1.0),
        Unit("mg/L", "mass", 1e-3),
        Unit("µg/mL", "mass", 1e-3),
        Unit("µg/L", "mass", 1e-6),
        Unit("ng/mL", "mass", 1e-6),
        Unit("ng/L", "mass", 1e-9),
        Unit("pg/mL", "mass", 1e-9),
        Unit("%", "percent", 0.01),
    )
}
#: Units offered in drop-downs, per kind, smallest last.
CHOICES: dict[Kind, list[str]] = {
    "molar": ["M", "mM", "µM", "nM", "pM"],
    "mass": ["mg/mL", "µg/mL", "ng/mL", "pg/mL"],
    "percent": ["%"],
}
_ALIASES = {
    "um": "µM",
    "μm": "µM",  # Greek mu
    "micromolar": "µM",
    "nm": "nM",
    "nanomolar": "nM",
    "mm": "mM",
    "millimolar": "mM",
    "pm": "pM",
    "picomolar": "pM",
    "fm": "fM",
    "m": "M",
    "molar": "M",
    "mol/l": "M",
    "ug/ml": "µg/mL",
    "μg/ml": "µg/mL",
    "ug/l": "µg/L",
    "μg/l": "µg/L",
    "mcg/ml": "µg/mL",
    "%v/v": "%",
    "% v/v": "%",
    "percent": "%",
}


def parse_unit(text: str) -> Unit:
    """A unit from its symbol, case-insensitively for unambiguous spellings
    (``uM``, ``µM``, ``ug/mL``, ``ng/ml``, ``%``)."""
    cleaned = str(text).strip()
    if cleaned in UNITS:
        return UNITS[cleaned]
    lowered = cleaned.lower().replace(" ", "")
    if lowered in _ALIASES:
        return UNITS[_ALIASES[lowered]]
    for unit in UNITS.values():
        if unit.symbol.lower().replace("µ", "u") == lowered.replace("µ", "u").replace("μ", "u"):
            return unit
    known = ", ".join(UNITS)
    raise ValueError(f"Unknown concentration unit {text!r} (known: {known})")


def to_base(value: float, unit: str | Unit) -> float:
    """Value in ``unit`` → base units of its kind."""
    u = unit if isinstance(unit, Unit) else parse_unit(unit)
    return float(value) * u.factor


def from_base(value: float, unit: str | Unit) -> float:
    """Base units → value in ``unit``."""
    u = unit if isinstance(unit, Unit) else parse_unit(unit)
    return float(value) / u.factor


def convert(
    value: float, source: str | Unit, target: str | Unit, molecular_weight: float | None = None
) -> float:
    """Convert between units; molar ↔ mass needs ``molecular_weight`` in g/mol."""
    a = source if isinstance(source, Unit) else parse_unit(source)
    b = target if isinstance(target, Unit) else parse_unit(target)
    base = to_base(value, a)
    if a.kind == b.kind:
        return from_base(base, b)
    if {a.kind, b.kind} == {"molar", "mass"}:
        if not molecular_weight or molecular_weight <= 0:
            raise ValueError(f"Converting {a.symbol} to {b.symbol} needs the molecular weight")
        base = base * molecular_weight if a.kind == "molar" else base / molecular_weight
        return from_base(base, b)
    raise ValueError(f"{a.symbol} ({a.kind}) cannot be converted to {b.symbol} ({b.kind})")


def best_unit(base_value: float, kind: Kind) -> str:
    """The unit in which a base value reads as 1–999 (for display)."""
    choices = CHOICES[kind]
    if base_value <= 0 or not math.isfinite(base_value):
        return choices[0] if kind == "percent" else choices[2 if kind == "molar" else 1]
    for symbol in choices:
        if from_base(base_value, symbol) >= 1:
            return symbol
    return choices[-1]


def format_value(value: float) -> str:
    """A concentration for display: up to 4 significant digits, no trailing zeros."""
    if value == 0:
        return "0"
    text = f"{value:.4g}"
    return text if "e" not in text else f"{value:.3g}"


def format_concentration(base_value: float, unit: str) -> str:
    return f"{format_value(from_base(base_value, unit))} {parse_unit(unit).symbol}"
