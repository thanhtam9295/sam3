"""Hit rules as expressions over well scores (Phase 23a).

A rule is a small boolean expression, e.g. ``"count_z <= -3 and not viability_pct < 50"``: names
are score columns of the wells table (``<readout alias>_<score>``: ``pct``, ``z``, ``b``,
``score`` …, and the raw readout under its alias), numbers, comparisons (``< <= > >= == !=``,
chained like ``-6 < a_z <= -3``), ``and``, ``or``, ``not``, ``+ - * /``, parentheses and
``abs(…)``. Anything else (attribute access, other calls, strings) is refused before
evaluation; nothing is passed to ``eval``. A comparison with a missing value is false.
"""

from __future__ import annotations

import ast
import operator
from typing import Any

import numpy as np
import pandas as pd

_COMPARE = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}
_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


class RuleError(ValueError):
    """A rule that cannot be parsed or uses something not allowed."""


def parse(rule: str) -> ast.Expression:
    try:
        tree = ast.parse(rule.strip(), mode="eval")
    except SyntaxError as exc:
        raise RuleError(f"Cannot read the rule {rule!r}: {exc.msg}") from None
    for node in ast.walk(tree):
        allowed = (
            ast.Expression,
            ast.BoolOp,
            ast.And,
            ast.Or,
            ast.UnaryOp,
            ast.Not,
            ast.USub,
            ast.UAdd,
            ast.Compare,
            ast.BinOp,
            ast.Name,
            ast.Load,
            ast.Constant,
            ast.Call,
            *_COMPARE,
            *_BINARY,
        )
        if not isinstance(node, allowed):
            raise RuleError(f"{type(node).__name__} is not allowed in a rule ({rule!r})")
        if isinstance(node, ast.Call) and not (
            isinstance(node.func, ast.Name) and node.func.id == "abs" and len(node.args) == 1
        ):
            raise RuleError(f"Only abs(…) may be called in a rule ({rule!r})")
        if isinstance(node, ast.Constant) and not isinstance(node.value, int | float | bool):
            raise RuleError(f"Only numbers are allowed as constants in a rule ({rule!r})")
    return tree


def names(rule: str) -> set[str]:
    """Columns a rule uses."""
    return {n.id for n in ast.walk(parse(rule)) if isinstance(n, ast.Name) and n.id != "abs"}


def evaluate(rule: str, table: pd.DataFrame) -> pd.Series:
    """Boolean Series (False where a value is missing)."""
    tree = parse(rule)
    missing = sorted(names(rule) - set(table.columns))
    if missing:
        raise RuleError(f"The rule {rule!r} uses unknown column(s): {', '.join(missing)}")

    def value(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return value(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return pd.to_numeric(table[node.id], errors="coerce")
        if isinstance(node, ast.Call):
            return np.abs(value(node.args[0]))
        if isinstance(node, ast.UnaryOp):
            inner = value(node.operand)
            if isinstance(node.op, ast.Not):
                return ~_boolean(inner, table.index)
            return -inner if isinstance(node.op, ast.USub) else inner
        if isinstance(node, ast.BinOp):
            return _BINARY[type(node.op)](value(node.left), value(node.right))
        if isinstance(node, ast.BoolOp):
            parts = [_boolean(value(v), table.index) for v in node.values]
            result = parts[0]
            for part in parts[1:]:
                result = (result & part) if isinstance(node.op, ast.And) else (result | part)
            return result
        if isinstance(node, ast.Compare):
            left = value(node.left)
            result = pd.Series(True, index=table.index)
            for op, comparator in zip(node.ops, node.comparators, strict=True):
                right = value(comparator)
                with np.errstate(invalid="ignore"):
                    result &= _boolean(_COMPARE[type(op)](left, right), table.index)
                left = right
            return result
        raise RuleError(f"Unsupported expression {ast.dump(node)}")

    return _boolean(value(tree), table.index)


def _boolean(value: Any, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.fillna(False).astype(bool)
    return pd.Series(bool(value), index=index)
