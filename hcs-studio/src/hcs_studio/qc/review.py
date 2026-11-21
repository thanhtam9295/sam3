"""Reviewing QC decisions (Phase 18): accept or override a flag, with a reason.

Reviews are kept in ``qc/overrides.json`` (append-only; the latest review of a field or well
counts). Actions:

- field ``confirm``: the flag is right (the rules' decision stands; a "bad" example for
  learned QC); field ``include``: keep a field the rules excluded (a "good" example);
  field ``exclude``: leave out a field the rules kept (a "bad" example);
- well ``include``: pass a well the rules failed; well ``exclude``: fail a well;
  well ``confirm``: the well's decision stands.

After a review the decisions are recomputed from the field table (``rule_excluded`` keeps
what the rules decided), the ``qc/`` files and ``qc_report.html`` are rewritten, and in a run
folder the tables and ``index.html`` are rebuilt so the change reaches counts, statistics
and plate maps. Pharmacology results change when pharmacology is run again.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from hcs_studio.qc.decide import QcResult, add_compound_flags, decide_plate, decide_wells
from hcs_studio.qc.rules import QcRules, load_rules

OVERRIDES = "overrides.json"
Level = Literal["field", "well"]
Action = Literal["confirm", "include", "exclude"]


class Override(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: Level
    well: str
    field: int | None = None
    action: Action
    reason: str = Field(min_length=1)
    actor: str = ""
    time: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def key(self) -> tuple[str, str, int | None]:
        return (self.level, self.well, self.field if self.level == "field" else None)


def qc_folder(folder: Path) -> Path:
    qc = Path(folder) / "qc"
    if not (qc / "field_qc.parquet").exists():
        raise ValueError(f"{folder} has no image QC (qc/field_qc.parquet)")
    return qc


def load_overrides(folder: Path) -> list[Override]:
    path = Path(folder) / "qc" / OVERRIDES
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Override.model_validate(item) for item in data.get("overrides", [])]


def latest_overrides(folder: Path) -> dict[tuple[str, str, int | None], Override]:
    out: dict[tuple[str, str, int | None], Override] = {}
    for override in load_overrides(folder):
        out[override.key] = override
    return out


def load_fields(folder: Path) -> pd.DataFrame:
    fields = pd.read_parquet(qc_folder(folder) / "field_qc.parquet")
    fields["well"] = fields["well"].astype(str)
    fields["field"] = fields["field"].astype(int)
    if "rule_excluded" not in fields:
        fields["rule_excluded"] = fields["excluded"].astype(bool)
    return fields


def _layout(folder: Path) -> Any:
    from hcs_studio.model.layout import load_plate_layout, run_layout_file

    path = run_layout_file(Path(folder))
    if path is None:
        return None
    try:
        return load_plate_layout(path)
    except (OSError, ValueError):
        return None


def redecide(folder: Path) -> QcResult:
    """Recompute well and plate decisions with the reviews; rewrite the ``qc/`` files."""
    from hcs_studio.model.plate import well_name
    from hcs_studio.qc.run import write_qc

    folder = Path(folder)
    qc = qc_folder(folder)
    rules: QcRules = load_rules(qc / "rules.yaml")
    previous = json.loads((qc / "plate_qc.json").read_text(encoding="utf-8"))
    fields = load_fields(folder)
    reviews = latest_overrides(folder)
    fields["excluded"] = fields["rule_excluded"].astype(bool)
    fields["review"] = ""
    fields["review_reason"] = ""
    for (level, well, field), override in reviews.items():
        if level != "field":
            continue
        hit = (fields["well"] == well) & (fields["field"] == field)
        fields.loc[hit, "review"] = override.action
        fields.loc[hit, "review_reason"] = override.reason
        if override.action in ("include", "exclude"):
            fields.loc[hit, "excluded"] = override.action == "exclude"
    layout = _layout(folder)
    wells = add_compound_flags(decide_wells(fields, rules), fields, rules, layout)
    wells["review"] = ""
    for (level, well, _), override in reviews.items():
        if level != "well" or well not in set(wells["well"]):
            continue
        i = wells.index[wells["well"] == well][0]
        wells.at[i, "review"] = override.action
        if override.action == "include":
            wells.at[i, "failed"] = False
            wells.at[i, "reasons"] = f"passed on review: {override.reason}"
        elif override.action == "exclude":
            wells.at[i, "failed"] = True
            wells.at[i, "reasons"] = "; ".join(
                r
                for r in (str(wells.at[i, "reasons"]), f"failed on review: {override.reason}")
                if r
            )
    roles = {well_name(*k): a.role for k, a in layout.wells.items()} if layout else None
    plate = decide_plate(wells, rules, roles, previous.get("z_prime"))
    plate.update(
        {
            "excluded_fields": int(fields["excluded"].sum()),
            "fields": len(fields),
            "flag_counts": {
                f: int(fields[f"flag_{f}"].sum()) for f in rules.flags if f"flag_{f}" in fields
            },
            "compound_flags": previous.get("compound_flags"),
            "reviews": len(reviews),
        }
    )
    keep = {k: v for k, v in previous.items() if k in ("seconds", "plate")}
    result = QcResult(fields, wells, plate)
    write_qc(result, folder, rules, extra=keep)
    return result


def apply_override(folder: Path, override: Override, *, refresh: bool = True) -> QcResult:
    """Record a review and recompute the decisions (and, in a run folder, the tables)."""
    folder = Path(folder)
    fields = load_fields(folder)
    if override.level == "field":
        if override.field is None:
            raise ValueError("A field review needs the field number")
        if not ((fields["well"] == override.well) & (fields["field"] == override.field)).any():
            raise KeyError(f"No field {override.field} of well {override.well} in this QC")
    elif override.well not in set(fields["well"]):
        raise KeyError(f"No well {override.well} in this QC")
    path = qc_folder(folder) / OVERRIDES
    history = [o.model_dump() for o in load_overrides(folder)] + [override.model_dump()]
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps({"overrides": history}, indent=1), encoding="utf-8")
    part.replace(path)
    result = redecide(folder)
    if refresh and (folder / "run_manifest.json").exists() and (folder / "wells").is_dir():
        from hcs_studio.pipeline.executor import refresh_run_tables

        refresh_run_tables(folder)
    return result
