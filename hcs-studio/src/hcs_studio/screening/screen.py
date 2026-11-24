"""A primary screen: plates → scores → hits (Phase 23a).

``ScreenConfig`` (YAML/JSON) says which readouts to score, which score the thresholds use, the
hit rule, the replicate rule and the artifact filters. ``score_screen(folders, config)`` reads
finished runs (``per_well_summary.csv`` with image-QC columns, ``plate_layout.json``,
``run_manifest.json``) and returns a ``ScreenResult``:

- **wells** — every well with, per readout alias, the raw value and ``<alias>_pct``,
  ``<alias>_z``, ``<alias>_b`` (``scores.plate_scores``) and ``<alias>_score``: the B-score when
  ``score: b`` and the plate allows it (randomised library), else the robust Z (a note says so);
  ``artifact`` (the filters that fired), ``rule_hit`` and ``well_hit`` (sample well, rule true,
  no artifact);
- **compounds** — one row per compound (name and concentration, cell line) over its replicate
  wells: n, n_hit, mean of every score, SSMD of the first readout (``scores.ssmd``), ``hit``
  from the replicate rule, then manual overrides (``overrides.json``, latest per compound wins,
  each with a reason) → ``final_hit``;
- **hits** — the final hits ranked by ``rank_by``;
- **plates** — per plate: Z′, robust Z′, % activity mode, whether B-score was allowed, hit rate,
  sample median, acquisition time (plate order, for drift).

Artifact filters (``artifact_filters``): ``qc_failed`` (image QC failed the well), ``low_count``
(per-well summary), and any image-QC flag name found in ``qc_flags`` — e.g.
``autofluorescence``, ``precipitation``, ``out_of_focus``. An artefact well is never a hit.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from hcs_studio.screening import rules
from hcs_studio.screening.scores import plate_scores, ssmd


class ReadoutConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str = Field(description="Column of per_well_summary.csv.")
    alias: str = Field(description="Short name used in rules (letters, digits, _).")
    activity: Literal["auto", "percent_inhibition", "percent_of_control"] = "auto"


class ScreenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "screen"
    readouts: list[ReadoutConfig]
    score: Literal["b", "z", "pct"] = Field(
        default="b",
        description="Score behind <alias>_score: B-score (randomised libraries), "
        "robust Z or % activity.",
    )
    hit_rule: str = Field(description='e.g. "count_score <= -3 and not viability_pct < 50"')
    replicate_rule: Literal["all", "any", "majority"] | int = Field(
        default="all", description="all / any / majority of the replicates, or at least n."
    )
    artifact_filters: list[str] = Field(
        default_factory=lambda: [
            "qc_failed",
            "low_count",
            "autofluorescence",
            "precipitation",
            "out_of_focus",
        ]
    )
    rank_by: str | None = Field(
        default=None, description="Score column; default the first readout's score"
    )
    rank_descending: bool = Field(default=False, description="False: most negative first.")

    @classmethod
    def load(cls, path: Path) -> ScreenConfig:
        text = Path(path).read_text(encoding="utf-8")
        if Path(path).suffix.lower() in (".yaml", ".yml"):
            import yaml

            return cls.model_validate(yaml.safe_load(text))
        return cls.model_validate(json.loads(text))

    def check(self) -> None:
        aliases = [r.alias for r in self.readouts]
        if len(set(aliases)) != len(aliases):
            raise ValueError("Readout aliases must be unique")
        columns = {f"{a}_{s}" for a in aliases for s in ("pct", "z", "b", "score", "ssmd_single")}
        columns |= set(aliases)
        unknown = rules.names(self.hit_rule) - columns
        if unknown:
            raise ValueError(
                f"The hit rule uses unknown name(s): {', '.join(sorted(unknown))} "
                f"(available: {', '.join(sorted(columns))})"
            )

    @property
    def primary(self) -> str:
        return self.readouts[0].alias

    @property
    def rank_column(self) -> str:
        return self.rank_by or f"{self.primary}_score"


class Override(BaseModel):
    """A manual decision on a compound (key = ``compound|concentration|cell line``)."""

    compound: str
    action: Literal["include", "exclude"]
    reason: str
    actor: str = ""
    when: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class ScreenResult:
    config: ScreenConfig
    wells: pd.DataFrame
    compounds: pd.DataFrame
    hits: pd.DataFrame
    plates: pd.DataFrame
    notes: list[str] = field(default_factory=list)
    run_folders: list[Path] = field(default_factory=list)


# ---------------------------------------------------------------------- reading


def _plate_inputs(folder: Path) -> tuple[dict[str, Any], pd.DataFrame, Any]:
    from hcs_studio.model.layout import load_plate_layout, run_layout_file

    folder = Path(folder)
    manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(folder / "tables" / "per_well_summary.csv")
    path = run_layout_file(folder)
    if path is None:
        raise ValueError(f"{folder.name}: no plate layout (screens need roles and compounds)")
    return manifest, summary, load_plate_layout(path)


def _key(compound: str, concentration: float, cell_line: str) -> str:
    return f"{compound}|{concentration:.6g}|{cell_line}"


def _artifacts(summary: pd.DataFrame, filters: list[str]) -> pd.Series:
    out = pd.Series("", index=summary.index, dtype=object)
    flags = summary["qc_flags"].fillna("").astype(str) if "qc_flags" in summary else None
    for name in filters:
        if name in ("qc_failed", "low_count"):
            if name not in summary:
                continue
            hit = summary[name].fillna(False).astype(bool)
        elif flags is not None:
            hit = flags.str.contains(rf"(?:^|[;, ]){name}(?:$|[;, ]|:)", regex=True)
        else:
            continue
        out[hit] = out[hit].map(lambda s, n=name: f"{s},{n}" if s else n)
    return out


def score_screen(folders: list[Path], config: ScreenConfig) -> ScreenResult:
    from hcs_studio.model.plate import parse_well_name
    from hcs_studio.normalization.plate import wells_from_layout
    from hcs_studio.pharmacology.normalize import drug_treatments

    config.check()
    notes: list[str] = []
    wells_parts, plate_rows = [], []
    for order, folder in enumerate(sorted(folders, key=lambda f: _started(Path(f)))):
        manifest, summary, layout = _plate_inputs(Path(folder))
        plate = manifest.get("plate") or {}
        name = str(plate.get("barcode") or plate.get("name") or Path(folder).name)
        rows, cols = int(plate.get("rows") or layout.rows), int(plate.get("cols") or layout.cols)
        info = wells_from_layout(layout)
        summary = summary.set_index("well", drop=False)
        frame = pd.DataFrame(
            {
                "plate": name,
                "run": Path(folder).name,
                "plate_order": order,
                "well": summary["well"].astype(str),
            }
        )
        frame["row"] = [parse_well_name(w)[0] for w in frame["well"]]
        frame["col"] = [parse_well_name(w)[1] for w in frame["well"]]
        annotations = {w.well: w for w in info}
        frame["role"] = [
            annotations[w].role if w in annotations else "empty" for w in frame["well"]
        ]
        names: list[str] = []
        concentrations: list[float] = []
        lines: list[str] = []
        for w in frame["well"]:
            a = layout.wells.get(parse_well_name(w))
            drugs = drug_treatments(a) if a is not None else ()
            names.append(" + ".join(d[0] for d in drugs))
            concentrations.append(drugs[0][1] if len(drugs) == 1 else math.nan)
            lines.append(a.cell_line if a is not None else "")
        frame["compound"], frame["concentration"], frame["cell_line"] = (
            names,
            concentrations,
            lines,
        )
        plate_info: dict[str, Any] = {
            "plate": name,
            "run": Path(folder).name,
            "plate_order": order,
            "started": manifest.get("started"),
        }
        for readout in config.readouts:
            if readout.column not in summary:
                raise ValueError(f"{name}: no column {readout.column!r} in per_well_summary.csv")
            values = {
                str(w): float(v) for w, v in summary[readout.column].items() if np.isfinite(v)
            }
            scores, facts = plate_scores(
                values, info, rows, cols, readout.activity, positional=config.score == "b"
            )
            scores = scores.set_index("well")
            a = readout.alias
            frame[a] = frame["well"].map(scores["raw"])
            for s in ("pct", "z", "b", "ssmd_single"):
                frame[f"{a}_{s}"] = frame["well"].map(scores[s])
            if config.score == "b" and not facts["b_score"]:
                frame[f"{a}_score"] = frame[f"{a}_z"]
                notes.append(
                    f"{name}, {a}: B-score not used (robust Z instead): "
                    + "; ".join(facts["notes"])
                )
            else:
                frame[f"{a}_score"] = frame[f"{a}_{config.score}"]
            plate_info.update(
                {
                    f"{a}_z_prime": facts["z_prime"],
                    f"{a}_z_prime_robust": facts["z_prime_robust"],
                    f"{a}_activity": facts["activity"],
                    f"{a}_b_score": facts["b_score"],
                    f"{a}_sample_median": facts["sample_median"],
                }
            )
        frame["artifact"] = frame["well"].map(_artifacts(summary, config.artifact_filters))
        wells_parts.append(frame)
        plate_rows.append(plate_info)
    wells = pd.concat(wells_parts, ignore_index=True)
    sample = wells["role"] == "sample"
    wells["rule_hit"] = rules.evaluate(config.hit_rule, wells) & sample
    wells["well_hit"] = wells["rule_hit"] & (wells["artifact"] == "")
    plates = pd.DataFrame(plate_rows)
    plates["samples"] = plates["plate"].map(wells[sample].groupby("plate").size())
    plates["well_hits"] = (
        plates["plate"].map(wells[wells["well_hit"]].groupby("plate").size()).fillna(0)
    )
    plates["hit_rate_pct"] = 100 * plates["well_hits"] / plates["samples"]
    compounds = _compounds(wells[sample].copy(), config)
    result = ScreenResult(
        config, wells, compounds, pd.DataFrame(), plates, notes, [Path(f) for f in folders]
    )
    return result


def _started(folder: Path) -> str:
    try:
        return str(
            json.loads((folder / "run_manifest.json").read_text("utf-8")).get("started") or ""
        )
    except (OSError, ValueError):
        return ""


def _replicate_hit(n: int, n_hit: int, rule: str | int) -> bool:
    if rule == "all":
        return n > 0 and n_hit == n
    if rule == "any":
        return n_hit > 0
    if rule == "majority":
        return n_hit > n / 2
    return n_hit >= int(rule)


def _compounds(samples: pd.DataFrame, config: ScreenConfig) -> pd.DataFrame:
    samples["key"] = [
        _key(c, conc, line)
        for c, conc, line in zip(
            samples["compound"],
            samples["concentration"].fillna(0),
            samples["cell_line"],
            strict=True,
        )
    ]
    primary = config.primary
    reference = samples.groupby("plate")[primary].transform("median")
    samples["_d"] = samples[primary] - reference
    score_columns = [
        c for c in samples.columns if any(c.startswith(f"{r.alias}_") for r in config.readouts)
    ]
    rows = []
    for key, group in samples.groupby("key", sort=False):
        usable = group[group["artifact"] == ""]
        row: dict[str, Any] = {
            "key": key,
            "compound": group["compound"].iloc[0],
            "concentration": group["concentration"].iloc[0],
            "cell_line": group["cell_line"].iloc[0],
            "n": len(group),
            "n_usable": len(usable),
            "n_hit": int(group["well_hit"].sum()),
            "plates": ",".join(sorted(set(group["plate"]))),
            "wells": ",".join(
                f"{p}:{w}" for p, w in zip(group["plate"], group["well"], strict=True)
            ),
            "artifacts": ",".join(
                sorted({a for s in group["artifact"] for a in s.split(",") if a})
            ),
        }
        for column in score_columns:
            row[column] = float(usable[column].mean()) if not usable.empty else math.nan
        row.update({f"{primary}_{k}": v for k, v in ssmd(usable["_d"]).items() if k != "n"})
        row["hit"] = _replicate_hit(
            len(usable), int(usable["well_hit"].sum()), config.replicate_rule
        )
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- overrides


def load_overrides(folder: Path) -> list[Override]:
    path = Path(folder) / "overrides.json"
    if not path.exists():
        return []
    return [Override.model_validate(o) for o in json.loads(path.read_text("utf-8"))]


def add_override(folder: Path, override: Override) -> None:
    """Append a manual decision (never rewritten: the history stays)."""
    if not override.reason.strip():
        raise ValueError("A manual hit decision needs a reason")
    existing = [o.model_dump() for o in load_overrides(folder)]
    existing.append(override.model_dump())
    Path(folder).mkdir(parents=True, exist_ok=True)
    (Path(folder) / "overrides.json").write_text(json.dumps(existing, indent=1), "utf-8")


def finalize(result: ScreenResult, overrides: list[Override]) -> ScreenResult:
    """Apply overrides (latest per compound) and rank the hits."""
    compounds = result.compounds.copy()
    latest = {o.compound: o for o in overrides}
    unknown = sorted(set(latest) - set(compounds["key"]) - set(compounds["compound"]))
    if unknown:
        result.notes.append(f"overrides for compounds not in this screen: {', '.join(unknown)}")
    decision, reason = [], []
    for key, name in zip(compounds["key"], compounds["compound"], strict=True):
        override = latest.get(key) or latest.get(name)
        decision.append(override.action if override else "")
        reason.append(override.reason if override else "")
    compounds["override"], compounds["override_reason"] = decision, reason
    compounds["final_hit"] = np.where(
        compounds["override"] == "include",
        True,
        np.where(compounds["override"] == "exclude", False, compounds["hit"]),
    )
    rank = result.config.rank_column
    hits = compounds[compounds["final_hit"].astype(bool)].sort_values(
        rank, ascending=not result.config.rank_descending, na_position="last"
    )
    hits.insert(0, "rank", range(1, len(hits) + 1))
    compounds["rank"] = compounds["key"].map(dict(zip(hits["key"], hits["rank"], strict=True)))
    result.compounds, result.hits = compounds, hits.reset_index(drop=True)
    return result


def write_screen(result: ScreenResult, out: Path, folders: list[Path]) -> list[str]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "wells.csv": result.wells,
        "compounds.csv": result.compounds,
        "hits.csv": result.hits,
        "plates.csv": result.plates,
    }
    for name, table in files.items():
        table.to_csv(out / name, index=False)
    sample = result.wells["role"] == "sample"
    summary = {
        "name": result.config.name,
        "config": result.config.model_dump(mode="json"),
        "runs": [str(f) for f in folders],
        "plates": len(result.plates),
        "sample_wells": int(sample.sum()),
        "compounds": len(result.compounds),
        "hits": len(result.hits),
        "hit_rate_pct": 100 * len(result.hits) / max(len(result.compounds), 1),
        "artifact_wells": int((result.wells.loc[sample, "artifact"] != "").sum()),
        "notes": result.notes,
        "written": datetime.now(UTC).isoformat(),
    }
    (out / "screen.json").write_text(json.dumps(summary, indent=1, default=str), "utf-8")
    from hcs_studio.screening.report import write_report

    write_report(result, out)
    return [*files, "screen.json", "report.html"]


def run_screen(
    folders: list[Path], config: ScreenConfig, out: Path, gallery: int = 0
) -> ScreenResult:
    """Score, apply the overrides already in ``out`` and write the screen files (and the hit
    gallery of the first ``gallery`` hits)."""
    result = finalize(score_screen(folders, config), load_overrides(out))
    if gallery:
        from hcs_studio.screening.report import write_gallery

        Path(out).mkdir(parents=True, exist_ok=True)
        write_gallery(result, Path(out), max_hits=gallery)
    write_screen(result, out, folders)
    return result
