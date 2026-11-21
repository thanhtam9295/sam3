"""QC rules: a versioned YAML file with field thresholds and well / plate decisions (Phase 18).

A field rule names a metric (see :mod:`hcs_studio.qc.metrics`) and one or more conditions; the
rule fires when any condition holds and adds its ``flag`` to the field. Conditions:

- absolute: ``above`` / ``below`` (value > above, value < below);
- plate-relative: ``z_above`` / ``z_below`` on the robust Z of the value among the plate's
  fields (median, 1.4826 × MAD), and ``rel_above`` / ``rel_below`` on value / plate median.

Per-channel metrics take ``channel``: ``content`` (the content channel), ``any`` (fires if
any channel meets a condition) or a channel number from 1. A field is excluded when a rule
with ``exclude: true`` fires. Wells fail with too few good fields or cells; plates fail with
too many failed wells, failed controls or a low Z′.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA = "hcs-studio-qc-rules/1"

#: Metrics computed for every channel (columns ``c<n>_<metric>``).
CHANNEL_METRICS = (
    "plls",
    "normalized_variance",
    "laplacian_variance",
    "saturated_fraction",
    "background",
    "p50",
    "p99",
    "p999",
    "gradient",
    "best_z",
    "best_z_offset",
)
#: Metrics of the content channel only.
FIELD_METRICS = (
    "object_count",
    "area_fraction",
    "largest_gap_fraction",
    "debris_count",
    "debris_area_fraction",
    "fiber_count",
    "bubble_fraction",
    "out_of_well_fraction",
    "particle_fraction",
)
CONDITIONS = ("above", "below", "z_above", "z_below", "rel_above", "rel_below")


class FieldRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flag: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    metric: str
    channel: Literal["content", "any"] | int = "content"
    above: float | None = None
    below: float | None = None
    z_above: float | None = None
    z_below: float | None = None
    rel_above: float | None = None
    rel_below: float | None = None
    exclude: bool = True
    description: str = ""

    @model_validator(mode="after")
    def _check(self) -> FieldRule:
        if self.metric not in CHANNEL_METRICS + FIELD_METRICS:
            raise ValueError(f"Unknown QC metric {self.metric!r}")
        if all(getattr(self, c) is None for c in CONDITIONS):
            raise ValueError(f"Rule {self.flag!r} has no condition ({', '.join(CONDITIONS)})")
        if isinstance(self.channel, int) and self.channel < 1:
            raise ValueError("Channels are numbered from 1")
        return self

    def columns(self, n_channels: int, content_channel: int) -> list[str]:
        if self.metric in FIELD_METRICS:
            return [self.metric]
        if self.channel == "content":
            return [f"c{content_channel}_{self.metric}"]
        if self.channel == "any":
            return [f"c{c}_{self.metric}" for c in range(1, n_channels + 1)]
        if self.channel > n_channels:
            raise ValueError(f"Rule {self.flag!r}: channel {self.channel} does not exist")
        return [f"c{self.channel}_{self.metric}"]


class WellDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_good_fields: int = Field(default=1, ge=0)
    min_good_field_fraction: float = Field(default=0.5, ge=0, le=1)
    #: Cells (content-channel objects) in the good fields; keep 0 for dose–response plates,
    #: where potent wells legitimately have few cells.
    min_cells: int = Field(default=0, ge=0)


class PlateDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_failed_well_fraction: float = Field(default=0.1, ge=0, le=1)
    #: A control role fails when more than this fraction of its wells failed.
    max_failed_control_fraction: float = Field(default=0.5, ge=0, le=1)
    controls: tuple[str, ...] = ("negative_control", "positive_control")
    min_z_prime: float | None = 0.5
    #: Stop a run when the plate fails QC (unattended runs fail fast instead of analysing).
    stop_run: bool = False


class QcSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content_channel: int = Field(default=1, ge=1)
    nucleus_diameter_um: float = Field(default=10.0, gt=0)
    #: Brightfield / digital phase contrast channel (from 1) for precipitation; none = off.
    brightfield_channel: int | None = Field(default=None, ge=1)


class CompoundChecks(BaseModel):
    """Compound-related artifacts, judged per well against the plate's vehicle wells and along
    each compound's dose series (needs a plate layout)."""

    model_config = ConfigDict(extra="forbid")

    #: Robust Z (vs vehicle wells) above which a well's background / particles are raised.
    z: float = Field(default=5.0, gt=0)
    #: Spearman ρ of the raised metric with concentration along the series (monotonic rise).
    min_rho: float = Field(default=0.8, ge=-1, le=1)
    #: Concentrations a series needs before a trend is judged.
    min_concentrations: int = Field(default=3, ge=2)
    autofluorescence: bool = True
    precipitation: bool = True
    #: Fail flagged wells (left out of fits) instead of only flagging them.
    exclude: bool = False


class LearnedQc(BaseModel):
    """A field classifier trained on reviewed fields (``hcs-studio qc-train``)."""

    model_config = ConfigDict(extra="forbid")

    model: str  # path of the classifier file
    sha256: str = ""
    #: Flag a field when P(bad) reaches this.
    threshold: float = Field(default=0.5, gt=0, lt=1)
    exclude: bool = False


class QcRules(BaseModel):
    model_config = ConfigDict(extra="forbid", serialize_by_alias=True, validate_by_name=True)

    schema_: str = Field(default=SCHEMA, alias="schema")
    name: str = "default"
    version: int = Field(default=1, ge=1)
    description: str = ""
    settings: QcSettings = QcSettings()
    field: list[FieldRule] = Field(default_factory=list)
    well: WellDecision = WellDecision()
    plate: PlateDecision = PlateDecision()
    compound: CompoundChecks = CompoundChecks()
    learned: LearnedQc | None = None

    @model_validator(mode="after")
    def _check(self) -> QcRules:
        if self.schema_ != SCHEMA:
            raise ValueError(f"Not a QC rules file of this version ({self.schema_!r} ≠ {SCHEMA})")
        return self

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    def sha256(self) -> str:
        text = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @property
    def flags(self) -> list[str]:
        names = [r.flag for r in self.field] + (["learned"] if self.learned else [])
        return list(dict.fromkeys(names))

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "sha256": self.sha256()}


def load_rules(path: Path) -> QcRules:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a QC rules file")
    return QcRules.model_validate(data)


def save_rules(rules: QcRules, path: Path) -> Path:
    path = Path(path)
    path.write_text(rules.to_yaml(), encoding="utf-8")
    return path


def default_rules(**settings: Any) -> QcRules:
    """The shipped rules. Thresholds were set on synthetic artifacts (docs/validation.md) and
    must be tuned on real fields of each instrument."""
    rules = [
        FieldRule(
            flag="blur",
            metric="laplacian_variance",
            rel_below=0.5,
            description="Out of focus: fine detail below half the plate's median",
        ),
        FieldRule(
            flag="saturation",
            metric="saturated_fraction",
            channel="any",
            above=0.002,
            description="More than 0.2 % of pixels at the detector maximum",
        ),
        FieldRule(
            flag="autofocus",
            metric="best_z_offset",
            above=1.5,
            description="Best-focus plane more than 1.5 planes from the plate trend",
        ),
        FieldRule(
            flag="bubble",
            metric="bubble_fraction",
            above=0.02,
            description="Round region inside the field (> 2 %) darker or brighter than the rest",
        ),
        FieldRule(
            flag="out_of_well",
            metric="out_of_well_fraction",
            above=0.02,
            description="Dark region at the field border: well edge or meniscus shadow",
        ),
        FieldRule(
            flag="gradient",
            metric="gradient",
            channel="any",
            above=0.3,
            exclude=False,
            description="Illumination varies by more than 30 % across the field",
        ),
        FieldRule(
            flag="debris",
            metric="debris_area_fraction",
            above=0.002,
            exclude=False,
            description="Very bright, large objects (debris, dust)",
        ),
        FieldRule(
            flag="fiber",
            metric="fiber_count",
            above=0,
            exclude=False,
            description="Long thin bright object (fiber, lint)",
        ),
        FieldRule(
            flag="scratch",
            metric="largest_gap_fraction",
            above=0.08,
            exclude=False,
            description="Cell-free band or scratch across a field that has cells",
        ),
        FieldRule(
            flag="empty",
            metric="object_count",
            below=1,
            exclude=False,
            description="No cells (may be biology: potent compounds)",
        ),
        FieldRule(
            flag="background",
            metric="background",
            channel="any",
            z_above=6,
            z_below=-6,
            exclude=False,
            description="Background far from the plate's other fields",
        ),
    ]
    rules.append(
        FieldRule(
            flag="particles",
            metric="particle_fraction",
            z_above=8,
            exclude=False,
            description="Many small refractile particles in brightfield (precipitate, dust)",
        )
    )
    return QcRules(settings=QcSettings(**settings), field=rules)
