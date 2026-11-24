"""Routing rules: which pipeline (and follow-up analyses) an incoming plate gets (Phase 19.2).

A YAML file (``AutomationSettings.routing_file``)::

    rules:
      - name: viability screen
        barcode_prefix: VIA          # or barcode_regex: "^VIA-\\d{4}$"
        instrument: ImageXpress      # optional; case-insensitive
        protocol: "20x*"             # optional; shell-style pattern
        assay: viability             # optional; the registry's assay of the barcode
        pipeline: pipelines/viability.yaml   # relative to this file
        qc_rules: qc/rules.yaml      # optional: image QC before the analysis
        pharmacology: {readouts: [nuclei_count], normalize: percent_inhibition}
        screen: screens/primary.yaml # optional: hit calling (ScreenConfig)
        kinetics: false              # analyse the plate's time points together
        notify: [owner@lab.org]      # who owns plates of this rule
      - name: everything else
        default: true
        pipeline: pipelines/basic.yaml

Rules are tried in order; a rule matches when every criterion it gives matches. ``default``
rules match everything (put them last). No match → the acquisition is ``unrouted`` and people
are told.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Rule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    default: bool = False
    barcode_prefix: str = ""
    barcode_regex: str = ""
    instrument: str = ""
    protocol: str = ""
    assay: str = ""
    pipeline: str
    qc_rules: str = ""
    pharmacology: dict[str, Any] | None = None
    screen: str = ""
    kinetics: bool = False
    workers: int | None = None
    notify: list[str] = Field(default_factory=list)

    @field_validator("barcode_regex")
    @classmethod
    def _regex(cls, value: str) -> str:
        if value:
            try:
                re.compile(value)
            except re.error as exc:
                raise ValueError(f"barcode_regex is not a valid regular expression: {exc}") from exc
        return value

    def matches(self, plate: dict[str, Any]) -> bool:
        if self.default:
            return True
        criteria = [
            self.barcode_prefix,
            self.barcode_regex,
            self.instrument,
            self.protocol,
            self.assay,
        ]
        if not any(criteria):
            return False  # a rule without criteria must say ``default: true``
        barcode = str(plate.get("barcode", ""))
        if self.barcode_prefix and not barcode.upper().startswith(self.barcode_prefix.upper()):
            return False
        if self.barcode_regex and not re.search(self.barcode_regex, barcode):
            return False
        if self.instrument and self.instrument.lower() != str(plate.get("instrument", "")).lower():
            return False
        if self.protocol and not fnmatch.fnmatch(
            str(plate.get("protocol", "")).lower(), self.protocol.lower()
        ):
            return False
        return not self.assay or self.assay.lower() == str(plate.get("assay", "")).lower()


class Routing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rules: list[Rule] = Field(default_factory=list)
    #: folder the relative paths of the rules are relative to (the YAML file's folder)
    base: str = ""

    def route(self, plate: dict[str, Any]) -> Rule | None:
        return next((r for r in self.rules if r.matches(plate)), None)

    def path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() or not self.base else Path(self.base) / path


def load_routing(path: str | Path) -> Routing:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    routing = Routing.model_validate({**data, "base": str(path.parent)})
    return routing


def check_routing(routing: Routing) -> list[str]:
    """Problems of a routing file: missing pipelines and QC rules, unreadable files, rules after
    a default that can never match."""
    from hcs_studio.pipeline.schema import Pipeline

    problems = []
    seen_default = False
    for rule in routing.rules:
        if seen_default:
            problems.append(f"{rule.name}: never used (a default rule comes before it)")
        seen_default = seen_default or rule.default
        for label, value in (
            ("pipeline", rule.pipeline),
            ("qc_rules", rule.qc_rules),
            ("screen", rule.screen),
        ):
            if not value:
                continue
            target = routing.path(value)
            if not target.exists():
                problems.append(f"{rule.name}: {label} {target} does not exist")
            elif label == "pipeline":
                try:
                    Pipeline.model_validate(yaml.safe_load(target.read_text(encoding="utf-8")))
                except Exception as exc:
                    problems.append(f"{rule.name}: pipeline {target} is not valid: {exc}")
        if not rule.default and not any(
            (rule.barcode_prefix, rule.barcode_regex, rule.instrument, rule.protocol, rule.assay)
        ):
            problems.append(f"{rule.name}: no criteria (add some, or default: true)")
    return problems


def registry_assay(barcode: str) -> str:
    """The assay the registry links to a barcode (plate assay, else its experiment's)."""
    from sqlalchemy import select

    from hcs_studio.db import session_scope
    from hcs_studio.db.catalog import Plate

    with session_scope() as session:
        plate = session.scalars(select(Plate).where(Plate.barcode == barcode)).first()
        if plate is None:
            return ""
        if plate.assay:
            return plate.assay
        return plate.experiment.assay if plate.experiment is not None else ""
