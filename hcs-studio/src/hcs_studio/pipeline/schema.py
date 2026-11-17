"""Pipelines are data: an ordered list of module steps, saved as YAML or JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from hcs_studio.processing.base import MODULES, Module, load_builtin_modules

FORMAT_VERSION = 1


class PipelineStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    module: str
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> PipelineStep:
        load_builtin_modules()
        if self.module not in MODULES:
            known = ", ".join(MODULES.names())
            raise ValueError(f"Unknown module '{self.module}'. Available: {known}")
        validated = MODULES.get(self.module).Params.model_validate(self.params)
        # Store every parameter explicitly so a saved pipeline is fully self-describing.
        object.__setattr__(self, "params", validated.model_dump(mode="json"))
        return self

    def is_table_step(self) -> bool:
        from hcs_studio.analysis.table_steps import is_table_module

        return is_table_module(self.module)

    def create(self) -> Module:
        cls = MODULES.get(self.module)
        return cls(cls.Params.model_validate(self.params))


class Pipeline(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="pipeline", pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$", max_length=60)
    format_version: int = FORMAT_VERSION
    steps: list[PipelineStep] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> Pipeline:
        if self.format_version != FORMAT_VERSION:
            raise ValueError(
                f"Pipeline format version {self.format_version} is not supported "
                f"(this version reads {FORMAT_VERSION})"
            )
        return self

    def enabled_steps(self) -> list[PipelineStep]:
        return [s for s in self.steps if s.enabled]

    def image_steps(self) -> list[PipelineStep]:
        """Enabled steps that run on each field."""
        return [s for s in self.enabled_steps() if not s.is_table_step()]

    def table_steps(self) -> list[PipelineStep]:
        """Enabled steps that run on the object tables after all wells, in order."""
        return [s for s in self.enabled_steps() if s.is_table_step()]

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    @classmethod
    def from_text(cls, text: str) -> Pipeline:
        """Parse YAML (of which JSON is a subset)."""
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(f"Not valid YAML: {exc}") from None
        if not isinstance(data, dict):
            raise ValueError("A pipeline file must contain a mapping with 'name' and 'steps'")
        return cls.model_validate(data)

    @classmethod
    def load(cls, path: Path) -> Pipeline:
        return cls.from_text(Path(path).read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        path = Path(path)
        if path.suffix.lower() == ".json":
            text = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        else:
            text = self.to_yaml()
        path.write_text(text, encoding="utf-8")
