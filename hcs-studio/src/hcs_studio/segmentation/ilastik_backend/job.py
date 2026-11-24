"""What every runner takes and returns (PROMPT_ILASTIK.md §3): runners are interchangeable,
steps do not know which one ran."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

#: workflow → the export sources it offers (as ilastik names them; checked on 1.4.2)
EXPORT_SOURCES = {
    "pixel": ("Probabilities", "Simple Segmentation", "Uncertainty", "Features", "Labels"),
    "autocontext": (
        "Probabilities Stage 2",
        "Simple Segmentation Stage 2",
        "Probabilities Stage 1",
        "Simple Segmentation Stage 1",
        "Probabilities All Stages",
    ),
}


class IlastikError(RuntimeError):
    """ilastik failed; the message holds what it printed last."""


class IlastikJob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project: str
    workflow: str
    #: ilastik input specs ``<file>.h5/data``, one per image
    raw_data: list[str]
    #: other input roles by flag name (e.g. ``prediction_mask``), paired with raw_data by position
    roles: dict[str, list[str]] = Field(default_factory=dict)
    export_source: str
    output_format: str = "hdf5"
    #: folder of the exports; files are ``<input name>_<suffix>.h5``
    output_folder: str
    output_suffix: str
    threads: int = -1
    ram_mb: int = 0
    timeout_s: float | None = None

    def expected_outputs(self) -> list[str]:
        from pathlib import Path

        out = []
        for spec in self.raw_data:
            stem = Path(spec.rsplit(".h5", 1)[0] + ".h5").stem
            out.append(str(Path(self.output_folder) / f"{stem}_{self.output_suffix}.h5"))
        return out


class IlastikResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outputs: list[str]
    command: list[str]
    runner: str
    returncode: int
    seconds: float
    log_tail: str = ""
    project_sha256_before: str = ""
    project_sha256_after: str = ""
