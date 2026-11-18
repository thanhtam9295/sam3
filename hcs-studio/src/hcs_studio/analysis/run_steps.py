"""Run steps: pipeline steps that run once per run, after all wells, the tables and the
report — pharmacology, a figure per compound, the experiment report.

They pass images through unchanged (so previews and resumes ignore them), are left out of
the pipeline hash, and can be applied to a finished run with ``hcs-studio analyze``.
Their outcome (outputs, time, errors) is recorded in ``run_manifest.json`` → ``run_steps``;
a failing run step never fails the run.
"""

from __future__ import annotations

import logging
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from hcs_studio.processing.base import MODULES, ImageData, Module, ModuleParams


class RunStep(Module):
    """A step that works on the finished run folder."""

    def apply(self, image: ImageData, state: Any) -> ImageData:
        return image

    @abstractmethod
    def run(self, folder: Path, log: logging.Logger) -> dict[str, Any]:
        """Do the work on ``folder``; return what to record (outputs, summary)."""


def is_run_module(name: str) -> bool:
    return name in MODULES and issubclass(MODULES.get(name), RunStep)


def run_run_steps(
    folder: Path, steps: list[Any], log: logging.Logger | None = None
) -> list[dict[str, Any]]:
    """Run the run steps (``PipelineStep`` or ``RunStep``) in order on a run folder."""
    log = log or logging.getLogger("hcs_studio.run_steps")
    records = []
    for number, step in enumerate(steps, start=1):
        module = step if isinstance(step, RunStep) else step.create()
        started = time.monotonic()
        record: dict[str, Any] = {
            "step": number,
            "module": module.name,
            "title": module.title,
            "params": module.params.model_dump(mode="json"),
        }
        try:
            record.update(module.run(Path(folder), log))
            record["status"] = "completed"
            log.info("Run step %s completed", module.name)
        except Exception as exc:  # recorded, never fails the run
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            log.exception("Run step %s failed", module.name)
        record["seconds"] = round(time.monotonic() - started, 2)
        records.append(record)
    return records


def _relative(folder: Path, paths: list[Path]) -> list[str]:
    out = []
    for p in paths:
        try:
            out.append(str(Path(p).relative_to(folder)))
        except ValueError:
            out.append(str(p))
    return out


# ---------------------------------------------------------------- pharmacology


SynergyModel = Literal["bliss", "hsa", "loewe", "zip"]


def _all_models() -> list[SynergyModel]:
    return ["bliss", "hsa", "loewe", "zip"]


class PharmacologyParams(ModuleParams):
    readouts: list[str] = Field(
        default_factory=list,
        description="Per-well readouts (columns of per_well_summary); empty = the first "
        "*_count readout.",
    )
    normalization: Literal["percent_of_control", "percent_inhibition", "gr"] = Field(
        default="percent_inhibition",
        description="% of control, % inhibition (needs positive controls) or GR.",
    )
    models: list[SynergyModel] = Field(
        default_factory=_all_models,
        description="Synergy models for combination matrices.",
    )
    curve_model: Literal["4pl", "5pl", "3pl_bottom", "3pl_top"] = Field(
        default="4pl", description="Dose–response model."
    )
    robust: bool = Field(default=False, description="Robust (soft L1) curve fitting.")
    outlier_mad: float | None = Field(
        default=None, gt=0, description="Exclude wells beyond median ± k·MAD; empty = off."
    )
    z_threshold: float = Field(default=0.5, description="Flag plates with Z′ below this.")
    doubling_time_h: float | None = Field(default=None, gt=0, description="GR: doubling time.")
    assay_duration_h: float | None = Field(default=None, gt=0, description="GR: assay duration.")
    pdf: bool = Field(default=True, description="Also write the PDF report.")


def default_readouts(folder: Path) -> list[str]:
    import pandas as pd

    table = Path(folder) / "tables" / "per_well_summary.csv"
    if not table.exists():
        raise ValueError("the run has no per-well summary")
    columns = list(pd.read_csv(table, nrows=1).columns)
    counts = [
        c
        for c in columns
        if c.endswith("_count") and not c.startswith(("gate_", "class_", "cluster_"))
    ]
    if not counts:
        raise ValueError("no *_count readout in the per-well summary; give 'readouts'")
    return counts[:1]


@MODULES.register("pharmacology")
class PharmacologyStep(RunStep):
    name = "pharmacology"
    title = "Pharmacology: dose–response and synergy"
    Params = PharmacologyParams

    def options(self, folder: Path) -> Any:
        from hcs_studio.pharmacology.analysis import PharmOptions

        p: PharmacologyParams = self.params
        return PharmOptions(
            readouts=p.readouts or default_readouts(folder),
            normalization=p.normalization,
            models=tuple(p.models),
            curve_model=p.curve_model,
            robust=p.robust,
            outlier_mad=p.outlier_mad,
            z_threshold=p.z_threshold,
            doubling_time_h=p.doubling_time_h,
            assay_duration_h=p.assay_duration_h,
            pdf=p.pdf,
        )

    def run(self, folder: Path, log: logging.Logger) -> dict[str, Any]:
        from hcs_studio.pharmacology.analysis import run_pharmacology

        result = run_pharmacology([folder], self.options(folder))
        summary = {
            readout: {
                "curves": len(rr.curves),
                "combinations": len(rr.combinations),
                "z_prime": [round(float(z), 3) for z in rr.qc["z_prime"]],
            }
            for readout, rr in result.readouts.items()
        }
        return {"outputs": ["pharmacology/"], "summary": summary}


# ----------------------------------------------------------- figure per compound


class CompoundFiguresParams(ModuleParams):
    source: str = Field(default="corrected", description="corrected, enhanced, raw or step:<n>.")
    representative_readout: str = Field(
        default="",
        description="Field closest to the well median of this readout (a fields.csv column "
        "or object feature); empty = by object count.",
    )
    format: Literal["png", "tiff", "svg", "pdf"] = Field(default="png", description="Format.")
    dpi: int = Field(default=300, ge=72, le=1200, description="PNG/TIFF resolution.")
    width_mm: float = Field(default=180.0, gt=20, description="Figure width (85 / 180 mm).")
    split_channels: bool = Field(default=False, description="Each channel in grayscale + merge.")
    compounds: list[str] = Field(default_factory=list, description="Only these; empty = all.")


@MODULES.register("compound_figures")
class CompoundFigures(RunStep):
    name = "compound_figures"
    title = "Figures: a montage per compound"
    Params = CompoundFiguresParams

    def run(self, folder: Path, log: logging.Logger) -> dict[str, Any]:
        from hcs_studio.compare.export import batch_compound_figures
        from hcs_studio.compare.spec import Comparison, ExportSettings, PanelSpec

        p: CompoundFiguresParams = self.params
        base = Comparison(
            panels=[PanelSpec(run=str(folder), well="A01", source=p.source)],
            representative_readout=p.representative_readout,
            settings=ExportSettings(
                format=p.format,
                dpi=p.dpi,
                width_mm=p.width_mm,
                split_channels=p.split_channels,
                label_template="{well}",
            ),
        )
        files = batch_compound_figures(folder, folder / "figures", base, p.compounds or None)
        return {"outputs": _relative(folder, files)}


# ------------------------------------------------------------- experiment report


class ExperimentReportParams(ModuleParams):
    readouts: list[str] = Field(
        default_factory=list,
        description="Readouts for pharmacology when it has not run yet; empty = the first "
        "*_count readout.",
    )
    montages: bool = Field(default=True, description="Representative image montages.")
    montage_source: str = Field(default="corrected", description="Image of the montages.")
    representative_readout: str = Field(default="", description="Field choice for montages.")
    pdf: bool = Field(default=True, description="Also write the PDF.")


@MODULES.register("experiment_report")
class ExperimentReportStep(RunStep):
    name = "experiment_report"
    title = "Experiment report (HTML + PDF)"
    Params = ExperimentReportParams

    def run(self, folder: Path, log: logging.Logger) -> dict[str, Any]:
        from hcs_studio.output.experiment import write_experiment_report

        p: ExperimentReportParams = self.params
        files = write_experiment_report(
            folder,
            readouts=p.readouts or None,
            montages=p.montages,
            montage_source=p.montage_source,
            representative_readout=p.representative_readout,
            pdf=p.pdf,
        )
        return {"outputs": _relative(folder, files)}
