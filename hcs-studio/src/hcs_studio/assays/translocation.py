"""Nuclear translocation (Phase 20d): NF-κB p65, YAP/TAZ, TFEB, β-catenin, FOXO and others.

Two steps:

- **``cytoplasm``** (image step, before ``measure_objects``): a cytoplasm compartment for each
  nucleus, with the nucleus's ID. ``ring``: pixels between ``gap_um`` and ``gap_um +
  width_um`` from the nucleus edge that are closer to this nucleus than to any other (so a ring
  never enters a neighbouring nucleus or its territory); with a cell segmentation (``cells``,
  cells grown from the nuclei share their IDs) the ring also stays inside its own cell.
  ``cells``: the whole cell minus its nucleus (and the gap). The gap keeps the blurred nuclear
  edge out of the cytoplasm.
- **``translocation``** (assay table step): per cell, mean reporter intensity in the nucleus and
  in its cytoplasm, the N/C ratio and its log2. A cell *responds* when its N/C ratio is above a
  threshold set from the **unstimulated** control cells (median + ``threshold_sd`` robust SD of
  log N/C by default; a percentile; Otsu over unstimulated and stimulated cells; or a fixed
  ratio). Per well: % responding, median N/C, mean log2 N/C, and the activation between the
  unstimulated (0 %) and stimulated (100 %) controls.

Which wells are unstimulated depends on the ``mode`` — the same convention as % inhibition in
pharmacology (negative control = 0 % effect, positive control = 100 %):

- ``agonist``: negative controls (vehicle) are unstimulated, positive controls are fully
  stimulated (e.g. TNF-α for p65) — the agonist's EC50;
- ``inhibitor``: every well is stimulated; negative controls = stimulus + vehicle, positive
  controls = unstimulated (or a reference inhibitor) — the inhibitor's IC50.

EC50 / IC50 come from Phase 13: ``hcs-studio pharm <run> --readout tl_responding_pct
--normalize percent_inhibition``.

N/C ratios need **linear, background-subtracted** intensities: an offset left in the image pulls
every ratio towards 1; a top-hat (or rolling-ball) background with a radius near the cell
size flattens the cytoplasm; a polynomial or spline surface fitted under dense cells rises into
the cytoplasm and over-subtracts. The preset subtracts a constant: the mean of blank (no-cell)
wells when the layout has them, else a low percentile of each field; the step warns about the
other cases.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import Field

from hcs_studio.analysis.table_steps import TableParams, WellContext
from hcs_studio.assays.base import (
    POSITIVE_ROLES,
    VEHICLE_ROLES,
    AssayModule,
    ChannelRole,
    channel_column,
)
from hcs_studio.assays.cell_cycle import otsu_log, robust_upper_log
from hcs_studio.model.plate import ImageMetadata
from hcs_studio.processing.base import (
    MODULES,
    ImageData,
    LabelSet,
    Module,
    ModuleError,
    ModuleParams,
    um_to_px,
)

KEYS = ["well", "field", "timepoint", "object_id"]


# ------------------------------------------------------------ cytoplasm step


def cytoplasm_labels(
    nuclei: NDArray[Any],
    gap_px: float,
    width_px: float,
    cells: NDArray[Any] | None = None,
    mode: str = "ring",
) -> NDArray[np.int32]:
    """Cytoplasm with the nucleus IDs, plane by plane (see the module docstring)."""
    from skimage.segmentation import expand_labels

    out = np.zeros(nuclei.shape, dtype=np.int32)
    for z in range(nuclei.shape[0]):
        nuc = nuclei[z].astype(np.int32)
        inner = expand_labels(nuc, distance=gap_px) if gap_px > 0 else nuc
        if mode == "cells":
            if cells is None:
                raise ModuleError("mode 'cells' needs a cell segmentation")
            plane = np.where(inner == 0, cells[z], 0).astype(np.int32)
        else:
            outer = expand_labels(nuc, distance=gap_px + width_px)
            plane = np.where(inner == 0, outer, 0).astype(np.int32)
            if cells is not None:
                plane[cells[z] != plane] = 0
        out[z] = plane
    return out


class CytoplasmParams(ModuleParams):
    nuclei: str = Field(default="nuclei", description="Label set of the nuclei.")
    cells: str = Field(
        default="",
        description="Label set of the cells (grown from the nuclei, same IDs); empty = none.",
    )
    mode: Literal["ring", "cells"] = Field(
        default="ring",
        description="ring: a band around each nucleus; cells: the whole cell minus its nucleus.",
    )
    gap_um: float = Field(default=0.5, ge=0, description="Gap between nucleus and cytoplasm.")
    width_um: float = Field(default=2.0, gt=0, description="Ring width.")
    output: str = Field(default="cytoplasm", description="Name of the new label set.")


@MODULES.register("cytoplasm")
class Cytoplasm(Module):
    """A cytoplasm ring (or cell minus nucleus) per nucleus, with the nucleus ID."""

    name = "cytoplasm"
    title = "Cytoplasm ring around nuclei"
    Params = CytoplasmParams

    def validate(self, metadata: ImageMetadata) -> None:
        um_to_px(self.params.width_um, metadata, "The ring width")
        if self.params.mode == "cells" and not self.params.cells:
            raise ModuleError("mode 'cells' needs the cells label set (e.g. from expand_cells)")

    def apply(self, image: ImageData, state: Any) -> ImageData:
        params: CytoplasmParams = self.params
        nuclei = image.label_set(params.nuclei)
        cells = image.label_set(params.cells).data if params.cells else None
        width = um_to_px(params.width_um, image.metadata, "The ring width")
        gap = um_to_px(params.gap_um, image.metadata, "The gap") if params.gap_um > 0 else 0.0
        data = cytoplasm_labels(nuclei.data, gap, width, cells, params.mode)
        ids = np.arange(max(int(np.max(nuclei.data)), int(np.max(data))) + 1, dtype=np.int32)
        result = LabelSet(params.output, data, parent=params.nuclei, parent_ids=ids,
                          source=f"cytoplasm ({params.mode})")  # fmt: skip
        empty = int(nuclei.count - len(np.setdiff1d(np.unique(data), [0])))
        report = {"module": self.name, "output": params.output, "objects": result.count,
                  "nuclei_without_cytoplasm": empty}  # fmt: skip
        return image.with_labels(result, self.name, report)


# -------------------------------------------------------------- assay step


class TranslocationParams(TableParams):
    dna_channel: str = Field(default="DAPI", description="Nuclear stain.")
    reporter_channel: str = Field(
        default="p65", description="Translocating protein (as on the plate)."
    )
    cytoplasm: str = Field(
        default="cytoplasm",
        description="Label set of the cytoplasm (cytoplasm step), or of the cells (cytoplasm = "
        "cell minus nucleus, from the cells' nc_ratio measurement).",
    )
    statistic: Literal["mean", "median"] = Field(
        default="mean", description="Intensity statistic of nucleus and cytoplasm."
    )
    mode: Literal["agonist", "inhibitor"] = Field(
        default="agonist",
        description="agonist: vehicle wells are unstimulated, positive controls stimulated; "
        "inhibitor: all wells stimulated, positive controls unstimulated.",
    )
    unstimulated_wells: list[str] = Field(
        default_factory=list, description="Unstimulated wells (overrides the layout roles)."
    )
    threshold_method: Literal["unstimulated_robust", "unstimulated_percentile", "controls"] = Field(
        default="unstimulated_robust",
        description="unstimulated_robust: median + threshold_sd robust SD of log N/C of the "
        "unstimulated cells; unstimulated_percentile: a percentile of them; controls: Otsu "
        "over unstimulated and stimulated cells.",
    )
    threshold_sd: float = Field(default=3.0, gt=0)
    percentile: float = Field(default=95.0, gt=50, lt=100)
    ratio_threshold: float | None = Field(default=None, gt=0, description="Fixed N/C threshold.")
    min_cytoplasm_um2: float = Field(
        default=2.0, ge=0, description="Cells with less cytoplasm area are left out."
    )


@MODULES.register("translocation")
class Translocation(AssayModule):
    """N/C ratio per cell and % responding cells, calibrated on unstimulated controls."""

    name = "translocation"
    title = "Assay: nuclear translocation"
    Params = TranslocationParams
    prefix: ClassVar[str] = "tl_"
    roles: ClassVar[tuple[ChannelRole, ...]] = (
        ChannelRole("dna", "DNA", "DAPI or Hoechst: nuclei"),
        ChannelRole("reporter", "Reporter", "the translocating protein (p65, YAP, TFEB …)"),
    )

    def __init__(self, params: Any) -> None:
        super().__init__(params)
        self.wells: dict[str, dict[str, Any]] = {}
        self.control_ratios: dict[str, NDArray[Any]] = {}

    @property
    def p(self) -> TranslocationParams:
        params: TranslocationParams = self.params
        return params

    @classmethod
    def preset(cls, mapping: dict[str, str], **options: Any) -> Any:
        return preset(mapping, **options)

    # ----------------------------------------------------------- controls

    def control_roles(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """(unstimulated roles, stimulated roles) for the mode."""
        if self.p.mode == "agonist":
            return VEHICLE_ROLES, POSITIVE_ROLES
        return POSITIVE_ROLES, VEHICLE_ROLES

    def well_roles(self) -> dict[str, str]:
        roles = super().well_roles()
        for well in self.p.unstimulated_wells:
            roles[well.upper()] = "unstimulated"
        return roles

    def _wells_of(self, kind: str) -> list[str]:
        unstimulated, stimulated = self.control_roles()
        roles = self.well_roles()
        if kind == "unstimulated":
            if self.p.unstimulated_wells:
                return [w for w, r in roles.items() if r == "unstimulated"]
            return [w for w, r in roles.items() if r in unstimulated]
        return [w for w, r in roles.items() if r in stimulated]

    def _objects_of(self, wells: list[str]) -> pd.DataFrame:
        if not wells:
            return pd.DataFrame()
        roles = self.well_roles()
        objects = self.all_objects(tuple({roles[w] for w in wells}))
        return objects[objects["well"].isin(wells)] if not objects.empty else objects

    # ------------------------------------------------------------ the ratio

    def columns(self) -> dict[str, str]:
        return {
            "reporter": channel_column(self.p.reporter_channel, self.p.statistic),
            "reporter_mean": channel_column(self.p.reporter_channel, "mean"),
        }

    def ratios(self, objects: pd.DataFrame, nuclei: pd.DataFrame) -> pd.DataFrame:
        """Nuclear and cytoplasmic reporter intensity, N/C ratio and whether it is usable,
        for the rows of ``nuclei``."""
        column = self.columns()["reporter"]
        kind = self.p.cytoplasm
        cyto_rows = objects[objects["object_type"] == kind]
        if cyto_rows.empty:
            raise ModuleError(
                f"no '{kind}' objects in the run: add the cytoplasm step (or a cell "
                "segmentation) before measure_objects"
            )
        cell_minus_nucleus = channel_column(self.p.reporter_channel, "cytoplasm_mean")
        if kind != "cytoplasm" and cell_minus_nucleus in cyto_rows and self.p.statistic != "mean":
            raise ModuleError("with cells as cytoplasm only the mean is measured")
        if column not in nuclei:
            raise ModuleError(f"{column} is not measured on the {self.params.object_type}")
        nuc = pd.to_numeric(nuclei[column], errors="coerce")
        keys = [k for k in KEYS if k in nuclei and k in cyto_rows]
        area: str | None = "area_um2"
        value_column = column
        if kind != "cytoplasm" and cell_minus_nucleus in cyto_rows:
            # cells grown from nuclei: measure_objects measured the cell minus its nucleus
            value_column, area = cell_minus_nucleus, None
        if value_column not in cyto_rows:
            raise ModuleError(f"{value_column} is not measured on the {kind}")
        lookup = cyto_rows.set_index(keys)
        lookup = lookup[~lookup.index.duplicated()]
        index = pd.MultiIndex.from_arrays([nuclei[k] for k in keys])
        cyto = pd.Series(lookup[value_column].reindex(index).to_numpy(float), index=nuclei.index)
        if area is not None and area in lookup:
            cyto_area = lookup[area].reindex(index).to_numpy(float)
        else:
            cyto_area = np.full(len(nuclei), np.inf)
        usable = (cyto > 0) & (nuc > 0) & (cyto_area >= self.p.min_cytoplasm_um2)
        ratio = (nuc / cyto).where(usable)
        return pd.DataFrame({"nuc": nuc, "cyto": cyto, "ratio": ratio, "usable": usable})

    def _control_ratios(self, wells: list[str]) -> NDArray[Any]:
        objects = self._objects_of(wells)
        if objects.empty:
            return np.array([])
        kind = str(self.params.object_type)
        nuclei = objects[objects["object_type"] == kind]
        table = self.ratios(objects, nuclei)
        return np.asarray(table["ratio"].dropna().to_numpy(float))

    # ------------------------------------------------------------ calibration

    def fit_table(self, sample: pd.DataFrame) -> dict[str, Any]:
        unstimulated_wells = self._wells_of("unstimulated")
        stimulated_wells = self._wells_of("stimulated")
        unstimulated = self._control_ratios(unstimulated_wells)
        source = f"unstimulated wells ({self.p.mode} mode)"
        warnings = self.run_warnings()
        if unstimulated.size == 0:
            kind = str(self.params.object_type)
            unstimulated = np.asarray(
                self.ratios(sample, sample[sample["object_type"] == kind])["ratio"].dropna()
            )
            source = "all wells (no unstimulated wells in the plate layout)"
            warnings.append("no unstimulated control wells: threshold set from all wells")
        stimulated = self._control_ratios(stimulated_wells)
        self.control_ratios = {"unstimulated": unstimulated, "stimulated": stimulated}
        p = self.p
        if p.ratio_threshold:
            threshold = float(p.ratio_threshold)
        elif p.threshold_method == "controls" and stimulated.size:
            threshold = otsu_log(np.concatenate([unstimulated, stimulated]))
        elif p.threshold_method == "unstimulated_percentile":
            if unstimulated.size < 20:
                raise ModuleError("too few unstimulated cells to set a threshold")
            threshold = float(np.percentile(unstimulated, p.percentile))
        else:
            if p.threshold_method == "controls":
                warnings.append("no stimulated control wells: threshold from unstimulated cells")
            threshold = robust_upper_log(unstimulated, p.threshold_sd)
        state: dict[str, Any] = {
            "calibrated_on": source,
            "mode": p.mode,
            "unstimulated_wells": unstimulated_wells,
            "stimulated_wells": stimulated_wells,
            "unstimulated_cells": int(unstimulated.size),
            "stimulated_cells": int(stimulated.size),
            "unstimulated_median_ratio": float(np.median(unstimulated))
            if unstimulated.size
            else math.nan,
            "stimulated_median_ratio": float(np.median(stimulated))
            if stimulated.size
            else math.nan,
            "threshold_method": "fixed" if p.ratio_threshold else p.threshold_method,
            "ratio_threshold": threshold,
            "warnings": warnings,
        }
        return state

    def preview_values(self, state: Any) -> tuple[Any, str]:
        values = self.control_ratios.get("unstimulated")
        if values is None or not len(values):
            values = self._control_ratios(self._wells_of("unstimulated"))
        values = values[values > 0]
        return np.log2(values), "log2 N/C, unstimulated cells"

    def run_warnings(self) -> list[str]:
        manifest = self.manifest()
        steps = [
            s for s in (manifest.get("pipeline") or {}).get("steps", []) if s.get("enabled", True)
        ]
        if not steps:
            return []
        names = [
            str(c.get("name", ""))
            for c in ((manifest.get("plate") or {}).get("metadata") or {}).get("channels", [])
        ]
        number = (
            names.index(self.p.reporter_channel) + 1 if self.p.reporter_channel in names else None
        )
        backgrounds = [s for s in steps if s.get("module") in ("background", "flatfield")]
        warnings = []
        if not any(s.get("module") == "background" for s in backgrounds):
            warnings.append(
                "no background subtraction in the pipeline: the camera offset and background "
                "pull every N/C ratio towards 1"
            )
        for step in backgrounds:
            params = step.get("params") or {}
            channels = params.get("channels")
            applies = not channels or number is None or number in channels
            method = params.get("method", "tophat")
            radius = float(params.get("radius_um", 25.0))
            if step.get("module") != "background" or not applies:
                continue
            if method in ("tophat", "rolling_ball", "paraboloid") and radius < 40:
                warnings.append(
                    f"background '{method}' with radius {radius:g} µm on the reporter channel: "
                    "a radius near the cell size flattens the cytoplasm (use 'constant', or a "
                    "radius of several cell diameters)"
                )
            elif method in ("polynomial", "spline"):
                warnings.append(
                    f"background '{method}' on the reporter channel: under dense cells the "
                    "surface rises into the cytoplasm and N/C ratios are inflated (use 'constant')"
                )
        return warnings

    # --------------------------------------------------------------- per well

    def transform(self, objects: pd.DataFrame, state: Any, well: WellContext) -> pd.DataFrame:
        rows = self.selected(objects)
        nuclei = objects[rows]
        table = self.ratios(objects, nuclei)
        out = pd.DataFrame(index=objects.index)
        for name, values in (("tl_nuc_intensity", table["nuc"]), ("tl_cyto_intensity", table["cyto"]),
                             ("tl_nc_ratio", table["ratio"])):  # fmt: skip
            out[name] = np.nan
            out.loc[nuclei.index, name] = values.to_numpy(float)
        out["tl_log2_nc"] = np.log2(out["tl_nc_ratio"])
        responding = pd.Series(pd.NA, index=objects.index, dtype="boolean")
        usable = table["usable"].to_numpy(bool)
        responding.loc[nuclei.index[usable]] = (
            table["ratio"].to_numpy(float)[usable] > state["ratio_threshold"]
        )
        out["tl_responding"] = responding
        return out

    def summarize(
        self, objects: pd.DataFrame, state: Any, well: WellContext
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        rows = self.selected(objects)
        ratio = objects.loc[rows, "tl_nc_ratio"].astype(float)
        usable = ratio.dropna()
        n_all, n = int(rows.sum()), len(usable)
        responding = objects.loc[rows, "tl_responding"].dropna().astype(bool)
        values: dict[str, Any] = {
            "tl_cells": n,
            "tl_no_cytoplasm_pct": 100 * (n_all - n) / n_all if n_all else math.nan,
            "tl_nc_ratio": float(usable.median()) if n else math.nan,
            "tl_nc_ratio_mean": float(usable.mean()) if n else math.nan,
            "tl_log2_nc": float(np.log2(usable).mean()) if n else math.nan,
            "tl_responding_pct": 100 * float(responding.mean()) if n else math.nan,
        }
        self.wells[well.well] = values
        return values, {}

    def finish_table(self, state: Any, folder: Path, per_well: dict[str, dict[str, Any]]) -> None:
        unstimulated = [w for w in state["unstimulated_wells"] if w in self.wells]
        stimulated = [w for w in state["stimulated_wells"] if w in self.wells]

        def level(wells: list[str], key: str) -> float:
            values = [self.wells[w][key] for w in wells if math.isfinite(self.wells[w][key])]
            return float(np.median(values)) if values else math.nan

        u, s = level(unstimulated, "tl_responding_pct"), level(stimulated, "tl_responding_pct")
        u_ratio = level(unstimulated, "tl_nc_ratio")
        for well, values in self.wells.items():
            values["tl_activation_pct"] = (
                100 * (values["tl_responding_pct"] - u) / (s - u) if math.isfinite(s - u) and s != u
                else math.nan
            )  # fmt: skip
            values["tl_fold_vs_unstimulated"] = (
                values["tl_nc_ratio"] / u_ratio if u_ratio > 0 else math.nan
            )
            per_well.setdefault(well, {}).update(
                {k: values[k] for k in ("tl_activation_pct", "tl_fold_vs_unstimulated")}
            )
        if len(unstimulated) >= 2 and len(stimulated) >= 2:
            for key in ("tl_responding_pct", "tl_log2_nc"):
                a = np.array([self.wells[w][key] for w in unstimulated])
                b = np.array([self.wells[w][key] for w in stimulated])
                spread = abs(float(b.mean() - a.mean()))
                z = 1 - 3 * (a.std(ddof=1) + b.std(ddof=1)) / spread if spread else math.nan
                state[f"z_prime_{key.removeprefix('tl_')}"] = float(z)
            if state["z_prime_responding_pct"] < 0.5:
                state["warnings"].append(
                    f"Z′ of % responding = {state['z_prime_responding_pct']:.2f} (< 0.5): the "
                    "stimulated and unstimulated controls overlap"
                )
        out = self.output_folder(folder)
        calibration = {"module": self.name, "params": self.p.model_dump(mode="json"), **state}
        (out / "calibration.json").write_text(
            json.dumps(calibration, indent=1, default=str), "utf-8"
        )
        table = pd.DataFrame([{"well": w, **v} for w, v in sorted(self.wells.items())])
        table.to_csv(out / "translocation_wells.csv", index=False)
        write_report(out, calibration, table, self.control_ratios)


def write_report(
    folder: Path, calibration: dict[str, Any], table: pd.DataFrame, ratios: dict[str, NDArray[Any]]
) -> Path:
    import base64
    import io

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(7.5, 3.0), dpi=110)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    pooled = [np.log2(v[v > 0]) for v in ratios.values() if len(v)]
    if pooled:
        bins: Any = np.linspace(min(map(np.min, pooled)), max(map(np.max, pooled)), 80)
        for (name, values), colour in zip(ratios.items(), ("#4477aa", "#ee6677"), strict=False):
            if len(values):
                ax.hist(
                    np.log2(values[values > 0]),
                    bins=bins,
                    alpha=0.6,
                    color=colour,
                    label=name,
                    density=True,
                )
        ax.axvline(
            math.log2(calibration["ratio_threshold"]),
            color="black",
            linestyle="--",
            label="threshold",
        )
        ax.legend(fontsize=7)
    ax.set_xlabel("log2 N/C ratio")
    ax.set_ylabel("density")
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", metadata={"Software": None})
    image = base64.b64encode(buffer.getvalue()).decode("ascii")
    shown = [c for c in table.columns if c == "well" or c.startswith("tl_")]

    def cell(value: Any) -> str:
        if isinstance(value, float):
            return f"<td>{value:.3g}</td>" if math.isfinite(value) else "<td></td>"
        return f"<td>{html.escape(str(value))}</td>"

    rows = "".join(
        "<tr>" + "".join(cell(r[c]) for c in shown) + "</tr>" for r in table.to_dict("records")
    )
    head = "".join(f"<th>{html.escape(c)}</th>" for c in shown)
    warnings = "".join(f"<li>{html.escape(w)}</li>" for w in calibration.get("warnings", []))
    z = [f"{label} {calibration[k]:.2f}" for k, label in (("z_prime_responding_pct", "% responding"),
         ("z_prime_log2_nc", "log2 N/C")) if isinstance(calibration.get(k), float) and math.isfinite(calibration[k])]  # fmt: skip
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Translocation</title>
<style>body{{font-family:system-ui,sans-serif;margin:20px;max-width:1200px}}table{{border-collapse:collapse;font-size:12px}}
td,th{{border:1px solid #ccc;padding:2px 6px;text-align:right}}th{{background:#f3f3f3}}.warn{{color:#b00020}}</style></head><body>
<h1>Nuclear translocation</h1>
<p>Calibrated on {html.escape(str(calibration.get("calibrated_on", "")))} ({calibration.get("unstimulated_cells", 0)} cells,
{html.escape(str(calibration.get("threshold_method", "")))}): responding = N/C &gt; {calibration["ratio_threshold"]:.3g}.
Median N/C unstimulated {calibration.get("unstimulated_median_ratio", math.nan):.3g}, stimulated {calibration.get("stimulated_median_ratio", math.nan):.3g}.
{("Z′: " + ", ".join(z) + ".") if z else ""}</p>
{f'<ul class="warn">{warnings}</ul>' if warnings else ""}
<img alt="N/C ratio of the controls" src="data:image/png;base64,{image}">
<h2>Per well</h2><table><tr>{head}</tr>{rows}</table></body></html>"""
    target = folder / "report.html"
    target.write_text(page, encoding="utf-8")
    return target


def preset(
    mapping: dict[str, str],
    *,
    channels: list[str] | None = None,
    nucleus_diameter_um: float = 10.0,
    z_planes: int = 1,
    ring_width_um: float = 2.0,
    mode: str = "agonist",
    blank_wells: bool = False,
) -> Any:
    """Background (a constant: blank wells, else the 5th percentile of each field — keeps the
    cytoplasm), nuclei on the DNA
    channel (Otsu: the mask edge at half height, where the nucleus ends — the triangle threshold
    of the other assays extends masks into the cytoplasm), a cytoplasm ring, measurement and the
    translocation step."""
    from hcs_studio.pipeline.schema import Pipeline

    dna = mapping.get("dna", "DAPI")
    number = channels.index(dna) + 1 if channels and dna in channels else 1
    steps: list[dict[str, Any]] = []
    if z_planes > 1:
        steps.append({"module": "projection", "params": {"method": "max"}})
    steps += [
        {
            "module": "background",
            "params": {
                "method": "constant",
                "constant_source": "blank_wells" if blank_wells else "percentile",
            },
        },
        {
            "module": "segment_classical",
            "params": {
                "channel": number,
                "method": "otsu",
                "min_distance_um": nucleus_diameter_um / 4,
            },
        },
        {"module": "cytoplasm", "params": {"width_um": ring_width_um}},
        {"module": "measure_objects", "params": {"texture": False, "granularity": False}},
        {"module": "qc"},
        {
            "module": "translocation",
            "params": {
                "dna_channel": dna,
                "reporter_channel": mapping.get("reporter", "p65"),
                "mode": mode,
            },
        },
    ]
    return Pipeline.model_validate({"name": "translocation", "steps": steps})
