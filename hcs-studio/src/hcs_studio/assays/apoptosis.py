"""Apoptosis and cytotoxicity (Phase 20c).

A table step on the nuclei (``object_type``) with up to two markers and nuclear morphology:

- **apoptosis marker** (caspase-3/7 dye, cleaved caspase-3, annexin V): mean intensity per nucleus
  — or per cell for membrane stains such as annexin V (``marker_object_type: cells``; cells grown
  from nuclei share the nucleus ID);
- **dead-cell dye** (PI, SYTOX, DRAQ7): mean nuclear intensity;
- **nuclear morphology**: *condensed* nuclei are small and bright (area ≤ ``condensed_area``
  × the vehicle median and DNA brightness ≥ ``condensed_brightness`` × the vehicle median;
  brightness = the 95th percentile of the nucleus's DNA intensity, ``dna_feature`` — the mean
  of a small nucleus is diluted by the dim edge pixels of its mask);
  *fragments* are very small, bright objects (area ≤ ``fragment_area`` × the vehicle median)
  — pieces of fragmented nuclei, left out of cell counts. Or a Phase 8 classifier: give the class
  column (``morphology_column``, e.g. ``class_morphology``) and its apoptotic classes.

**Thresholds** come from the run's controls: by default median + ``threshold_sd`` (5) robust SDs
(1.4826 × MAD) of the log intensity of the vehicle nuclei — robust to the few apoptotic or dead
cells vehicle wells always have (a plain percentile is not: with 3 % positive vehicle cells the
99th percentile falls inside the positives); ``vehicle_percentile`` uses a percentile instead;
``controls`` uses Otsu's threshold of log intensity over the vehicle and positive-control nuclei
(e.g. staurosporine). The positive control's separation from the vehicle is reported as Z′ of %
apoptotic. Time-zero wells (time point 0 h) are never used as vehicle.

**Categories** per nucleus (fragments apart): *necrotic* = dead+ and apoptosis marker−; *late
apoptotic* = dead+ and (marker+ or condensed); *early apoptotic* = dead− and (marker+ or
condensed); *viable* = the rest. % apoptotic = early + late; % dead = late + necrotic.

**Cytostatic vs cytotoxic** (needs a time-zero count: wells with time point 0 h in the layout, or
``time_zero_count``): GR = 2^(log2(x/x0) / log2(x_vehicle/x0)) − 1 on viable counts (Hafner et al.
2016, as in pharmacology). GR < 0: fewer viable cells than at time zero — *cytotoxic*;
0 ≤ GR < ``no_effect_gr`` (0.8): growth slowed — *cytostatic*; above: no effect.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
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
from hcs_studio.processing.base import MODULES, ModuleError

CATEGORIES = ("viable", "early_apoptotic", "late_apoptotic", "necrotic")


def vehicle_threshold(values: Any, percentile: float) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 20:
        raise ModuleError("too few vehicle nuclei to set a threshold")
    return float(np.percentile(x, percentile))


def growth_ratio(x: float, x0: float, x_vehicle: float) -> float:
    """GR (Hafner et al. 2016); NaN when undefined."""
    if not (x0 > 0 and x_vehicle > 0 and x_vehicle != x0):
        return math.nan
    if x <= 0:
        return -1.0
    return float(2 ** (math.log2(x / x0) / math.log2(x_vehicle / x0)) - 1)


class ApoptosisParams(TableParams):
    dna_channel: str = Field(default="DAPI", description="Nuclear stain (morphology).")
    dna_feature: str = Field(
        default="p95", description="DNA brightness feature for condensation (p95, max, mean)."
    )
    apoptosis_channel: str = Field(
        default="", description="Caspase-3/7, cleaved caspase-3 or annexin V; empty = none."
    )
    dead_channel: str = Field(default="", description="PI, SYTOX, DRAQ7; empty = none.")
    marker_object_type: str = Field(
        default="",
        description="Measure the apoptosis marker on these objects (e.g. 'cells' for annexin "
        "V); empty = the nuclei.",
    )
    threshold_method: Literal["vehicle_robust", "vehicle_percentile", "controls"] = Field(
        default="vehicle_robust",
        description="vehicle_robust: median + threshold_sd robust SD of log intensity of the "
        "vehicle nuclei; vehicle_percentile: a percentile of them; controls: Otsu over vehicle "
        "and positive-control nuclei.",
    )
    threshold_sd: float = Field(default=5.0, gt=0)
    vehicle_percentile: float = Field(default=99.0, gt=50, lt=100)
    apoptosis_threshold: float | None = Field(default=None, gt=0, description="Fixed threshold.")
    dead_threshold: float | None = Field(default=None, gt=0, description="Fixed threshold.")
    morphology: bool = Field(default=True, description="Count condensed nuclei as apoptotic.")
    condensed_area: float = Field(default=0.6, gt=0, lt=1)
    condensed_brightness: float = Field(default=1.5, gt=1)
    fragment_area: float = Field(default=0.25, gt=0, lt=1)
    morphology_column: str = Field(
        default="", description="Classifier column instead of the morphology rules."
    )
    apoptotic_classes: list[str] = Field(default_factory=lambda: ["apoptotic", "condensed"])
    time_zero_count: float | None = Field(
        default=None, gt=0, description="Viable nuclei per well at time zero (else 0 h wells)."
    )
    no_effect_gr: float = Field(default=0.8, gt=0, le=1)


@MODULES.register("apoptosis")
class Apoptosis(AssayModule):
    """Apoptosis, dead cells and viable counts, calibrated on the controls (module docs)."""

    name = "apoptosis"
    title = "Assay: apoptosis and cytotoxicity"
    Params = ApoptosisParams
    prefix: ClassVar[str] = "ap_"
    roles: ClassVar[tuple[ChannelRole, ...]] = (
        ChannelRole("dna", "DNA", "DAPI or Hoechst: nuclei and their morphology"),
        ChannelRole(
            "apoptosis", "Apoptosis", "caspase-3/7, cleaved caspase-3 or annexin V", required=False
        ),
        ChannelRole("dead", "Dead", "PI, SYTOX or DRAQ7", required=False),
    )

    def __init__(self, params: Any) -> None:
        super().__init__(params)
        self.wells: dict[str, dict[str, Any]] = {}

    @property
    def p(self) -> ApoptosisParams:
        params: ApoptosisParams = self.params
        return params

    @classmethod
    def preset(cls, mapping: dict[str, str], **options: Any) -> Any:
        return preset(mapping, **options)

    def time_zero_wells(self) -> set[str]:
        layout = self.layout()
        if layout is None:
            return set()
        from hcs_studio.model.plate import well_name

        return {well_name(*k) for k, a in layout.wells.items() if a.time_point_h == 0}

    def well_roles(self) -> dict[str, str]:
        """Roles of the layout; time-zero wells (0 h) get the role 'time_zero', never vehicle."""
        roles = super().well_roles()
        for well in self.time_zero_wells():
            roles[well] = "time_zero"
        return roles

    def columns(self) -> dict[str, str]:
        out = {"dna": channel_column(self.p.dna_channel, self.p.dna_feature), "area": "area_um2"}
        if self.p.apoptosis_channel:
            out["apoptosis"] = channel_column(self.p.apoptosis_channel, "mean")
        if self.p.dead_channel:
            out["dead"] = channel_column(self.p.dead_channel, "mean")
        return out

    def _marker_values(self, objects: pd.DataFrame, nuclei: pd.DataFrame) -> pd.Series:
        """Apoptosis-marker intensity of each nucleus (from its cell for membrane stains)."""
        column = self.columns()["apoptosis"]
        kind = self.p.marker_object_type
        if not kind or kind == str(self.params.object_type):
            return pd.to_numeric(nuclei[column], errors="coerce")
        cells = objects[objects["object_type"] == kind]
        if column not in cells:
            raise ModuleError(f"{column} is not measured on the {kind}")
        index = ["well", "field", "object_id"]
        lookup = cells.set_index(index)[column]
        lookup = lookup[~lookup.index.duplicated()]
        keys = pd.MultiIndex.from_arrays([nuclei[c] for c in index])
        return pd.Series(lookup.reindex(keys).to_numpy(float), index=nuclei.index)

    def preview_values(self, state: Any) -> tuple[Any, str]:
        """Apoptosis-marker (else dead-dye) intensity of the vehicle nuclei, log10."""
        columns = self.columns()
        key = "apoptosis" if "apoptosis" in columns else "dead" if "dead" in columns else ""
        if not key:
            return np.array([]), ""
        vehicle = self.control_objects(VEHICLE_ROLES)
        if vehicle.empty:
            return np.array([]), ""
        if key == "apoptosis":
            values = self._marker_values(self._all_for(VEHICLE_ROLES), vehicle).to_numpy(float)
        else:
            values = pd.to_numeric(vehicle[columns[key]], errors="coerce").to_numpy(float)
        values = values[np.isfinite(values) & (values > 0)]
        channel = self.p.apoptosis_channel if key == "apoptosis" else self.p.dead_channel
        return np.log10(values), f"log10 {channel} intensity, vehicle nuclei"

    def _time_zero(self) -> tuple[float | None, str]:
        """(count, source): a fixed count, or NaN + the time-zero wells (counted later)."""
        if self.p.time_zero_count:
            return float(self.p.time_zero_count), "time_zero_count parameter"
        wells = self.time_zero_wells()
        return (None, "") if not wells else (math.nan, ",".join(sorted(wells)))

    def fit_table(self, sample: pd.DataFrame) -> dict[str, Any]:
        columns = self.columns()
        kind = str(self.params.object_type)
        needed = [columns["dna"], columns["area"]]
        vehicle = self.control_objects(VEHICLE_ROLES)
        source = "vehicle wells"
        if vehicle.empty:
            vehicle = sample[sample["object_type"] == kind]
            source = "all wells (no vehicle wells in the plate layout)"
        self.require(vehicle, needed + [c for k, c in columns.items() if k == "dead"])
        area = float(np.nanmedian(vehicle[columns["area"]].to_numpy(float)))
        dna = float(np.nanmedian(vehicle[columns["dna"]].to_numpy(float)))
        # fragments are left out of the vehicle reference
        whole = vehicle[vehicle[columns["area"]] > self.p.fragment_area * area]
        state: dict[str, Any] = {
            "calibrated_on": source,
            "vehicle_nuclei": len(whole),
            "vehicle_area_um2": area,
            "vehicle_dna_mean": dna,
            "threshold_method": self.p.threshold_method,
            "warnings": [],
        }
        positive = self.control_objects(POSITIVE_ROLES)
        if positive.empty and self.p.threshold_method == "controls":
            state["warnings"].append(
                "no positive-control wells: thresholds from vehicle percentiles"
            )

        everything = self._all_for(VEHICLE_ROLES) if source == "vehicle wells" else sample

        def values_of(key: str, nuclei: pd.DataFrame, objects: pd.DataFrame) -> Any:
            if key == "apoptosis":
                return self._marker_values(objects, nuclei).to_numpy(float)
            return pd.to_numeric(nuclei[columns[key]], errors="coerce").to_numpy(float)

        def threshold(key: str, fixed: float | None) -> float:
            if fixed:
                return float(fixed)
            vehicle_values = values_of(key, whole, everything)
            if self.p.threshold_method == "controls" and not positive.empty:
                positive_values = values_of(key, positive, self._all_for(POSITIVE_ROLES))
                return otsu_log(np.concatenate([vehicle_values, positive_values]))
            if self.p.threshold_method == "vehicle_percentile":
                return vehicle_threshold(vehicle_values, self.p.vehicle_percentile)
            return robust_upper_log(vehicle_values, self.p.threshold_sd)

        if "apoptosis" in columns:
            state["apoptosis_threshold"] = threshold("apoptosis", self.p.apoptosis_threshold)
        if "dead" in columns:
            state["dead_threshold"] = threshold("dead", self.p.dead_threshold)
        t0, t0_source = self._time_zero()
        state["time_zero_source"] = t0_source
        state["time_zero_count"] = t0
        state["warnings"] += [] if "apoptosis" in columns or "dead" in columns else [
            "no apoptosis marker or dead-cell dye: apoptosis from nuclear morphology only"
        ]  # fmt: skip
        return state

    def _all_for(self, roles: tuple[str, ...]) -> pd.DataFrame:
        """All objects (any type) of the control wells, for markers measured on cells."""
        return self.all_objects(roles)

    def transform(self, objects: pd.DataFrame, state: Any, well: WellContext) -> pd.DataFrame:
        columns = self.columns()
        rows = self.selected(objects)
        nuclei = objects[rows]
        area = pd.to_numeric(nuclei[columns["area"]], errors="coerce")
        dna = pd.to_numeric(nuclei[columns["dna"]], errors="coerce")
        fragment = area <= self.p.fragment_area * state["vehicle_area_um2"]
        if self.p.morphology_column:
            if self.p.morphology_column not in nuclei:
                raise ModuleError(
                    f"{self.p.morphology_column} is not in the object table; put the classifier "
                    "step before the apoptosis step"
                )
            condensed = nuclei[self.p.morphology_column].isin(self.p.apoptotic_classes)
        elif self.p.morphology:
            condensed = (area <= self.p.condensed_area * state["vehicle_area_um2"]) & (
                dna >= self.p.condensed_brightness * state["vehicle_dna_mean"]
            )
        else:
            condensed = pd.Series(False, index=nuclei.index)
        condensed &= ~fragment
        marker = (
            self._marker_values(objects, nuclei) > state["apoptosis_threshold"]
            if "apoptosis_threshold" in state
            else pd.Series(False, index=nuclei.index)
        )
        dead = (
            pd.to_numeric(nuclei[columns["dead"]], errors="coerce") > state["dead_threshold"]
            if "dead_threshold" in state
            else pd.Series(False, index=nuclei.index)
        )
        apoptotic = marker | condensed
        category = np.select(
            [fragment, dead & ~apoptotic, dead & apoptotic, apoptotic],
            ["fragment", "necrotic", "late_apoptotic", "early_apoptotic"],
            default="viable",
        )
        out = pd.DataFrame(index=objects.index)
        out["ap_category"] = pd.Series(None, index=objects.index, dtype=object)
        out.loc[nuclei.index, "ap_category"] = category
        out["ap_condensed"] = pd.Series(pd.NA, index=objects.index, dtype="boolean")
        out.loc[nuclei.index, "ap_condensed"] = condensed.to_numpy(bool)
        out["ap_marker_positive"] = pd.Series(pd.NA, index=objects.index, dtype="boolean")
        out.loc[nuclei.index, "ap_marker_positive"] = marker.to_numpy(bool)
        out["ap_dead_positive"] = pd.Series(pd.NA, index=objects.index, dtype="boolean")
        out.loc[nuclei.index, "ap_dead_positive"] = dead.to_numpy(bool)
        return out

    def summarize(
        self, objects: pd.DataFrame, state: Any, well: WellContext
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        category = objects["ap_category"].dropna()
        cells = category[category != "fragment"]
        n = len(cells)
        values: dict[str, Any] = {
            "ap_nuclei": n,
            "ap_fragments": int((category == "fragment").sum()),
        }
        for name in CATEGORIES:
            values[f"ap_{name}_pct"] = 100 * float((cells == name).mean()) if n else math.nan
        values["ap_apoptotic_pct"] = (
            values["ap_early_apoptotic_pct"] + values["ap_late_apoptotic_pct"]
        )
        values["ap_dead_pct"] = values["ap_late_apoptotic_pct"] + values["ap_necrotic_pct"]
        condensed = objects["ap_condensed"].dropna().astype(bool)
        values["ap_condensed_pct"] = 100 * float(condensed[category.index.isin(cells.index)].mean()) if n else math.nan  # fmt: skip
        values["ap_viable_count"] = int((cells == "viable").sum())
        self.wells[well.well] = values
        return values, {}

    def finish_table(self, state: Any, folder: Path, per_well: dict[str, dict[str, Any]]) -> None:
        """Plate-level values: viable count vs vehicle, GR and the response class, Z′."""
        roles = self.well_roles()
        vehicle = [w for w, r in roles.items() if r in VEHICLE_ROLES and w in self.wells]
        positive = [w for w, r in roles.items() if r in POSITIVE_ROLES and w in self.wells]
        viable = {w: float(v["ap_viable_count"]) for w, v in self.wells.items()}
        x_vehicle = float(np.median([viable[w] for w in vehicle])) if vehicle else math.nan
        t0 = state.get("time_zero_count")
        if state.get("time_zero_source") and t0 is not None and not math.isfinite(t0):
            wells = state["time_zero_source"].split(",")
            counts = [viable[w] for w in wells if w in viable]
            t0 = float(np.median(counts)) if counts else math.nan
            state["time_zero_count"] = t0
        reference_wells = self.time_zero_wells()
        for well, values in self.wells.items():
            values["ap_viable_vs_vehicle"] = viable[well] / x_vehicle if x_vehicle > 0 else math.nan
            usable = t0 is not None and math.isfinite(t0) and well not in reference_wells
            gr = growth_ratio(viable[well], t0, x_vehicle) if usable else math.nan
            values["ap_gr"] = gr
            values["ap_response"] = (
                "" if not math.isfinite(gr) else "cytotoxic" if gr < 0
                else "cytostatic" if gr < self.p.no_effect_gr else "no effect"
            )  # fmt: skip
            per_well.setdefault(well, {}).update(
                {k: values[k] for k in ("ap_viable_vs_vehicle", "ap_gr", "ap_response")}
            )
        if len(vehicle) >= 2 and len(positive) >= 2:
            a = np.array([self.wells[w]["ap_apoptotic_pct"] for w in vehicle])
            b = np.array([self.wells[w]["ap_apoptotic_pct"] for w in positive])
            spread = abs(float(b.mean() - a.mean()))
            state["z_prime_apoptotic"] = (
                float(1 - 3 * (a.std(ddof=1) + b.std(ddof=1)) / spread) if spread else math.nan
            )
        out = self.output_folder(folder)
        calibration = {"module": self.name, "params": self.p.model_dump(mode="json"), **state}
        (out / "calibration.json").write_text(
            json.dumps(calibration, indent=1, default=str), "utf-8"
        )
        table = pd.DataFrame([{"well": w, **v} for w, v in sorted(self.wells.items())])
        table.to_csv(out / "apoptosis_wells.csv", index=False)
        write_report(out, calibration, table, roles)


def write_report(
    folder: Path, calibration: dict[str, Any], table: pd.DataFrame, roles: dict[str, str]
) -> Path:
    import base64
    import io

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(7.5, 3.2), dpi=110)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    if not table.empty:
        bottom = np.zeros(len(table))
        for name, colour in zip(
            CATEGORIES, ("#88ccaa", "#ddcc77", "#cc6677", "#555555"), strict=True
        ):
            heights = table[f"ap_{name}_pct"].fillna(0).to_numpy()
            ax.bar(
                range(len(table)),
                heights,
                bottom=bottom,
                color=colour,
                label=name.replace("_", " "),
            )
            bottom += heights
        ax.set_xticks(range(len(table)), table["well"], rotation=90, fontsize=6)
        ax.legend(fontsize=7, ncols=4)
    ax.set_ylabel("% of nuclei")
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", metadata={"Software": None})
    image = base64.b64encode(buffer.getvalue()).decode("ascii")
    shown = [c for c in table.columns if c == "well" or c.startswith("ap_")]

    def cell(value: Any) -> str:
        if isinstance(value, float):
            return f"<td>{value:.2f}</td>" if math.isfinite(value) else "<td></td>"
        return f"<td>{html.escape(str(value))}</td>"

    rows = "".join(
        "<tr>" + "".join(cell(r[c]) for c in shown) + "</tr>" for r in table.to_dict("records")
    )
    head = "".join(f"<th>{html.escape(c)}</th>" for c in shown)
    warnings = "".join(f"<li>{html.escape(w)}</li>" for w in calibration.get("warnings", []))
    thresholds = ", ".join(
        f"{k.replace('_threshold', '')} {calibration[k]:.4g}"
        for k in ("apoptosis_threshold", "dead_threshold") if k in calibration
    )  # fmt: skip
    z = calibration.get("z_prime_apoptotic")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Apoptosis</title>
<style>body{{font-family:system-ui,sans-serif;margin:20px;max-width:1200px}}table{{border-collapse:collapse;font-size:12px}}
td,th{{border:1px solid #ccc;padding:2px 6px;text-align:right}}th{{background:#f3f3f3}}.warn{{color:#b00020}}</style></head><body>
<h1>Apoptosis and cytotoxicity</h1>
<p>Calibrated on {html.escape(str(calibration.get("calibrated_on", "")))} ({calibration.get("vehicle_nuclei", 0)} nuclei,
{html.escape(str(calibration.get("threshold_method", "")))}): thresholds {html.escape(thresholds or "none")};
vehicle nuclear area {calibration.get("vehicle_area_um2", 0):.1f} µm².
{f"Time-zero count {calibration['time_zero_count']:.0f} ({html.escape(str(calibration.get('time_zero_source', '')))})." if isinstance(calibration.get("time_zero_count"), float) and math.isfinite(calibration["time_zero_count"]) else "No time-zero count: GR not computed."}
{f"Z′ (% apoptotic, positive control vs vehicle) = {z:.2f}." if isinstance(z, float) and math.isfinite(z) else ""}</p>
{f'<ul class="warn">{warnings}</ul>' if warnings else ""}
<img alt="categories" src="data:image/png;base64,{image}">
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
) -> Any:
    """Background removal, nuclei on the DNA channel (triangle threshold: dim and condensed nuclei
    alike), measurement and the apoptosis step."""
    from hcs_studio.pipeline.schema import Pipeline

    dna = mapping.get("dna", "DAPI")
    number = channels.index(dna) + 1 if channels and dna in channels else 1
    steps: list[dict[str, Any]] = []
    if z_planes > 1:
        steps.append({"module": "projection", "params": {"method": "max"}})
    steps += [
        {
            "module": "background",
            "params": {"method": "tophat", "radius_um": 2.5 * nucleus_diameter_um},
        },
        {
            "module": "segment_classical",
            "params": {
                "channel": number,
                "method": "triangle",
                "min_distance_um": nucleus_diameter_um / 4,
            },
        },
        {"module": "measure_objects", "params": {"texture": False, "granularity": False}},
        {"module": "qc"},
        {
            "module": "apoptosis",
            "params": {
                "dna_channel": dna,
                "apoptosis_channel": mapping.get("apoptosis", ""),
                "dead_channel": mapping.get("dead", ""),
            },
        },
    ]
    return Pipeline.model_validate({"name": "apoptosis", "steps": steps})
