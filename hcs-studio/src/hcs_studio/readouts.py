"""The readout registry (PROMPT_PLATFORM.md §2.6).

Every module registers the readouts it produces: ID pattern, display name, unit, level
(object / field / well / fit), direction of effect, default normalization and a one-line
description. Plate maps, pharmacology, QC, hit calling, profiling and the assistant look
readouts up here instead of relying on column names.

Most column names are generated (``<label set>_<channel>_mean_<stat>``), so a readout is a
*pattern* with placeholders: ``{labels}``, ``{channel}``, ``{child}``, ``{name}`` … match one
or more ``_``-separated tokens; ``{stat:mean|median|std}`` lists the allowed values. A
column resolves to the most specific matching pattern (the one with the most literal text).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cache, lru_cache
from typing import Literal

Level = Literal["object", "field", "well", "fit"]
Direction = Literal["up", "down", "either", "none"]
Normalization = Literal["percent_of_control", "percent_inhibition", "robust_z", "none"]

#: Lazy: earlier placeholders take as few tokens as possible, so "cells_alexa_488_mean_mean"
#: is label set "cells" + channel "alexa_488" (label-set names are usually one word;
#: channel names often contain "_").
_TOKEN = r"[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*?"
_LEVEL_RANK = {"object": 0, "fit": 1, "field": 2, "well": 3}
_PLACEHOLDER = re.compile(r"\{(\w+)(?::([^}]*))?\}")


@dataclass(frozen=True)
class ReadoutSpec:
    pattern: str
    display: str
    unit: str = ""
    level: Level = "well"
    direction: Direction = "either"
    default_normalization: Normalization = "none"
    description: str = ""
    module: str = ""

    @property
    def specificity(self) -> tuple[int, int]:
        """Literal text plus constrained placeholders (choices count their shortest value,
        integers 1); ties go to the well level, then field, fit and object."""
        constrained = 0
        for m in _PLACEHOLDER.finditer(self.pattern):
            if m.group(2) in ("int", "any"):
                constrained += 1 if m.group(2) == "int" else 0
            elif m.group(2):
                constrained += min(len(c) for c in m.group(2).split("|"))
        literal = len(_PLACEHOLDER.sub("", self.pattern))
        return (literal + constrained, _LEVEL_RANK[self.level])

    def regex(self) -> re.Pattern[str]:
        return _compile(self.pattern)


@cache
def _compile(pattern: str) -> re.Pattern[str]:
    out, last = [], 0
    for m in _PLACEHOLDER.finditer(pattern):
        out.append(re.escape(pattern[last : m.start()]))
        name, choices = m.group(1), m.group(2)
        if choices == "int":
            body = r"\d+"
        elif choices == "any":  # free text (original channel names, with spaces)
            body = r".+?"
        elif choices:
            body = "|".join(re.escape(c) for c in sorted(choices.split("|"), key=len, reverse=True))
        else:
            body = _TOKEN
        out.append(f"(?P<{name}>{body})")
        last = m.end()
    out.append(re.escape(pattern[last:]))
    return re.compile("".join(out))


@dataclass(frozen=True)
class Readout:
    """A concrete readout: a column name resolved against its spec."""

    id: str
    spec: ReadoutSpec
    values: dict[str, str] = field(default_factory=dict)

    @property
    def display(self) -> str:
        values = dict(self.values)
        if "stat" in values:
            values["stat"] = STAT_TEXT.get(values["stat"], values["stat"])
        return self.spec.display.format_map(_Defaults(values))

    @property
    def unit(self) -> str:
        return self.spec.unit

    @property
    def label(self) -> str:
        return f"{self.display} ({self.unit})" if self.unit else self.display


class _Defaults(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return key


class ReadoutRegistry:
    def __init__(self) -> None:
        self._specs: list[ReadoutSpec] = []

    def register(self, *specs: ReadoutSpec) -> None:
        known = {(s.pattern, s.level) for s in self._specs}
        for spec in specs:
            if (spec.pattern, spec.level) in known:
                raise ValueError(f"Readout {spec.pattern!r} ({spec.level}) is already registered")
            spec.regex()  # fail early on a bad pattern
            self._specs.append(spec)
            known.add((spec.pattern, spec.level))
        _resolve_cached.cache_clear()

    def specs(self, level: Level | None = None, module: str | None = None) -> list[ReadoutSpec]:
        return [
            s
            for s in self._specs
            if (level is None or s.level == level) and (module is None or s.module == module)
        ]

    def resolve(self, column: str, level: Level | None = None) -> Readout | None:
        return _resolve_cached(self, column, level)

    def _resolve(self, column: str, level: Level | None) -> Readout | None:
        best: tuple[tuple[int, int], ReadoutSpec, dict[str, str]] | None = None
        for spec in self._specs:
            if level is not None and spec.level != level:
                continue
            match = spec.regex().fullmatch(column)
            if match and (best is None or spec.specificity > best[0]):
                best = (spec.specificity, spec, {k: v for k, v in match.groupdict().items() if v})
        return Readout(column, best[1], best[2]) if best else None

    def unknown(self, columns: list[str], level: Level | None = None) -> list[str]:
        return [c for c in columns if self.resolve(c, level) is None]


@lru_cache(maxsize=65536)
def _resolve_cached(registry: ReadoutRegistry, column: str, level: Level | None) -> Readout | None:
    return registry._resolve(column, level)


READOUTS = ReadoutRegistry()


def resolve(column: str, level: Level | None = None) -> Readout | None:
    return READOUTS.resolve(column, level)


def label(column: str) -> str:
    """Display name with unit, or the column itself when it is not registered."""
    found = READOUTS.resolve(column)
    return found.label if found else column


# =========================================================================
# Readouts of Phases 5–15 (retrofit). New modules register theirs next to their code
# with READOUTS.register(...).
# =========================================================================

STAT = "{stat:mean|median|std}"
STAT_TEXT = {"mean": "mean", "median": "median", "std": "SD"}

#: object feature → (display, unit, direction, description)
OBJECT_FEATURES: dict[str, tuple[str, str, Direction, str]] = {
    "area_um2": ("area", "µm²", "either", "object area (2D)"),
    "volume_um3": ("volume", "µm³", "either", "object volume (3D)"),
    "perimeter_um": ("perimeter", "µm", "either", "boundary length (2D)"),
    "surface_area_um2": ("surface area", "µm²", "either", "surface (3D)"),
    "equivalent_diameter_um": (
        "equivalent diameter",
        "µm",
        "either",
        "diameter of a circle or sphere of the same size",
    ),
    "major_axis_um": (
        "major axis",
        "µm",
        "either",
        "major axis of the ellipse with the same moments",
    ),
    "minor_axis_um": (
        "minor axis",
        "µm",
        "either",
        "minor axis of the ellipse with the same moments",
    ),
    "eccentricity": ("eccentricity", "", "either", "0 = circle, towards 1 = elongated"),
    "orientation_deg": ("orientation", "°", "none", "angle of the major axis"),
    "form_factor": ("form factor", "", "either", "4π·area / perimeter²; 1 for a circle"),
    "solidity": ("solidity", "", "either", "area / convex area"),
    "extent": ("extent", "", "either", "area / bounding-box area"),
    "centroid_x_um": ("centroid x", "µm", "none", "centre of mass in the field"),
    "centroid_y_um": ("centroid y", "µm", "none", "centre of mass in the field"),
    "centroid_z_um": ("centroid z", "µm", "none", "centre of mass in the field"),
    "well_x_um": ("well position x", "µm", "none", "centroid in the stitched well"),
    "well_y_um": ("well position y", "µm", "none", "centroid in the stitched well"),
    "nearest_neighbour_um": (
        "nearest neighbour distance",
        "µm",
        "either",
        "distance to the nearest object of the same type",
    ),
    "neighbours_within_radius": (
        "neighbours within radius",
        "objects",
        "either",
        "objects of the same type within the neighbour radius",
    ),
    "zernike_{n:int}": ("Zernike moment {n}", "", "either", "Zernike shape moment"),
}

#: per-channel feature (after ``<channel>_``) → (display, unit, direction, description)
CHANNEL_FEATURES: dict[str, tuple[str, str, Direction, str]] = {
    "mean": ("{channel} mean intensity", "a.u.", "either", "mean of the object's pixels"),
    "median": ("{channel} median intensity", "a.u.", "either", "median of the object's pixels"),
    "sd": ("{channel} intensity SD", "a.u.", "either", "SD of the object's pixels"),
    "min": ("{channel} minimum intensity", "a.u.", "either", "minimum of the object's pixels"),
    "max": ("{channel} maximum intensity", "a.u.", "either", "maximum of the object's pixels"),
    "integrated": (
        "{channel} integrated intensity",
        "a.u.",
        "either",
        "sum of the object's pixels",
    ),
    "p{p:05|25|75|95}": (
        "{channel} intensity percentile {p}",
        "a.u.",
        "either",
        "percentile of the object's pixels",
    ),
    "radial_{ring:outer|middle|inner}": (
        "{channel} {ring} ring fraction",
        "",
        "either",
        "fraction of the integrated intensity in the ring",
    ),
    "center_periphery_ratio": (
        "{channel} centre/periphery ratio",
        "",
        "either",
        "inner / outer ring fraction",
    ),
    "nc_ratio": (
        "{channel} nuclear/cytoplasmic ratio",
        "",
        "either",
        "nuclear mean / cytoplasmic mean",
    ),
    "cytoplasm_mean": (
        "{channel} cytoplasmic mean",
        "a.u.",
        "either",
        "mean of the cell minus its nucleus",
    ),
    "haralick_{feature}_s{d:int}": (
        "{channel} Haralick {feature} (d={d})",
        "",
        "either",
        "Haralick texture feature, averaged over directions",
    ),
    "granularity_r{r:int}": (
        "{channel} granularity r{r}",
        "%",
        "either",
        "signal removed by grey opening at this radius",
    ),
}

CHILD_FEATURES: dict[str, tuple[str, str, Direction, str]] = {
    "{child}_count": ("{child} per object", "objects", "either", "number of child objects"),
    "{child}_mean_area_um2": ("{child} mean area", "µm²", "either", "mean size of the children"),
    "{child}_{channel}_total_intensity": (
        "{child} {channel} total intensity",
        "a.u.",
        "either",
        "summed intensity of the children",
    ),
    "{child}_{channel}_mean_intensity": (
        "{child} {channel} mean intensity",
        "a.u.",
        "either",
        "mean intensity of the children",
    ),
}

COLOC_METRICS = (
    "pearson|spearman|icq|m1|m2|tm1|tm2|overlap|overlap_above_thresholds|p_value|"
    "random_r_mean|random_r_sd|threshold_a|threshold_b|costes_slope|n_pixels"
)


def _object_and_well(
    feature: str, display: str, unit: str, direction: Direction, description: str, module: str
) -> list[ReadoutSpec]:
    return [
        ReadoutSpec(feature, display, unit, "object", direction, "none", description, module),
        ReadoutSpec(
            "{labels}_" + feature + "_" + STAT,
            "{labels} " + display + " ({stat})",
            unit,
            "well",
            direction,
            "robust_z",
            f"per-well statistic of: {description}",
            module,
        ),
    ]


#: measure_spheroids features (Phase 21) → (display, unit, direction, description)
SPHEROID_FEATURES: dict[str, tuple[str, str, Direction, str]] = {
    "volume_sphere_um3": ("volume (sphere)", "µm³", "either", "π/6 · equivalent diameter³"),
    "volume_ellipsoid_um3": ("volume (ellipsoid)", "µm³", "either", "π/6 · major · minor²"),
    "circularity": ("circularity", "", "either", "4π·area / perimeter²; 1 for a circle"),
    "compactness": ("compactness", "", "either", "spread around the centroid; 1 for a disc"),
    "offcentre": ("off-centre", "", "up", "centroid distance from the field centre / half field"),
    "pixel_um": ("pixel size", "µm", "none", "pixel size of the field"),
    "field_height_px": ("field height", "px", "none", "field height"),
    "field_width_px": ("field width", "px", "none", "field width"),
    "mean_intensity": ("mean intensity", "", "either", "mean intensity inside the object"),
    "background": ("local background", "", "none", "median intensity around the object"),
    "darkness": ("darkness", "", "either", "1 − interior / local background (brightfield)"),
    "edge_sigma_um": ("edge blur", "µm", "up", "Gaussian blur of the outline (focus)"),
}


def _builtin() -> list[ReadoutSpec]:
    specs: list[ReadoutSpec] = []
    for feat, (display, unit, direction, text) in OBJECT_FEATURES.items():
        specs += _object_and_well(feat, display, unit, direction, text, "measure_objects")
    for feat, (display, unit, direction, text) in CHANNEL_FEATURES.items():
        specs += _object_and_well(
            "{channel}_" + feat, display, unit, direction, text, "measure_objects"
        )
    for feat, (display, unit, direction, text) in CHILD_FEATURES.items():
        specs += _object_and_well(feat, display, unit, direction, text, "measure_objects")
    for feat, (display, unit, direction, text) in SPHEROID_FEATURES.items():
        specs += _object_and_well(feat, display, unit, direction, text, "measure_spheroids")
    w: Level = "well"
    specs += [
        # well summary basics (output/tables.py)
        ReadoutSpec(
            "fields",
            "fields used",
            "fields",
            w,
            "none",
            "none",
            "fields measured in the well",
            "tables",
        ),
        ReadoutSpec(
            "fields_excluded",
            "fields excluded",
            "fields",
            w,
            "down",
            "none",
            "fields left out by QC",
            "qc",
        ),
        ReadoutSpec(
            "fields_out_of_focus",
            "fields out of focus",
            "fields",
            w,
            "down",
            "none",
            "fields flagged out of focus",
            "qc",
        ),
        ReadoutSpec(
            "fields_saturated",
            "fields saturated",
            "fields",
            w,
            "down",
            "none",
            "fields flagged saturated",
            "qc",
        ),
        ReadoutSpec(
            "area_mm2", "imaged area", "mm²", w, "none", "none", "area of the fields used", "tables"
        ),
        ReadoutSpec(
            "low_count",
            "low count",
            "",
            w,
            "down",
            "none",
            "well count below the low-count fraction of the plate median",
            "qc",
        ),
        ReadoutSpec(
            "{labels}_count",
            "{labels} count",
            "objects",
            w,
            "either",
            "percent_of_control",
            "objects per well (excluded fields left out)",
            "qc",
        ),
        ReadoutSpec(
            "{labels}_per_mm2",
            "{labels} density",
            "objects/mm²",
            w,
            "either",
            "percent_of_control",
            "objects per mm² of imaged area",
            "qc",
        ),
        ReadoutSpec(
            "{labels}_objects_measured",
            "{labels} measured",
            "objects",
            w,
            "none",
            "none",
            "objects with measurements",
            "measure_objects",
        ),
        # colocalization
        ReadoutSpec(
            "coloc_{pair:any}_{metric:" + COLOC_METRICS + "}_median",
            "colocalization {pair}: {metric} (median)",
            "",
            w,
            "either",
            "none",
            "median of the per-object/field colocalization statistic",
            "colocalization",
        ),
        ReadoutSpec(
            "{metric:" + COLOC_METRICS + "}",
            "colocalization {metric}",
            "",
            "object",
            "either",
            "none",
            "colocalization statistic",
            "colocalization",
        ),
        # table steps (Phase 8)
        # assays (Phase 20)
        ReadoutSpec(
            "cc_{phase:sub_g1|g1|s|g2m|polyploid}_pct",
            "cell cycle: % {phase}",
            "%",
            w,
            "either",
            "none",
            "nuclei in this phase (Dean–Jett–Fox fit of DNA content; vehicle G1 = 2N)",
            "cell_cycle",
        ),
        ReadoutSpec(
            "cc_edu_{phase:sub_g1|g1|s|g2m|polyploid}_pct",
            "cell cycle (EdU gating): % {phase}",
            "%",
            w,
            "either",
            "none",
            "nuclei in this phase by EdU vs DNA content gating",
            "cell_cycle",
        ),
        ReadoutSpec(
            "cc_edu_pct",
            "EdU-positive",
            "%",
            w,
            "either",
            "none",
            "nuclei above the EdU threshold from the vehicle wells",
            "cell_cycle",
        ),
        ReadoutSpec(
            "cc_mitotic_index_pct",
            "mitotic index (pH3)",
            "%",
            w,
            "either",
            "none",
            "pH3-positive nuclei",
            "cell_cycle",
        ),
        ReadoutSpec(
            "cc_g2m_arrest_index",
            "G2/M arrest index",
            "",
            w,
            "up",
            "none",
            "% G2/M of the well / % G2/M of the vehicle wells",
            "cell_cycle",
        ),
        ReadoutSpec(
            "cc_g1_peak_n",
            "G1 peak (N)",
            "N",
            w,
            "none",
            "none",
            "position of the well's G1 peak; 2 = as in the vehicle wells",
            "cell_cycle",
        ),
        ReadoutSpec(
            "cc_g1_cv_pct",
            "G1 CV",
            "%",
            w,
            "down",
            "none",
            "width of the G1 and G2 peaks (coefficient of variation)",
            "cell_cycle",
        ),
        ReadoutSpec(
            "cc_nuclei",
            "cell cycle: nuclei",
            "nuclei",
            w,
            "none",
            "none",
            "nuclei analysed by the cell-cycle step",
            "cell_cycle",
        ),
        ReadoutSpec(
            "cc_dna_n",
            "DNA content",
            "N",
            "object",
            "none",
            "none",
            "integrated DNA stain scaled so the vehicle G1 peak = 2N",
            "cell_cycle",
        ),
        ReadoutSpec(
            "sp_volume_um3",
            "spheroid volume",
            "µm³",
            w,
            "down",
            "percent_of_control",
            "volume of the well's spheroid (sphere or ellipsoid estimate)",
            "spheroid",
        ),
        ReadoutSpec(
            "sp_volume_{x:sphere|ellipsoid}_um3",
            "spheroid volume ({x})",
            "µm³",
            w,
            "down",
            "percent_of_control",
            "volume estimate of the well's spheroid",
            "spheroid",
        ),
        ReadoutSpec(
            "sp_volume_vs_vehicle_pct",
            "spheroid volume vs vehicle",
            "%",
            w,
            "down",
            "none",
            "volume / median volume of the vehicle wells",
            "spheroid",
        ),
        ReadoutSpec(
            "sp_area_um2",
            "spheroid area",
            "µm²",
            w,
            "down",
            "percent_of_control",
            "projected area of the well's spheroid",
            "spheroid",
        ),
        ReadoutSpec(
            "sp_diameter_um",
            "spheroid diameter",
            "µm",
            w,
            "down",
            "percent_of_control",
            "equivalent diameter of the well's spheroid",
            "spheroid",
        ),
        ReadoutSpec(
            "sp_{x:circularity|solidity|compactness}",
            "spheroid {x}",
            "",
            w,
            "either",
            "none",
            "shape of the well's spheroid",
            "spheroid",
        ),
        ReadoutSpec(
            "sp_darkness",
            "spheroid darkness",
            "",
            w,
            "either",
            "none",
            "1 − interior / background (brightfield)",
            "spheroid",
        ),
        ReadoutSpec(
            "sp_edge_sigma_um",
            "spheroid edge blur",
            "µm",
            w,
            "up",
            "none",
            "Gaussian blur of the outline (focus)",
            "spheroid",
        ),
        ReadoutSpec(
            "sp_offcentre",
            "spheroid off-centre",
            "",
            w,
            "up",
            "none",
            "centroid distance from the field centre / half field",
            "spheroid",
        ),
        ReadoutSpec(
            "sp_objects",
            "objects detected",
            "objects",
            w,
            "none",
            "none",
            "spheroids and debris detected in the well",
            "spheroid",
        ),
        ReadoutSpec(
            "sp_found",
            "spheroid found",
            "",
            w,
            "none",
            "none",
            "1 when the well has a spheroid",
            "spheroid",
        ),
        ReadoutSpec(
            "sp_field",
            "spheroid field",
            "",
            w,
            "none",
            "none",
            "field holding the well's spheroid",
            "spheroid",
        ),
        ReadoutSpec(
            "sp_qc_ok",
            "spheroid QC passed",
            "",
            w,
            "none",
            "none",
            "0 when a failing flag is set (exclude_flags)",
            "spheroid",
        ),
        ReadoutSpec(
            "pf_{m:ki67|edu|ph3}_pos_pct",
            "{m} positive",
            "%",
            w,
            "down",
            "percent_of_control",
            "nuclei above the marker threshold",
            "proliferation",
        ),
        ReadoutSpec(
            "pf_{m:ki67|edu|ph3}_median",
            "{m} intensity (median)",
            "",
            w,
            "down",
            "none",
            "median nuclear intensity of the marker",
            "proliferation",
        ),
        ReadoutSpec(
            "pf_{m:ki67|edu|ph3}_pos_vs_vehicle",
            "{m} positive vs vehicle",
            "×",
            w,
            "down",
            "none",
            "% positive / vehicle median",
            "proliferation",
        ),
        ReadoutSpec(
            "pf_nuclei",
            "proliferation: nuclei",
            "nuclei",
            w,
            "none",
            "none",
            "nuclei analysed (micronuclei apart)",
            "proliferation",
        ),
        ReadoutSpec(
            "pf_mn_pct",
            "nuclei with micronuclei",
            "%",
            w,
            "up",
            "none",
            "nuclei with at least one micronucleus",
            "proliferation",
        ),
        ReadoutSpec(
            "pf_mn_per_100",
            "micronuclei per 100 nuclei",
            "",
            w,
            "up",
            "none",
            "micronuclei per 100 nuclei",
            "proliferation",
        ),
        ReadoutSpec(
            "pf_mn_per_100_vs_vehicle",
            "micronuclei vs vehicle",
            "×",
            w,
            "up",
            "none",
            "micronuclei per 100 nuclei / vehicle median",
            "proliferation",
        ),
        ReadoutSpec(
            "pf_cells",
            "proliferation: cells",
            "cells",
            w,
            "none",
            "none",
            "cells with at least one nucleus",
            "proliferation",
        ),
        ReadoutSpec(
            "pf_{x:multinucleated|binucleated}_pct",
            "{x} cells",
            "%",
            w,
            "up",
            "none",
            "cells with ≥ 2 (multinucleated) or exactly 2 (binucleated) nuclei",
            "proliferation",
        ),
        ReadoutSpec(
            "pf_multinucleated_vs_vehicle",
            "multinucleated vs vehicle",
            "×",
            w,
            "up",
            "none",
            "% multinucleated / vehicle median",
            "proliferation",
        ),
        ReadoutSpec(
            "pf_nuclei_per_cell",
            "nuclei per cell",
            "nuclei",
            w,
            "up",
            "none",
            "mean nuclei per cell",
            "proliferation",
        ),
        ReadoutSpec(
            "pf_micronuclei",
            "micronuclei",
            "",
            "object",
            "up",
            "none",
            "micronuclei of the nucleus",
            "proliferation",
        ),
        ReadoutSpec(
            "tl_responding_pct",
            "responding cells",
            "%",
            w,
            "either",
            "percent_inhibition",
            "cells with N/C above the threshold from unstimulated controls",
            "translocation",
        ),
        ReadoutSpec(
            "tl_nc_ratio",
            "N/C ratio (median)",
            "",
            w,
            "either",
            "percent_inhibition",
            "median nuclear/cytoplasmic mean-intensity ratio",
            "translocation",
        ),
        ReadoutSpec(
            "tl_nc_ratio_mean",
            "N/C ratio (mean)",
            "",
            w,
            "either",
            "none",
            "mean nuclear/cytoplasmic ratio",
            "translocation",
        ),
        ReadoutSpec(
            "tl_log2_nc",
            "log2 N/C",
            "",
            w,
            "either",
            "percent_inhibition",
            "mean log2 nuclear/cytoplasmic ratio",
            "translocation",
        ),
        ReadoutSpec(
            "tl_activation_pct",
            "activation",
            "%",
            w,
            "either",
            "none",
            "% responding scaled from unstimulated (0 %) to stimulated (100 %) controls",
            "translocation",
        ),
        ReadoutSpec(
            "tl_fold_vs_unstimulated",
            "N/C vs unstimulated",
            "×",
            w,
            "either",
            "none",
            "median N/C / median N/C of the unstimulated wells",
            "translocation",
        ),
        ReadoutSpec(
            "tl_cells",
            "translocation: cells",
            "cells",
            w,
            "none",
            "none",
            "cells with a usable N/C ratio",
            "translocation",
        ),
        ReadoutSpec(
            "tl_no_cytoplasm_pct",
            "cells without cytoplasm",
            "%",
            w,
            "down",
            "none",
            "nuclei without a usable cytoplasm (crowded or at the image edge)",
            "translocation",
        ),
        ReadoutSpec(
            "tl_{x:nuc_intensity|cyto_intensity}",
            "reporter {x}",
            "",
            "object",
            "either",
            "none",
            "reporter intensity in the nucleus / cytoplasm",
            "translocation",
        ),
        ReadoutSpec(
            "tl_nc_ratio",
            "N/C ratio",
            "",
            "object",
            "either",
            "none",
            "nuclear/cytoplasmic ratio of the cell",
            "translocation",
        ),
        ReadoutSpec(
            "tl_log2_nc",
            "log2 N/C",
            "",
            "object",
            "either",
            "none",
            "log2 nuclear/cytoplasmic ratio of the cell",
            "translocation",
        ),
        ReadoutSpec(
            "ap_{cat:viable|early_apoptotic|late_apoptotic|necrotic}_pct",
            "% {cat}",
            "%",
            w,
            "either",
            "none",
            "nuclei in this category (fragments apart)",
            "apoptosis",
        ),
        ReadoutSpec(
            "ap_apoptotic_pct",
            "apoptotic",
            "%",
            w,
            "up",
            "none",
            "early + late apoptotic nuclei",
            "apoptosis",
        ),
        ReadoutSpec(
            "ap_dead_pct",
            "dead",
            "%",
            w,
            "up",
            "none",
            "late apoptotic + necrotic nuclei (dead-cell dye positive)",
            "apoptosis",
        ),
        ReadoutSpec(
            "ap_condensed_pct",
            "condensed nuclei",
            "%",
            w,
            "up",
            "none",
            "small, bright nuclei (or the classifier's apoptotic classes)",
            "apoptosis",
        ),
        ReadoutSpec(
            "ap_viable_count",
            "viable cells",
            "cells",
            w,
            "down",
            "percent_of_control",
            "nuclei that are neither apoptotic nor dead",
            "apoptosis",
        ),
        ReadoutSpec(
            "ap_viable_vs_vehicle",
            "viable vs vehicle",
            "×",
            w,
            "down",
            "none",
            "viable count / median viable count of the vehicle wells",
            "apoptosis",
        ),
        ReadoutSpec(
            "ap_gr",
            "GR (viable cells)",
            "",
            w,
            "down",
            "none",
            "growth-rate inhibition from the time-zero count: < 0 cytotoxic, 0–1 cytostatic",
            "apoptosis",
        ),
        ReadoutSpec(
            "ap_nuclei",
            "apoptosis: nuclei",
            "nuclei",
            w,
            "none",
            "none",
            "nuclei analysed (fragments apart)",
            "apoptosis",
        ),
        ReadoutSpec(
            "ap_fragments",
            "nuclear fragments",
            "objects",
            w,
            "up",
            "none",
            "very small bright objects (pieces of fragmented nuclei)",
            "apoptosis",
        ),
        ReadoutSpec(
            "dd_foci_per_nucleus",
            "foci per nucleus",
            "foci",
            w,
            "up",
            "none",
            "mean foci per nucleus (pan-nuclear nuclei left out)",
            "foci",
        ),
        ReadoutSpec(
            "dd_foci_per_nucleus_median",
            "foci per nucleus (median)",
            "foci",
            w,
            "up",
            "none",
            "median foci per nucleus (pan-nuclear nuclei left out)",
            "foci",
        ),
        ReadoutSpec(
            "dd_positive_pct",
            "foci-positive nuclei",
            "%",
            w,
            "up",
            "none",
            "nuclei with at least positive_min_foci foci",
            "foci",
        ),
        ReadoutSpec(
            "dd_pan_nuclear_pct",
            "pan-nuclear nuclei",
            "%",
            w,
            "up",
            "none",
            "bright, diffuse nuclei (mean ≥ pan_factor × vehicle median)",
            "foci",
        ),
        ReadoutSpec(
            "dd_focus_area_um2",
            "focus area",
            "µm²",
            w,
            "either",
            "none",
            "mean area of the foci of nuclei with foci",
            "foci",
        ),
        ReadoutSpec(
            "dd_foci_intensity",
            "foci intensity per nucleus",
            "a.u.",
            w,
            "up",
            "none",
            "summed intensity of a nucleus's foci (nuclei with foci)",
            "foci",
        ),
        ReadoutSpec(
            "dd_foci_fold_vs_vehicle",
            "foci per nucleus vs vehicle",
            "×",
            w,
            "up",
            "none",
            "foci per nucleus / vehicle foci per nucleus",
            "foci",
        ),
        ReadoutSpec(
            "dd_nuclei",
            "foci: nuclei",
            "nuclei",
            w,
            "none",
            "none",
            "nuclei analysed by the foci step",
            "foci",
        ),
        ReadoutSpec(
            "dd_foci_per_nucleus_{phase:g1|s|g2m}",
            "foci per nucleus in {phase}",
            "foci",
            w,
            "up",
            "none",
            "foci per nucleus of one cell-cycle phase",
            "foci",
        ),
        ReadoutSpec(
            "dd_positive_{phase:g1|s|g2m}_pct",
            "foci-positive in {phase}",
            "%",
            w,
            "up",
            "none",
            "foci-positive nuclei of one cell-cycle phase",
            "foci",
        ),
        ReadoutSpec(
            "dd_foci", "foci", "foci", "object", "up", "none", "foci in the nucleus", "foci"
        ),
        ReadoutSpec(
            "qc_{labels}_outlier_count",
            "{labels}: QC outliers",
            "objects",
            w,
            "down",
            "none",
            "objects flagged by object QC (likely segmentation errors)",
            "object_qc",
        ),
        ReadoutSpec(
            "qc_{labels}_outlier_fraction",
            "{labels}: QC outlier fraction",
            "fraction",
            w,
            "down",
            "none",
            "fraction of objects flagged by object QC",
            "object_qc",
        ),
        ReadoutSpec(
            "gate_{name}_count",
            "gate {name}: count",
            "objects",
            w,
            "either",
            "none",
            "objects inside the gate",
            "gate_objects",
        ),
        ReadoutSpec(
            "gate_{name}_fraction",
            "gate {name}: % positive",
            "fraction",
            w,
            "either",
            "none",
            "fraction of objects inside the gate",
            "gate_objects",
        ),
        ReadoutSpec(
            "class_{name}_{cls}_count",
            "class {name} = {cls}: count",
            "objects",
            w,
            "either",
            "none",
            "objects of the class",
            "classify_objects",
        ),
        ReadoutSpec(
            "class_{name}_{cls}_fraction",
            "class {name} = {cls}: fraction",
            "fraction",
            w,
            "either",
            "none",
            "fraction of objects of the class",
            "classify_objects",
        ),
        ReadoutSpec(
            "cluster_{name}_{k}_count",
            "cluster {name} {k}: count",
            "objects",
            w,
            "either",
            "none",
            "objects in the cluster",
            "phenotypic_clusters",
        ),
        ReadoutSpec(
            "cluster_{name}_{k}_fraction",
            "cluster {name} {k}: fraction",
            "fraction",
            w,
            "either",
            "none",
            "fraction of objects in the cluster",
            "phenotypic_clusters",
        ),
        ReadoutSpec(
            "spatial_objects",
            "spatial: objects analysed",
            "objects",
            w,
            "none",
            "none",
            "objects in the spatial analysis",
            "spatial_clusters",
        ),
        ReadoutSpec(
            "spatial_density_per_mm2",
            "spatial: density",
            "objects/mm²",
            w,
            "either",
            "none",
            "objects per mm² of imaged area",
            "spatial_clusters",
        ),
        ReadoutSpec(
            "spatial_colonies",
            "spatial: colonies",
            "colonies",
            w,
            "either",
            "none",
            "number of colonies",
            "spatial_clusters",
        ),
        ReadoutSpec(
            "spatial_colony_size_{stat:mean|median}",
            "spatial: colony size ({stat})",
            "objects",
            w,
            "either",
            "none",
            "objects per colony",
            "spatial_clusters",
        ),
        ReadoutSpec(
            "spatial_colony_area_um2_mean",
            "spatial: colony area (mean)",
            "µm²",
            w,
            "either",
            "none",
            "convex hull area of the colonies",
            "spatial_clusters",
        ),
        ReadoutSpec(
            "spatial_fraction_in_colonies",
            "spatial: fraction in colonies",
            "fraction",
            w,
            "either",
            "none",
            "objects inside a colony",
            "spatial_clusters",
        ),
        ReadoutSpec(
            "spatial_nn_{stat:mean|median}_um",
            "spatial: nearest neighbour ({stat})",
            "µm",
            w,
            "either",
            "none",
            "nearest-neighbour distance",
            "spatial_clusters",
        ),
        ReadoutSpec(
            "spatial_clark_evans_R",
            "spatial: Clark–Evans R",
            "",
            w,
            "either",
            "none",
            "1 random, < 1 clustered, > 1 regular",
            "spatial_clusters",
        ),
        ReadoutSpec(
            "spatial_H_{r}um",
            "spatial: Ripley's H at {r} µm",
            "µm",
            w,
            "either",
            "none",
            "L(r) − r; 0 for random placement",
            "spatial_clusters",
        ),
        ReadoutSpec(
            "spatial_H_{r}um_csr_{bound:low|high}",
            "spatial: H envelope {bound} at {r} µm",
            "µm",
            w,
            "none",
            "none",
            "simulated random-placement envelope",
            "spatial_clusters",
        ),
        ReadoutSpec(
            "spatial_g_{r}um",
            "spatial: pair correlation at {r} µm",
            "",
            w,
            "either",
            "none",
            "1 = random",
            "spatial_clusters",
        ),
        ReadoutSpec(
            "spatial_pattern_{r}um",
            "spatial: pattern at {r} µm",
            "",
            w,
            "none",
            "none",
            "+1 clustered, −1 regular, 0 within the envelope",
            "spatial_clusters",
        ),
        ReadoutSpec(
            "spatial_groups",
            "spatial: patterns averaged",
            "",
            w,
            "none",
            "none",
            "fields or timepoints averaged",
            "spatial_clusters",
        ),
        # profiles (Phase 9)
        ReadoutSpec(
            "profile_distance_to_control",
            "profile distance to control",
            "SD",
            w,
            "up",
            "none",
            "distance of the well profile from the negative controls",
            "profiling",
        ),
        ReadoutSpec(
            "profile_distance_ratio",
            "profile distance ratio",
            "",
            w,
            "up",
            "none",
            "distance ÷ median control-well distance",
            "profiling",
        ),
        ReadoutSpec(
            "profile_{axis:PC|UMAP}{n:int}",
            "profile {axis}{n}",
            "",
            w,
            "none",
            "none",
            "well position in the embedding",
            "profiling",
        ),
    ]
    f: Level = "fit"
    pharm = "pharmacology"
    specs += [
        ReadoutSpec(
            "ec50",
            "EC50",
            "M",
            f,
            "down",
            "none",
            "relative EC50 (half-way between the fitted top and bottom)",
            pharm,
        ),
        ReadoutSpec(
            "log_ec50",
            "log10 EC50",
            "log10 M",
            f,
            "down",
            "none",
            "log10 of the relative EC50",
            pharm,
        ),
        ReadoutSpec(
            "abs_ic50",
            "absolute IC50",
            "M",
            f,
            "down",
            "none",
            "concentration giving 50 % on the normalized scale",
            pharm,
        ),
        ReadoutSpec("hill", "Hill slope", "", f, "none", "none", "steepness of the curve", pharm),
        ReadoutSpec("top", "top", "", f, "none", "none", "fitted upper plateau", pharm),
        ReadoutSpec("bottom", "bottom", "", f, "none", "none", "fitted lower plateau", pharm),
        ReadoutSpec(
            "emax_{kind:observed|fitted}",
            "Emax ({kind})",
            "%",
            f,
            "up",
            "none",
            "maximal effect",
            pharm,
        ),
        ReadoutSpec(
            "auc", "AUC", "", f, "either", "none", "normalized area under the curve", pharm
        ),
        ReadoutSpec(
            "activity_area",
            "activity area",
            "",
            f,
            "up",
            "none",
            "area between the curve and no effect",
            pharm,
        ),
        ReadoutSpec("r2", "R²", "", f, "up", "none", "goodness of fit", pharm),
        ReadoutSpec(
            "{gr:GR50|GRmax|GR_AOC|GEC50|h_GR|GRinf|GI50}",
            "{gr}",
            "",
            f,
            "either",
            "none",
            "growth-rate metric (Hafner et al. 2016)",
            pharm,
        ),
        ReadoutSpec(
            "{model:bliss|hsa|loewe|zip}_{summary:mean|max_3x3|max_cell}",
            "{model} synergy ({summary})",
            "",
            f,
            "up",
            "none",
            "combination score: > 0 synergy, < 0 antagonism",
            pharm,
        ),
        ReadoutSpec(
            "z_prime", "Z′", "", "fit", "up", "none", "assay window of the plate controls", pharm
        ),
        ReadoutSpec(
            "ssmd",
            "SSMD",
            "",
            "fit",
            "either",
            "none",
            "strictly standardized mean difference of the controls",
            pharm,
        ),
        ReadoutSpec(
            "signal_window", "signal window", "", "fit", "up", "none", "control separation", pharm
        ),
        ReadoutSpec(
            "cv_{group:vehicle|positive}_pct",
            "CV {group}",
            "%",
            "fit",
            "down",
            "none",
            "coefficient of variation of the control wells",
            pharm,
        ),
    ]
    return specs


READOUTS.register(*_builtin())
