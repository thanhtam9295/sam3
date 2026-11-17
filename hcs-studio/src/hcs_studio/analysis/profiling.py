"""Phenotypic profiling: feature selection, normalisation to negative controls, PCA, UMAP,
per-well profiles, distance to control, and dose-response curves.

Works on one or more finished runs (plates). Per plate, features are expressed as robust
Z-scores relative to the objects in the plate's negative-control wells:
(x − median) / (1.4826 · MAD) of those objects. Results are written to
``advanced/`` of the first run (or a chosen folder):

* ``pca_coordinates.parquet``, ``pca_loadings.csv``, ``pca_variance.csv``
* ``umap_coordinates.parquet`` (when UMAP is on)
* ``well_profiles.parquet``: per-well median profile (scaled by the control objects, so it
  is the median of the objects' robust Z-scores), well-level PCA/UMAP and the distance to
  the negative controls
* ``dose_response.csv`` and ``figures/`` (SVG and PNG)
* ``profiling.json``: options, features kept and dropped (with the reason), notes
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcs_studio import __version__
from hcs_studio.analysis.table_steps import feature_columns, read_well_objects, unique_rows

META = ("plate", "run", "well", "field", "timepoint", "object_id")
ANNOTATION = ("compound", "concentration", "treatment", "control_type")
POSITION = ("centroid_y_um", "centroid_x_um", "well_y_um", "well_x_um")
#: Normalised values are clipped to ±CLIP before PCA and UMAP (outliers would dominate).
CLIP = 10.0


@dataclass
class ProfileOptions:
    object_type: str = "nuclei"
    #: Embed individual objects ("cell") or per-well median profiles ("well").
    level: Literal["cell", "well"] = "cell"
    #: Features to use; None = every measured feature.
    features: list[str] | None = None
    #: Objects sampled evenly across wells for the cell-level embedding.
    max_cells: int = 50_000
    max_missing_fraction: float = 0.1
    #: A feature is near-constant when this fraction of objects share one value.
    constant_fraction: float = 0.95
    correlation_threshold: float = 0.95
    normalisation: Literal["negative_controls", "all_wells", "none"] = "negative_controls"
    #: Subtract each plate's median profile after normalisation (removes plate offsets).
    plate_correction: bool = False
    pca_components: int = 10
    umap: bool = True
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "euclidean"
    seed: int = 0
    #: Variance kept for the distance to control (PCA of the well profiles).
    distance_variance: float = 0.9
    #: Per-well readouts (summary columns or profile columns) to fit dose-response curves to.
    dose_readouts: list[str] = field(default_factory=lambda: ["profile_distance_to_control"])
    figures: bool = True


@dataclass
class ProfileResult:
    folder: Path
    cells: pd.DataFrame
    wells: pd.DataFrame
    loadings: pd.DataFrame
    variance: pd.DataFrame
    dose: pd.DataFrame
    features: list[str]
    dropped: dict[str, str]
    notes: list[str]


# ------------------------------------------------------------------ loading


@dataclass
class _Plate:
    run: Path
    name: str
    layout: dict[str, Any]
    cells: pd.DataFrame  # sampled objects
    medians: pd.DataFrame  # per-well median of every numeric feature (all objects)
    counts: dict[str, int]
    summary: pd.DataFrame  # per-well summary table of the run


def _load_plate(
    run: Path, options: ProfileOptions, per_well: int, rng: np.random.Generator
) -> _Plate:
    from hcs_studio.output.results import RunResults

    results = RunResults(run)
    cells, medians, counts = [], [], {}
    for well in results.wells:
        directory = run / "wells" / well
        if not (directory / "objects.parquet").exists():
            continue
        objects = read_well_objects(directory)
        objects = objects[(objects["object_type"] == options.object_type) & unique_rows(objects)]
        if objects.empty:
            continue
        counts[well] = len(objects)
        numeric = objects.select_dtypes("number")
        medians.append(numeric.median().rename(well))
        if len(objects) > per_well:
            objects = objects.iloc[np.sort(rng.choice(len(objects), per_well, replace=False))]
        cells.append(objects)
    if not cells:
        raise ValueError(f"{run.name}: no '{options.object_type}' objects to profile")
    sample = pd.concat(cells, ignore_index=True)
    name = results.plate_name
    sample.insert(0, "plate", name)
    sample.insert(1, "run", run.name)
    well_medians = pd.DataFrame(medians)
    well_medians.index.name = "well"
    return _Plate(run, name, results.layout, sample, well_medians, counts, results.summary)


def _annotate(frame: pd.DataFrame, plates: list[_Plate]) -> pd.DataFrame:
    """Add compound / concentration / treatment / control_type from each plate's layout."""
    columns: dict[str, list[Any]] = {c: [] for c in ANNOTATION}
    layouts = {p.run.name: p.layout for p in plates}
    for run, well in zip(frame["run"], frame["well"], strict=True):
        annotation = layouts.get(run, {}).get(well)
        for column in ANNOTATION:
            value = getattr(annotation, column, None) if annotation else None
            columns[column].append(value if value not in ("", None) else None)
    out = frame.copy()
    for column in ANNOTATION:
        out[column] = columns[column]
    out["concentration"] = pd.to_numeric(out["concentration"], errors="coerce")
    return out


# ------------------------------------------------------------ feature selection


def select_features(
    frame: pd.DataFrame, candidates: list[str], options: ProfileOptions
) -> tuple[list[str], dict[str, str]]:
    """Drop features with many missing values or nearly one value; returns kept, dropped."""
    kept, dropped = [], {}
    for name in candidates:
        values = pd.to_numeric(frame[name], errors="coerce")
        missing = float(values.isna().mean())
        if missing > options.max_missing_fraction:
            dropped[name] = f"missing in {missing:.0%} of objects"
            continue
        present = values.dropna()
        top = float(present.value_counts(normalize=True).iloc[0]) if len(present) else 1.0
        if top >= options.constant_fraction or present.std() == 0:
            dropped[name] = f"near-constant ({top:.0%} share one value)"
            continue
        kept.append(name)
    return kept, dropped


def drop_correlated(
    matrix: NDArray[np.float64], names: list[str], threshold: float
) -> tuple[list[str], dict[str, str]]:
    """Greedy filter: while two features correlate above ``threshold`` (|r|), drop the one
    more correlated with all others. Returns kept names and the reason for each dropped."""
    if len(names) < 2:
        return list(names), {}
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.abs(np.corrcoef(matrix, rowvar=False))
    corr = np.nan_to_num(corr)
    np.fill_diagonal(corr, 0)
    alive = np.ones(len(names), dtype=bool)
    dropped: dict[str, str] = {}
    while True:
        sub = corr[np.ix_(alive, alive)]
        if not sub.size or sub.max() <= threshold:
            break
        index = np.flatnonzero(alive)
        i, j = np.unravel_index(np.argmax(sub), sub.shape)
        a, b = index[i], index[j]
        loser, keeper = (a, b) if sub[i].mean() >= sub[j].mean() else (b, a)
        alive[loser] = False
        dropped[names[loser]] = f"|r| = {corr[loser, keeper]:.3f} with {names[keeper]}"
    return [n for n, keep in zip(names, alive, strict=True) if keep], dropped


# ---------------------------------------------------------------- normalisation


def _scale(values: NDArray[np.float64], fallback: NDArray[np.float64]) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        finite = fallback[np.isfinite(fallback)]
    if not len(finite):
        return 0.0, 1.0
    median = float(np.median(finite))
    scale = 1.4826 * float(np.median(np.abs(finite - median)))
    if not scale > 0:
        spread = fallback[np.isfinite(fallback)]
        scale = 1.4826 * float(np.median(np.abs(spread - np.median(spread)))) if len(spread) else 0
    if not scale > 0:
        scale = float(np.std(fallback[np.isfinite(fallback)])) if np.isfinite(fallback).any() else 0
    return median, scale if scale > 0 else 1.0


Stats = dict[str, dict[str, tuple[float, float]]]  # plate → feature → (median, scale)


def plate_stats(frame: pd.DataFrame, features: list[str], reference: pd.Series) -> Stats:
    """(median, 1.4826 · MAD) per plate and feature, of the rows where ``reference`` is True
    on that plate (all rows of the plate when it has none)."""
    stats: Stats = {}
    for plate, rows in frame.groupby("plate").groups.items():
        part = frame.loc[rows, features].apply(pd.to_numeric, errors="coerce")
        ref = reference.loc[rows].to_numpy(bool)
        stats[str(plate)] = {}
        for name in features:
            values = part[name].to_numpy(float)
            stats[str(plate)][name] = _scale(values[ref] if ref.any() else values, values)
    return stats


def apply_stats(frame: pd.DataFrame, features: list[str], stats: Stats) -> pd.DataFrame:
    out = frame.copy()
    # Integer features (counts) become fractional Z-scores.
    out[features] = out[features].apply(pd.to_numeric, errors="coerce").astype(float)
    for plate, rows in frame.groupby("plate").groups.items():
        for name in features:
            median, scale = stats[str(plate)][name]
            values = pd.to_numeric(frame.loc[rows, name], errors="coerce").to_numpy(float)
            out.loc[rows, name] = (values - median) / scale
    return out


def normalise(frame: pd.DataFrame, features: list[str], reference: pd.Series) -> pd.DataFrame:
    """Robust Z per plate: (x − median) / (1.4826 · MAD) of the rows where ``reference`` is
    True on that plate (all rows of the plate when it has none)."""
    return apply_stats(frame, features, plate_stats(frame, features, reference))


def _is_label_column(name: str) -> bool:
    """Table-step labels worth colouring by: classes, gates, clusters, colonies."""
    if name.startswith(("class_", "gate_", "cluster_")):
        return not name.endswith("_probability") and "_p_" not in name
    return name == "spatial_colony"


def _matrix(frame: pd.DataFrame, features: list[str]) -> NDArray[np.float64]:
    x = frame[features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return np.asarray(np.clip(np.nan_to_num(x, nan=0.0), -CLIP, CLIP), dtype=np.float64)


def _plate_center(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for _, rows in frame.groupby("plate").groups.items():
        out.loc[rows, features] = frame.loc[rows, features] - frame.loc[rows, features].median()
    return out


# ------------------------------------------------------------------ embedding


def pca(
    x: NDArray[np.float64], components: int, seed: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """(coordinates, explained variance ratio, loadings features × components)."""
    from sklearn.decomposition import PCA

    k = max(1, min(components, x.shape[0], x.shape[1]))
    model = PCA(n_components=k, random_state=seed)
    coords = model.fit_transform(x)
    return coords, model.explained_variance_ratio_, model.components_.T


def umap_embedding(
    x: NDArray[np.float64], n_neighbors: int, min_dist: float, metric: str, seed: int
) -> NDArray[np.float64] | None:
    """2D UMAP, seeded (reproducible; single-threaded). None with fewer than 5 points."""
    if len(x) < 5:
        return None
    import warnings

    import umap

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "n_jobs overridden by random_state"
        model = umap.UMAP(
            n_neighbors=max(2, min(n_neighbors, len(x) - 1)),
            min_dist=min_dist,
            metric=metric,
            random_state=seed,
            n_components=2,
        )
        return np.asarray(model.fit_transform(x), dtype=np.float64)


def distance_to_control(
    profiles: NDArray[np.float64], controls: NDArray[np.bool_], variance: float, seed: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Euclidean distance of each well profile from the median negative-control profile, in
    the principal components that explain ``variance`` of the well profiles, and that
    distance divided by the median distance of the control wells themselves (≈ 1 for a
    control-like well). Returns (distance, ratio, components used)."""
    n = len(profiles)
    nan = np.full(n, np.nan)
    if n < 2 or not controls.any():
        return nan, nan, 0
    coords, ratio, _ = pca(profiles, min(n, profiles.shape[1]), seed)
    k = int(np.searchsorted(np.cumsum(ratio), variance) + 1)
    k = min(k, coords.shape[1])
    space = coords[:, :k]
    centre = np.median(space[controls], axis=0)
    distance = np.linalg.norm(space - centre, axis=1)
    spread = float(np.median(distance[controls])) if controls.sum() >= 2 else 0.0
    return distance, distance / spread if spread > 0 else nan, k


# ---------------------------------------------------------------- dose-response


def four_pl(x: NDArray[Any], bottom: float, top: float, log_ec50: float, hill: float) -> Any:
    """4-parameter logistic in log10 concentration."""
    return bottom + (top - bottom) / (1 + 10 ** ((log_ec50 - np.log10(x)) * hill))


def fit_dose_response(concentration: NDArray[Any], response: NDArray[Any]) -> dict[str, Any]:
    """Fit a 4-parameter logistic; needs 4 distinct positive concentrations. ``ec50`` is in
    the concentration's units; a decreasing curve has ``top < bottom``."""
    from scipy.optimize import curve_fit

    x = np.asarray(concentration, dtype=float)
    y = np.asarray(response, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[keep], y[keep]
    levels = np.unique(x)
    out: dict[str, Any] = {"n": len(x), "concentrations": len(levels)}
    if len(levels) < 4:
        return {**out, "fit": False, "note": "needs 4 or more concentrations"}
    means = np.array([y[x == c].mean() for c in levels])
    low, high = means[0], means[-1]
    logs = np.log10(levels)
    guess = [low, high, float(np.median(logs)), 1.0]
    bounds = ([-np.inf, -np.inf, logs.min() - 2, 0.1], [np.inf, np.inf, logs.max() + 2, 10.0])
    try:
        params, _ = curve_fit(four_pl, x, y, p0=guess, bounds=bounds, maxfev=20000)
    except (RuntimeError, ValueError) as exc:
        return {**out, "fit": False, "note": f"did not converge: {exc}"}
    predicted = four_pl(x, *params)
    total = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(((y - predicted) ** 2).sum()) / total if total > 0 else math.nan
    bottom, top, log_ec50, hill = (float(v) for v in params)
    at_edge = not logs.min() - 1e-9 <= log_ec50 <= logs.max() + 1e-9
    return {
        **out,
        "fit": True,
        "bottom": bottom,
        "top": top,
        "ec50": 10**log_ec50,
        "hill": hill,
        "r2": r2,
        "note": "EC50 outside the tested range (extrapolated)" if at_edge else "",
    }


def dose_response_table(wells: pd.DataFrame, readouts: list[str]) -> pd.DataFrame:
    """One fit per (plate, compound, readout) over the non-control wells with a concentration."""
    rows = []
    treated = wells[
        wells["compound"].notna()
        & wells["concentration"].notna()
        & (wells["concentration"] > 0)
        & wells["control_type"].isna()
    ]
    for (plate, compound), group in treated.groupby(["plate", "compound"]):
        for readout in readouts:
            if readout not in group:
                continue
            fit = fit_dose_response(group["concentration"].to_numpy(), group[readout].to_numpy())
            rows.append({"plate": plate, "compound": compound, "readout": readout, **fit})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- figures


def _figures(result: ProfileResult, options: ProfileOptions) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = result.folder / "figures"
    target.mkdir(parents=True, exist_ok=True)
    written = []

    def save(fig: Any, name: str) -> None:
        for suffix in ("svg", "png"):
            fig.savefig(target / f"{name}.{suffix}", dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(name)

    variance = result.variance
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(variance["component"], variance["explained_variance_ratio"] * 100, color="#4477aa")
    ax.plot(variance["component"], variance["cumulative"] * 100, color="#222", marker="o", ms=3)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title("PCA scree")
    save(fig, "pca_scree")

    def scatter(frame: pd.DataFrame, x: str, y: str, name: str, title: str) -> None:
        if x not in frame or y not in frame:
            return
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        groups = _colour_groups(frame)
        for label, rows, colour in groups:
            ax.scatter(
                frame.loc[rows, x],
                frame.loc[rows, y],
                s=6 if len(frame) > 500 else 18,
                alpha=0.6,
                color=colour,
                label=label,
                linewidths=0,
            )
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title)
        if 1 < len(groups) <= 12:
            ax.legend(fontsize=7, markerscale=2, frameon=False)
        save(fig, name)

    level = "objects" if options.level == "cell" else "wells"
    scatter(result.cells, "PC1", "PC2", "pca_cells", f"PCA of {options.object_type}")
    scatter(result.cells, "UMAP1", "UMAP2", "umap_cells", f"UMAP of {options.object_type}")
    scatter(result.wells, "PC1", "PC2", "pca_wells", f"PCA of well profiles ({level})")
    scatter(result.wells, "UMAP1", "UMAP2", "umap_wells", "UMAP of well profiles")

    for record in result.dose.to_dict("records") if not result.dose.empty else []:
        group = result.wells[
            (result.wells["plate"] == record["plate"])
            & (result.wells["compound"] == record["compound"])
            & (result.wells["concentration"] > 0)
        ]
        readout = record["readout"]
        if readout not in group or group.empty:
            continue
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        stats = group.groupby("concentration")[readout].agg(["mean", "std"])
        ax.errorbar(
            stats.index,
            stats["mean"],
            yerr=stats["std"].fillna(0),
            fmt="o",
            color="#4477aa",
            capsize=3,
        )
        if record.get("fit"):
            grid = np.logspace(np.log10(stats.index.min()), np.log10(stats.index.max()), 100)
            ax.plot(
                grid,
                four_pl(
                    grid,
                    record["bottom"],
                    record["top"],
                    math.log10(record["ec50"]),
                    record["hill"],
                ),
                color="#ee6677",
            )
            ax.set_title(f"{record['compound']}: EC50 {record['ec50']:.3g}", fontsize=10)
        else:
            ax.set_title(f"{record['compound']} (no fit: {record.get('note', '')})", fontsize=9)
        controls = result.wells[
            (result.wells["plate"] == record["plate"])
            & (result.wells["control_type"] == "negative")
        ]
        if readout in controls and controls[readout].notna().any():
            ax.axhline(
                float(controls[readout].median()),
                color="#999",
                ls="--",
                lw=1,
                label="negative control",
            )
            ax.legend(fontsize=7, frameon=False)
        ax.set_xscale("log")
        ax.set_xlabel("Concentration")
        ax.set_ylabel(readout)
        safe = "".join(c if c.isalnum() else "_" for c in f"{record['compound']}_{readout}")
        save(fig, f"dose_{safe}")
    return written


def save_scatter(
    path: Path,
    x: NDArray[Any],
    y: NDArray[Any],
    values: pd.Series,
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    highlight: NDArray[np.bool_] | None = None,
) -> None:
    """Scatter plot coloured by ``values`` (categories, or numbers on viridis), saved as SVG
    or PNG by the file suffix."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    size = 6 if len(x) > 500 else 22
    if pd.api.types.is_float_dtype(values) and values.notna().any():
        art = ax.scatter(x, y, c=values.to_numpy(float), cmap="viridis", s=size, linewidths=0)
        fig.colorbar(art, ax=ax, label=str(values.name or ""))
    else:
        labels = values.astype(str).where(values.notna(), "–")
        names = sorted(labels.unique(), key=lambda s: (not s.endswith("control"), s))
        for i, name in enumerate(names):
            rows = (labels == name).to_numpy()
            ax.scatter(
                x[rows],
                y[rows],
                s=size,
                color=PALETTE[i % len(PALETTE)],
                label=name,
                linewidths=0,
                alpha=0.8,
            )
        if len(names) <= 14:
            ax.legend(fontsize=7, markerscale=2, frameon=False)
    if highlight is not None and highlight.any():
        ax.scatter(
            x[highlight],
            y[highlight],
            s=size * 3,
            facecolors="none",
            edgecolors="black",
            linewidths=1,
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


PALETTE = (
    "#4477aa",
    "#ee6677",
    "#228833",
    "#ccbb44",
    "#66ccee",
    "#aa3377",
    "#bbbbbb",
    "#ee8866",
    "#44bb99",
    "#99ddff",
    "#332288",
    "#882255",
)


def colour_key(frame: pd.DataFrame) -> pd.Series:
    """Default grouping for colours: control type, else compound, else 'sample'."""
    control = frame["control_type"].fillna("") if "control_type" in frame else ""
    compound = frame["compound"].fillna("") if "compound" in frame else ""
    key = pd.Series("sample", index=frame.index)
    if isinstance(compound, pd.Series):
        key = key.where(compound == "", compound)
    if isinstance(control, pd.Series):
        key = key.where(control == "", control + " control")
    return key


def _colour_groups(frame: pd.DataFrame) -> list[tuple[str, pd.Index, str]]:
    key = colour_key(frame)
    labels = sorted(key.unique(), key=lambda s: (not s.endswith("control"), s))
    return [
        (label, key.index[key == label], PALETTE[i % len(PALETTE)])
        for i, label in enumerate(labels)
    ]


# --------------------------------------------------------------------- main entry


def run_profiling(
    runs: list[Path], options: ProfileOptions | None = None, output: Path | None = None
) -> ProfileResult:
    """Profile one or more runs; writes ``advanced/`` (see the module docstring)."""
    options = options or ProfileOptions()
    runs = [Path(r) for r in runs]
    if not runs:
        raise ValueError("Give at least one run folder")
    started = time.monotonic()
    folder = Path(output) if output is not None else runs[0] / "advanced"
    rng = np.random.default_rng(options.seed)
    from hcs_studio.output.results import RunResults

    total_wells = sum(len(RunResults(r).wells) for r in runs)
    per_well = max(1, options.max_cells // max(1, total_wells))
    plates = [_load_plate(run, options, per_well, rng) for run in runs]
    notes: list[str] = []
    names = [p.name for p in plates]
    if len(set(names)) < len(names):  # same plate name in several runs: keep them apart
        for plate in plates:
            plate.name = f"{plate.name} ({plate.run.name})"
            plate.cells["plate"] = plate.name
    cells = _annotate(pd.concat([p.cells for p in plates], ignore_index=True), plates)

    # Features measured on every plate.
    candidates = feature_columns(cells, options.features)
    for plate in plates:
        candidates = [c for c in candidates if c in plate.cells.columns]
    features, dropped = select_features(cells, candidates, options)
    if not features:
        raise ValueError("No usable features after removing missing and near-constant ones")

    # Cell level: normalise sampled objects to the control objects of their plate.
    negative = cells["control_type"] == "negative"
    if options.normalisation == "negative_controls":
        for plate in plates:
            if not negative[cells["plate"] == plate.name].any():
                notes.append(
                    f"{plate.name}: no negative-control wells in the layout; normalised to "
                    "all wells instead"
                )
    reference = (
        negative
        if options.normalisation == "negative_controls"
        else pd.Series(True, index=cells.index)
    )
    stats = plate_stats(cells, features, reference)
    cells_z = (
        cells.copy() if options.normalisation == "none" else apply_stats(cells, features, stats)
    )

    # Well level: median of all objects of each well, normalised to the control wells.
    well_rows = []
    for plate in plates:
        medians = plate.medians.reindex(columns=features)
        medians = medians.reset_index()
        medians.insert(0, "plate", plate.name)
        medians.insert(1, "run", plate.run.name)
        medians["n_objects"] = medians["well"].map(plate.counts)
        well_rows.append(medians)
    wells = _annotate(pd.concat(well_rows, ignore_index=True), plates)
    well_negative = wells["control_type"] == "negative"
    # Well profiles use the control *objects'* median and MAD (stable even with few control
    # wells), so a well profile is the median of its objects' robust Z-scores, and distances
    # are in units of object-to-object variation.
    wells_z = (
        wells.copy() if options.normalisation == "none" else apply_stats(wells, features, stats)
    )
    if options.plate_correction:
        cells_z = _plate_center(cells_z, features)
        wells_z = _plate_center(wells_z, features)

    # Correlation filter on the data that will be embedded.
    level_frame = cells_z if options.level == "cell" else wells_z
    features, correlated = drop_correlated(
        _matrix(level_frame, features), features, options.correlation_threshold
    )
    dropped.update(correlated)

    # Embeddings.
    embed = cells_z if options.level == "cell" else wells_z
    x = _matrix(embed, features)
    coords, ratio, loadings = pca(x, options.pca_components, options.seed)
    pcs = [f"PC{i + 1}" for i in range(coords.shape[1])]
    keep = [c for c in (*META, *ANNOTATION, "n_objects", *POSITION) if c in embed]
    keep += [c for c in embed.columns if _is_label_column(str(c))]
    embedded = embed[keep].copy()
    embedded[pcs] = coords
    umap_frame = None
    if options.umap:
        u = umap_embedding(x, options.n_neighbors, options.min_dist, options.metric, options.seed)
        if u is not None:
            embedded["UMAP1"], embedded["UMAP2"] = u[:, 0], u[:, 1]
            umap_frame = embedded[[c for c in embedded.columns if not str(c).startswith("PC")]]
        else:
            notes.append("too few points for UMAP")

    # Per-well profiles, their own embedding and the distance to control.
    wx = _matrix(wells_z, features)
    profiles = wells_z[[c for c in ("plate", "run", "well", *ANNOTATION, "n_objects")]].copy()
    if len(wx) >= 2:
        wcoords, _, _ = pca(wx, 2, options.seed)
        profiles["PC1"] = wcoords[:, 0]
        if wcoords.shape[1] > 1:
            profiles["PC2"] = wcoords[:, 1]
    if options.umap:
        wu = umap_embedding(wx, options.n_neighbors, options.min_dist, options.metric, options.seed)
        if wu is not None:
            profiles["UMAP1"], profiles["UMAP2"] = wu[:, 0], wu[:, 1]
    controls = well_negative.to_numpy(bool)
    if not controls.any():
        notes.append("no negative-control wells: the distance to control is not computed")
    distance, ratio_to_controls, k = distance_to_control(
        wx, controls, options.distance_variance, options.seed
    )
    profiles["profile_distance_to_control"] = distance
    profiles["profile_distance_ratio"] = ratio_to_controls
    # Per-well summary readouts (cluster fractions, counts, ...) for dose-response curves.
    for plate in plates:
        if plate.summary.empty:
            continue
        in_run = profiles["run"] == plate.run.name
        for column in options.dose_readouts:
            if column in plate.summary and column not in ("well",):
                profiles.loc[in_run, column] = (
                    profiles.loc[in_run, "well"].map(plate.summary[column]).to_numpy()
                )
    for name in features:
        profiles[f"z_{name}"] = wells_z[name].to_numpy()
    dose = dose_response_table(profiles, options.dose_readouts)

    variance = pd.DataFrame(
        {
            "component": pcs,
            "explained_variance_ratio": ratio,
            "cumulative": np.cumsum(ratio),
        }
    )
    loading_table = pd.DataFrame(loadings, index=features, columns=pcs)
    loading_table.index.name = "feature"
    result = ProfileResult(
        folder=folder,
        cells=embedded,
        wells=profiles,
        loadings=loading_table,
        variance=variance,
        dose=dose,
        features=features,
        dropped=dropped,
        notes=notes,
    )

    folder.mkdir(parents=True, exist_ok=True)
    pca_frame = embedded[[c for c in embedded.columns if not str(c).startswith("UMAP")]]
    pca_frame.to_parquet(folder / "pca_coordinates.parquet", index=False)
    if umap_frame is not None:
        umap_frame.to_parquet(folder / "umap_coordinates.parquet", index=False)
    else:
        (folder / "umap_coordinates.parquet").unlink(missing_ok=True)
    loading_table.to_csv(folder / "pca_loadings.csv")
    variance.to_csv(folder / "pca_variance.csv", index=False)
    profiles.to_parquet(folder / "well_profiles.parquet", index=False)
    dose.to_csv(folder / "dose_response.csv", index=False)
    figures = _figures(result, options) if options.figures else []
    record = {
        "software": {"name": "hcs-studio", "version": __version__},
        "created": datetime.now(UTC).isoformat(),
        "runs": [str(r) for r in runs],
        "options": asdict(options),
        "objects_embedded": len(embedded),
        "wells": len(profiles),
        "features_used": features,
        "features_dropped": dropped,
        "distance_components": k,
        "explained_variance": [float(v) for v in ratio],
        "notes": notes,
        "figures": figures,
        "seconds": round(time.monotonic() - started, 1),
    }
    (folder / "profiling.json").write_text(
        json.dumps(record, indent=2, default=str), encoding="utf-8"
    )
    return result


def load_profile(folder: Path) -> ProfileResult | None:
    """Read ``advanced/`` back (None if the run has not been profiled)."""
    folder = Path(folder)
    if not (folder / "profiling.json").exists():
        return None
    record = json.loads((folder / "profiling.json").read_text(encoding="utf-8"))
    cells = pd.read_parquet(folder / "pca_coordinates.parquet")
    if (folder / "umap_coordinates.parquet").exists():
        umap_frame = pd.read_parquet(folder / "umap_coordinates.parquet")
        if len(umap_frame) == len(cells):
            cells["UMAP1"], cells["UMAP2"] = umap_frame["UMAP1"], umap_frame["UMAP2"]
    try:
        dose = pd.read_csv(folder / "dose_response.csv")
    except (pd.errors.EmptyDataError, FileNotFoundError):
        dose = pd.DataFrame()
    return ProfileResult(
        folder=folder,
        cells=cells,
        wells=pd.read_parquet(folder / "well_profiles.parquet"),
        loadings=pd.read_csv(folder / "pca_loadings.csv", index_col="feature"),
        variance=pd.read_csv(folder / "pca_variance.csv"),
        dose=dose,
        features=list(record.get("features_used", [])),
        dropped=dict(record.get("features_dropped", {})),
        notes=list(record.get("notes", [])),
    )
