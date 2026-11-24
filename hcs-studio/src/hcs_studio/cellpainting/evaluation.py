"""Evaluating Cell Painting profiles (Phase 24c).

Works on the output of the profiling pipeline (``cellpainting/profiles.py``): the well profiles
after feature selection and the consensus profile of each treatment.

- **Activity** (copairs, Kalinin et al. 2025): each treatment's replicate wells should retrieve
  each other before the vehicle wells. Average precision per well (cosine), with positives =
  the other wells of the treatment and negatives = vehicle wells (of the same plate by default);
  mAP per treatment, p from copairs' permutation null, Benjamini–Hochberg FDR. *Active* = FDR
  below ``threshold``. As a negative control, a few vehicle wells of each plate (as many as a
  treatment's replicates, at most a third of the plate's vehicle wells, at least 2) are held
  out and scored as if they were a treatment (``vehicle (held out)``): they should not be
  active. Power depends on the number of vehicle wells: with 3 replicates and 4 reference
  vehicle wells, even a perfect mAP of 1 has p ≈ 0.05.
- **Consistency**: consensus profiles of active treatments with the same MoA (from an annotation
  file) should retrieve each other before those of other MoAs; mAP per MoA, p, FDR.
- **Similarity**: cosine and Pearson matrices between consensus profiles; a clustered heat map
  (average linkage on 1 − cosine); the nearest treatments of each (``similar``).
- **MoA suggestions** (kNN): for each treatment, the MoAs of its ``k`` most similar annotated
  treatments of other compounds, scored by summed cosine; for annotated ones the leave-one-
  compound-out top-1 accuracy shows how far to trust the suggestions.
- **Batch effects**: do vehicle wells cluster by plate (and by ``Metadata_Date`` or
  ``Metadata_Batch`` when present)? mAP of vehicle wells grouped by plate and the silhouette
  score (cosine), **before** normalization (aggregated profiles, each feature standardised over
  all wells) and **after** (the profiles used for everything above).
- **UMAP** of the well profiles (seeded), coloured by treatment and by plate.

``run_evaluation`` writes all of it into the profiling folder: ``activity.csv``,
``consistency.csv``, ``similarity_cosine.csv``, ``similarity_pearson.csv``, ``similar.csv``,
``moa_suggestions.csv``, ``batch.json``, ``umap.csv``, ``figures/`` and ``evaluation.html``.
"""

from __future__ import annotations

import html
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from hcs_studio.cellpainting.profiles import STRATA

HOLDOUT = "vehicle (held out)"
REFERENCE = "Metadata_reference_index"
BATCH_COLUMNS = ("Metadata_Plate", "Metadata_Date", "Metadata_Batch")


class EvaluationOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    treatment: str = Field(default="Metadata_Treatment", description="Column naming treatments.")
    threshold: float = Field(default=0.05, gt=0, lt=1, description="FDR for active / consistent.")
    null_size: int = Field(default=10000, ge=100)
    same_plate_negatives: bool = Field(
        default=True, description="Compare each well with the vehicle wells of its own plate."
    )
    holdout: int = Field(
        default=0, ge=0, description="Vehicle wells held out per plate (0 = automatic)."
    )
    k: int = Field(default=5, ge=1, description="Neighbours for MoA suggestions.")
    neighbours: int = Field(default=10, ge=1, description="Similar treatments listed per one.")
    umap: bool = True
    seed: int = 0


@dataclass
class Evaluation:
    activity: pd.DataFrame
    cosine: pd.DataFrame
    pearson: pd.DataFrame
    similar: pd.DataFrame
    batch: dict[str, Any]
    consistency: pd.DataFrame | None = None
    moa_suggestions: pd.DataFrame | None = None
    moa_accuracy: float | None = None
    umap: pd.DataFrame | None = None
    log: dict[str, Any] = field(default_factory=dict)


def _features(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if not str(c).startswith("Metadata_")]


def _complete(frame: pd.DataFrame, features: list[str]) -> list[str]:
    """Features without missing values (copairs refuses NaN)."""
    return [f for f in features if frame[f].notna().all()]


def _is_vehicle(wells: pd.DataFrame) -> pd.Series:
    if "Metadata_Role" not in wells:
        return pd.Series(False, index=wells.index)
    return wells["Metadata_Role"].eq("negative_control")


def _quiet() -> None:
    logging.getLogger("copairs").setLevel(logging.ERROR)


def _map(
    meta: pd.DataFrame,
    values: NDArray[np.float64],
    *,
    pos_sameby: list[str],
    pos_diffby: list[str],
    neg_sameby: list[str],
    neg_diffby: list[str],
    group: list[str],
    options: EvaluationOptions,
) -> pd.DataFrame:
    from copairs.map import average_precision, mean_average_precision

    _quiet()
    ap = average_precision(
        meta.reset_index(drop=True),
        values,
        pos_sameby=pos_sameby,
        pos_diffby=pos_diffby,
        neg_sameby=neg_sameby,
        neg_diffby=neg_diffby,
        progress_bar=False,
    )
    result = mean_average_precision(
        ap,
        group,
        null_size=options.null_size,
        threshold=options.threshold,
        seed=options.seed,
        progress_bar=False,
    )
    return pd.DataFrame(result)


def _holdout_size(wells: pd.DataFrame, vehicle: pd.Series, options: EvaluationOptions) -> int:
    if options.holdout:
        return options.holdout
    treated = wells[~vehicle]
    replicates = (
        int(treated.groupby(["Metadata_Plate", options.treatment]).size().median())
        if len(treated)
        else 2
    )
    return max(2, min(replicates, 4))


def activity(wells: pd.DataFrame, options: EvaluationOptions) -> pd.DataFrame:
    """mAP of each treatment against the vehicle wells (one row per treatment)."""
    wells = wells.reset_index(drop=True).copy()
    vehicle = _is_vehicle(wells)
    if not vehicle.any():
        raise ValueError("Activity needs vehicle (negative-control) wells")
    features = _complete(wells, _features(wells))
    key = options.treatment
    wells[key] = wells[key].fillna("").astype(str)
    # negative control: a few vehicle wells of each plate scored as if they were a treatment
    size = _holdout_size(wells, vehicle, options)
    rng = np.random.default_rng(options.seed)
    held: list[int] = []
    for _, group in wells[vehicle].groupby("Metadata_Plate", sort=True):
        n = min(size, len(group) // 3)  # at most a third: the rest stay the reference
        if n >= 2:
            held += [int(i) for i in rng.choice(group.index.to_numpy(), n, replace=False)]
    wells.loc[held, key] = HOLDOUT
    reference = vehicle & ~wells.index.isin(held)
    wells[REFERENCE] = np.where(reference, wells.index, -1)
    meta = wells[[c for c in wells.columns if c.startswith("Metadata_")]]
    neg_sameby = ["Metadata_Plate"] if options.same_plate_negatives else []
    scores = _map(
        meta,
        wells[features].to_numpy(float),
        pos_sameby=[key, REFERENCE],
        pos_diffby=[],
        neg_sameby=neg_sameby,
        neg_diffby=[key, REFERENCE],
        group=[key],
        options=options,
    )
    counts = wells[~reference].groupby(key).size().rename("n_wells")
    table = scores.merge(counts, left_on=key, right_index=True, how="left")
    if "Metadata_Compound" in wells:
        compounds = wells[~reference].groupby(key)["Metadata_Compound"].first()
        table["Metadata_Compound"] = table[key].map(compounds)
    table["active"] = table["corrected_p_value"] < options.threshold
    table["holdout"] = table[key] == HOLDOUT
    table = table.rename(columns={"mean_average_precision": "mAP"})
    keep = [
        key,
        "Metadata_Compound",
        "n_wells",
        "mAP",
        "mean_normalized_average_precision",
        "p_value",
        "corrected_p_value",
        "active",
        "holdout",
    ]
    return (
        table[[c for c in keep if c in table]]
        .sort_values(["holdout", "mAP"], ascending=[True, False])
        .reset_index(drop=True)
    )


def _label(consensus: pd.DataFrame, by: list[str]) -> pd.Series:
    """Row names of consensus profiles: the treatment, plus columns that vary (cell line)."""
    shown = [by[0], *[c for c in by[1:] if consensus[c].nunique() > 1]]
    return consensus[shown].astype(str).agg(" | ".join, axis=1)


def similarity(
    consensus: pd.DataFrame, features: list[str], labels: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = consensus[features].to_numpy(float)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    unit = x / np.where(norms == 0, 1, norms)
    cosine = unit @ unit.T
    centred = x - x.mean(axis=1, keepdims=True)
    spread = np.linalg.norm(centred, axis=1, keepdims=True)
    centred = centred / np.where(spread == 0, 1, spread)
    pearson = centred @ centred.T
    names = labels.tolist()
    return (
        pd.DataFrame(cosine, index=names, columns=names),
        pd.DataFrame(pearson, index=names, columns=names),
    )


def similar(cosine: pd.DataFrame, n: int, to: str | None = None) -> pd.DataFrame:
    """The ``n`` most similar treatments of each one (or of ``to``), by cosine."""
    rows = []
    for name in [to] if to is not None else list(cosine.index):
        if name not in cosine.index:
            raise ValueError(f"No profile named {name!r}")
        others = pd.Series(cosine.loc[name]).drop(name).sort_values(ascending=False).head(n)
        for rank, (other, value) in enumerate(others.items(), start=1):
            rows.append({"treatment": name, "rank": rank, "similar": other, "cosine": value})
    return pd.DataFrame(rows, columns=["treatment", "rank", "similar", "cosine"])


def read_moa(path: Path) -> pd.DataFrame:
    """A MoA annotation file (CSV/TSV/XLSX): a compound column (compound, Metadata_Compound,
    broad_sample, pert_iname …) and a MoA column (moa, Metadata_moa, mechanism)."""
    path = Path(path)
    table = (
        pd.read_excel(path)
        if path.suffix.lower() in (".xlsx", ".xls")
        else pd.read_csv(path, sep=None, engine="python")
    )
    lower = {str(c).lower().removeprefix("metadata_"): c for c in table.columns}
    compound = next(
        (
            lower[c]
            for c in ("compound", "broad_sample", "pert_iname", "name", "treatment")
            if c in lower
        ),
        None,
    )
    moa = next((lower[c] for c in ("moa", "mechanism", "target") if c in lower), None)
    if compound is None or moa is None:
        raise ValueError(f"{path.name}: needs a compound column and a moa column")
    out = table[[compound, moa]].dropna()
    out.columns = pd.Index(["compound", "moa"])
    return out.astype(str).drop_duplicates("compound")


def consistency(
    consensus: pd.DataFrame, features: list[str], options: EvaluationOptions
) -> pd.DataFrame | None:
    """mAP of consensus profiles grouped by MoA (rows with a ``moa``; MoAs with ≥ 2)."""
    rows = consensus[consensus["moa"].notna()].reset_index(drop=True)
    shared = rows.groupby("moa")["compound"].transform("nunique") >= 2
    rows = rows[shared].reset_index(drop=True)
    if rows["moa"].nunique() < 2:
        return None
    table = _map(
        rows[["moa", "compound"]],
        rows[features].to_numpy(float),
        pos_sameby=["moa"],
        pos_diffby=["compound"],
        neg_sameby=[],
        neg_diffby=["moa"],
        group=["moa"],
        options=options,
    )
    table = table.rename(columns={"mean_average_precision": "mAP"})
    table["n_compounds"] = table["moa"].map(rows.groupby("moa")["compound"].nunique())
    table["consistent"] = table["corrected_p_value"] < options.threshold
    keep = ["moa", "n_compounds", "mAP", "p_value", "corrected_p_value", "consistent"]
    return table[keep].sort_values("mAP", ascending=False).reset_index(drop=True)


def moa_suggestions(
    cosine: pd.DataFrame, info: pd.DataFrame, k: int
) -> tuple[pd.DataFrame, float | None]:
    """kNN MoA votes from annotated treatments of other compounds; top-1 accuracy (leave one
    compound out) on the annotated ones. ``info``: index = treatment label, columns
    ``compound``, ``moa``, ``vehicle``."""
    annotated = info[info["moa"].notna() & ~info["vehicle"]]
    rows = []
    hits = []
    for name, row in info[~info["vehicle"]].iterrows():
        pool = annotated[annotated["compound"] != row["compound"]]
        if pool.empty:
            continue
        nearest = pd.Series(cosine.loc[str(name)])[pool.index].sort_values(ascending=False).head(k)
        votes = (
            pd.DataFrame({"moa": pool.loc[nearest.index, "moa"], "cosine": nearest})
            .groupby("moa")["cosine"]
            .sum()
            .sort_values(ascending=False)
        )
        suggestion = str(votes.index[0])
        known = row["moa"] if isinstance(row["moa"], str) else None
        if known is not None:
            hits.append(suggestion == known)
        rows.append(
            {
                "treatment": name,
                "compound": row["compound"],
                "known_moa": known,
                "suggested_moa": suggestion,
                "score": float(votes.iloc[0]),
                "runner_up": str(votes.index[1]) if len(votes) > 1 else None,
                "neighbours": "; ".join(f"{n} ({v:.2f})" for n, v in nearest.items()),
            }
        )
    accuracy = float(np.mean(hits)) if hits else None
    return pd.DataFrame(rows), accuracy


def _standardise(frame: pd.DataFrame, features: list[str]) -> NDArray[np.float64]:
    x = frame[features].to_numpy(float)
    sd = np.nanstd(x, axis=0)
    keep = sd > 0
    return np.asarray((x[:, keep] - np.nanmean(x[:, keep], axis=0)) / sd[keep], dtype=np.float64)


def _cluster_by(
    meta: pd.DataFrame, x: NDArray[np.float64], column: str, options: EvaluationOptions
) -> dict[str, Any]:
    from sklearn.metrics import silhouette_score

    labels = meta[column].astype(str).to_numpy()
    frame = pd.DataFrame({column: labels, REFERENCE: np.arange(len(labels))})
    scores = _map(
        frame,
        x,
        pos_sameby=[column],
        pos_diffby=[],
        neg_sameby=[],
        neg_diffby=[column],
        group=[column],
        options=options,
    )
    return {
        "mAP": float(scores["mean_average_precision"].mean()),
        "groups_clustered": int((scores["corrected_p_value"] < options.threshold).sum()),
        "groups": len(scores),
        "silhouette": float(silhouette_score(x, labels, metric="cosine")),
    }


def batch_effects(
    aggregated: pd.DataFrame, wells: pd.DataFrame, options: EvaluationOptions
) -> dict[str, Any]:
    """Do vehicle wells cluster by plate / day / batch, before and after normalization?"""
    out: dict[str, Any] = {}
    stages = {"before": aggregated, "after": wells}
    for column in BATCH_COLUMNS:
        result: dict[str, Any] = {}
        for stage, frame in stages.items():
            mask = _is_vehicle(frame)
            vehicle = frame[mask].reset_index(drop=True)
            if column not in vehicle or vehicle[column].nunique() < 2:
                break
            if vehicle.groupby(column).size().min() < 2:
                result[stage] = {"note": "fewer than 2 vehicle wells in a group"}
                continue
            if stage == "before":  # each feature standardised over all wells, then the vehicles
                x = _standardise(frame, _complete(frame, _features(frame)))[mask.to_numpy()]
            else:
                x = vehicle[_complete(vehicle, _features(vehicle))].to_numpy(float)
            result[stage] = _cluster_by(vehicle, x, column, options)
        if result:
            out[column.removeprefix("Metadata_").lower()] = result
    return out


def _umap(
    wells: pd.DataFrame, features: list[str], options: EvaluationOptions
) -> pd.DataFrame | None:
    from hcs_studio.analysis.profiling import umap_embedding

    embedded = umap_embedding(wells[features].to_numpy(float), 15, 0.1, "cosine", options.seed)
    if embedded is None:
        return None
    meta = [c for c in (*STRATA, options.treatment, "Metadata_Role") if c in wells]
    frame = wells[meta].reset_index(drop=True).copy()
    frame["UMAP1"], frame["UMAP2"] = embedded[:, 0], embedded[:, 1]
    return frame


def evaluate(
    wells: pd.DataFrame,
    consensus: pd.DataFrame,
    consensus_by: list[str],
    options: EvaluationOptions | None = None,
    moa: pd.DataFrame | None = None,
    aggregated: pd.DataFrame | None = None,
) -> Evaluation:
    """Evaluate well profiles (after feature selection) and consensus profiles."""
    options = options or EvaluationOptions()
    key = options.treatment
    if key not in wells:
        raise ValueError(f"No column {key!r} in the profiles (choose --treatment)")
    act = activity(wells, options)
    features = _complete(consensus, [f for f in _features(consensus) if f in wells])
    labels = _label(consensus, consensus_by)
    cosine, pearson = similarity(consensus, features, labels)
    vehicle_treatments = set(wells.loc[_is_vehicle(wells), key].fillna("").astype(str))
    compounds = (
        wells.assign(**{key: wells[key].fillna("").astype(str)})
        .groupby(key)["Metadata_Compound"]
        .first()
        if "Metadata_Compound" in wells
        else pd.Series(dtype=object)
    )
    treatment = consensus[consensus_by[0]].astype(str).to_numpy()
    info = pd.DataFrame(
        {
            "treatment": treatment,
            "compound": [compounds.get(t, t) for t in treatment],
            "vehicle": [t in vehicle_treatments for t in treatment],
        },
        index=labels.to_numpy(),
    )
    info["moa"] = None
    result = Evaluation(
        activity=act,
        cosine=cosine,
        pearson=pearson,
        similar=similar(cosine, options.neighbours),
        batch=batch_effects(aggregated, wells, options) if aggregated is not None else {},
    )
    if moa is not None:
        mapping = dict(zip(moa["compound"], moa["moa"], strict=True))
        info["moa"] = [
            mapping.get(c, mapping.get(t))
            for c, t in zip(info["compound"], info["treatment"], strict=True)
        ]
        active = set(act.loc[act["active"], key].astype(str))
        rows = consensus.reset_index(drop=True).assign(
            compound=info["compound"].to_numpy(), moa=info["moa"].to_numpy()
        )
        chosen = rows[[t in active for t in treatment]]
        result.consistency = consistency(chosen, features, options)
        result.moa_suggestions, result.moa_accuracy = moa_suggestions(cosine, info, options.k)
    if options.umap:
        result.umap = _umap(wells, _complete(wells, _features(wells)), options)
    held = act[act["holdout"]]
    result.log = {
        "treatments": int((~act["holdout"]).sum()),
        "active": int((act["active"] & ~act["holdout"]).sum()),
        "holdout_active": bool(held["active"].any()) if len(held) else None,
        "features": len(features),
        "moa_annotated": int(info["moa"].notna().sum()),
        "moa_accuracy": result.moa_accuracy,
        "consistent_moas": None
        if result.consistency is None
        else int(result.consistency["consistent"].sum()),
    }
    return result


# --- files ---------------------------------------------------------------------------------


def _heatmap(cosine: pd.DataFrame, path: Path) -> list[str]:
    """Clustered cosine heat map (average linkage on 1 − cosine); returns the leaf order."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform

    names = list(cosine.index)
    order = names
    if len(names) > 2:
        distance = np.clip(1 - cosine.to_numpy(float), 0, 2)
        np.fill_diagonal(distance, 0)
        tree = linkage(squareform((distance + distance.T) / 2, checks=False), "average")
        order = [names[i] for i in dendrogram(tree, no_plot=True)["leaves"]]
    size = min(max(4.0, 0.28 * len(names) + 2), 24.0)
    fig, ax = plt.subplots(figsize=(size + 1, size))
    art = ax.imshow(cosine.loc[order, order].to_numpy(float), cmap="RdBu_r", vmin=-1, vmax=1)
    if len(names) <= 80:
        ax.set_xticks(range(len(order)), order, rotation=90, fontsize=6)
        ax.set_yticks(range(len(order)), order, fontsize=6)
    fig.colorbar(art, ax=ax, label="cosine similarity", shrink=0.7)
    ax.set_title("Consensus profiles (clustered)")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return order


def _activity_figure(table: pd.DataFrame, key: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    y = -np.log10(table["corrected_p_value"].clip(lower=1e-10).to_numpy(float))
    colours = np.where(table["holdout"], "#bbbbbb", np.where(table["active"], "#ee6677", "#4477aa"))
    ax.scatter(table["mAP"].to_numpy(float), y, c=colours.tolist(), s=24, linewidths=0)
    for x_, y_, name in zip(table["mAP"], y, table[key], strict=True):
        if len(table) <= 40:
            ax.annotate(str(name), (x_, y_), fontsize=6, xytext=(3, 2), textcoords="offset points")
    threshold = table.attrs.get("threshold", 0.05)
    ax.axhline(-np.log10(threshold), color="#888888", lw=0.8, ls="--")
    ax.set_xlabel("mAP (vs vehicle wells)")
    ax.set_ylabel("−log10 FDR")
    ax.set_title("Phenotypic activity")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _table_html(frame: pd.DataFrame | None, limit: int = 200) -> str:
    if frame is None or frame.empty:
        return "<p class=note>None.</p>"
    return str(
        frame.head(limit).to_html(
            index=False, float_format=lambda v: f"{v:.3g}", border=0, classes="t"
        )
    )


def write_report(
    result: Evaluation, out: Path, options: EvaluationOptions, figures: list[str]
) -> Path:
    batch_rows = [
        {"grouped by": column, "stage": stage, **values}
        for column, stages in result.batch.items()
        for stage, values in stages.items()
    ]
    images = "".join(
        f'<img src="figures/{html.escape(f)}" alt="{html.escape(f)}">' for f in figures
    )
    log = result.log
    summary = (
        f"{log['active']} of {log['treatments']} treatments active (FDR &lt; {options.threshold}); "
        f"held-out vehicle wells active: {log['holdout_active']}."
    )
    if log["moa_accuracy"] is not None:
        summary += f" MoA kNN top-1 accuracy (leave one compound out): {log['moa_accuracy']:.0%}."
    batch_html = _table_html(pd.DataFrame(batch_rows) if batch_rows else None)
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Profile evaluation</title>
<style>body{{font:14px system-ui,sans-serif;margin:24px;color:#222;background:#fff}}
h1{{font-size:20px}}h2{{font-size:16px;margin-top:28px}}.t{{border-collapse:collapse;font-size:12px}}
.t td,.t th{{padding:2px 8px;border-bottom:1px solid #ddd;text-align:left}}
img{{max-width:48%;margin:6px;vertical-align:top}}.note{{color:#666}}</style></head><body>
<h1>Cell Painting profile evaluation</h1><p>{summary}</p>{images}
<h2>Activity (copairs mAP vs vehicle wells)</h2>{_table_html(result.activity)}
<h2>Consistency (mAP of active treatments by MoA)</h2>{_table_html(result.consistency)}
<h2>MoA suggestions (kNN, k = {options.k})</h2>{_table_html(result.moa_suggestions)}
<h2>Batch effects (vehicle wells)</h2>{batch_html}
<p class=note>Higher mAP / silhouette = vehicle wells cluster by that grouping. "Before" =
aggregated profiles standardised over all wells; "after" = normalized, feature-selected.</p>
<h2>Most similar treatments</h2>{_table_html(result.similar)}
<p class=note>Options: {html.escape(json.dumps(options.model_dump()))}</p></body></html>"""
    path = out / "evaluation.html"
    path.write_text(page, encoding="utf-8")
    return path


def write_evaluation(result: Evaluation, out: Path, options: EvaluationOptions) -> list[str]:
    from hcs_studio.analysis.profiling import save_scatter

    out = Path(out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    written = [
        "activity.csv",
        "similarity_cosine.csv",
        "similarity_pearson.csv",
        "similar.csv",
        "batch.json",
    ]
    result.activity.to_csv(out / "activity.csv", index=False)
    result.cosine.to_csv(out / "similarity_cosine.csv")
    result.pearson.to_csv(out / "similarity_pearson.csv")
    result.similar.to_csv(out / "similar.csv", index=False)
    (out / "batch.json").write_text(json.dumps(result.batch, indent=1), encoding="utf-8")
    if result.consistency is not None:
        result.consistency.to_csv(out / "consistency.csv", index=False)
        written.append("consistency.csv")
    if result.moa_suggestions is not None:
        result.moa_suggestions.to_csv(out / "moa_suggestions.csv", index=False)
        written.append("moa_suggestions.csv")
    figures = ["activity.png", "similarity_heatmap.png"]
    result.activity.attrs["threshold"] = options.threshold
    _activity_figure(result.activity, options.treatment, out / "figures" / "activity.png")
    _heatmap(result.cosine, out / "figures" / "similarity_heatmap.png")
    if result.umap is not None:
        result.umap.to_csv(out / "umap.csv", index=False)
        written.append("umap.csv")
        x, y = result.umap["UMAP1"].to_numpy(), result.umap["UMAP2"].to_numpy()
        for column, name in (
            (options.treatment, "umap_treatment.png"),
            ("Metadata_Plate", "umap_plate.png"),
        ):
            if column in result.umap:
                save_scatter(
                    out / "figures" / name,
                    x,
                    y,
                    result.umap[column],
                    title=f"UMAP of well profiles by {column.removeprefix('Metadata_')}",
                    xlabel="UMAP1",
                    ylabel="UMAP2",
                )
                figures.append(name)
    write_report(result, out, options, figures)
    (out / "evaluation.json").write_text(
        json.dumps({"options": options.model_dump(), "log": result.log}, indent=1),
        encoding="utf-8",
    )
    return [*written, *(f"figures/{f}" for f in figures), "evaluation.html", "evaluation.json"]


def run_evaluation(
    folder: Path,
    options: EvaluationOptions | None = None,
    moa: Path | None = None,
    out: Path | None = None,
) -> Evaluation:
    """Evaluate a profiling folder written by ``run_profiling`` (files go next to it)."""
    folder = Path(folder)
    options = options or EvaluationOptions()
    record = json.loads((folder / "profiling.json").read_text(encoding="utf-8"))
    by = list(record["log"].get("consensus_by") or [options.treatment])
    wells = pd.read_parquet(folder / "feature_selected.parquet")
    consensus = pd.read_parquet(folder / "consensus.parquet")
    aggregated = pd.read_parquet(folder / "aggregated.parquet")
    if options.treatment not in wells and by[0] in wells:
        options = options.model_copy(update={"treatment": by[0]})
    result = evaluate(wells, consensus, by, options, read_moa(moa) if moa else None, aggregated)
    write_evaluation(result, Path(out) if out else folder, options)
    return result
