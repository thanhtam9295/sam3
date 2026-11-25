"""Command-line entry point: launches the GUI or prints environment information."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hcs_studio import __version__
from hcs_studio.utils.gpu import detect_gpu


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hcs-studio",
        description="Image analysis for high-content cellular imaging.",
    )
    parser.add_argument("--version", action="version", version=f"hcs-studio {__version__}")
    sub = parser.add_subparsers(dest="command")
    reason_commands: list[argparse.ArgumentParser] = []
    gui = sub.add_parser("gui", help="launch the desktop application (default)")
    gui.add_argument(
        "--smoke-test",
        action="store_true",
        help="open the window, show the test image and close (checks an installation)",
    )
    sub.add_parser("info", help="print version and GPU status, then exit")
    inspect = sub.add_parser(
        "inspect",
        help="describe how a plate folder or image file is read (opens nothing in the GUI)",
    )
    inspect.add_argument("path", type=Path)
    sub.add_parser("modules", help="list processing modules and their parameters")
    report = sub.add_parser("report", help="(re)write index.html of a run folder")
    reason_commands.append(report)
    report.add_argument("folder", type=Path)
    train = sub.add_parser("train-cellpose", help="fine-tune Cellpose on edited masks")
    reason_commands.append(train)
    train.add_argument("--data", type=Path, required=True, help="training folder")
    train.add_argument("--base", default="cpsam_v2", help="model to start from")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--learning-rate", type=float, default=1e-5)
    train.add_argument("--name", default="hcs_custom", help="name of the new model")
    train.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    analyze = sub.add_parser(
        "analyze",
        help="apply table steps (gates, classifiers, clustering, spatial) to a finished run",
    )
    reason_commands.append(analyze)
    analyze.add_argument("folder", type=Path, help="run folder")
    analyze.add_argument(
        "--pipeline", type=Path, required=True, help="pipeline file whose table steps to apply"
    )
    analyze.add_argument(
        "--only",
        action="store_true",
        help="apply only these steps (default: the run's own table steps first)",
    )
    analyze.add_argument(
        "--new-version",
        action="store_true",
        help="analyse a linked new version of the run (needed for approved, locked runs)",
    )
    annotate = sub.add_parser(
        "annotate", help="apply a plate layout to a finished run (tables and report rebuilt)"
    )
    reason_commands.append(annotate)
    annotate.add_argument("folder", type=Path, help="run folder")
    annotate.add_argument(
        "--layout", type=Path, required=True, help="layout: JSON, long CSV/XLSX or plate-map XLSX"
    )
    layout_cmd = sub.add_parser("layout", help="check or convert a plate layout file")
    layout_cmd.add_argument("path", type=Path)
    layout_cmd.add_argument("--validate", action="store_true", help="list problems")
    layout_cmd.add_argument(
        "--export", type=Path, help="write as .json, .csv, .xlsx, or a figure (.png/.pdf)"
    )
    layout_cmd.add_argument("--plate-map", action="store_true", help="export as a plate-map XLSX")
    layout_cmd.add_argument("--layer", default="compound", help="figure colouring")
    db = sub.add_parser("db", help="experiment registry: scan runs, search, potency across runs")
    db_sub = db.add_subparsers(dest="db_command", required=True)
    db_sub.add_parser("info", help="where the registry is, and what it holds")
    db_backup = db_sub.add_parser("backup", help="copy the registry and the audit chain head")
    db_backup.add_argument("--output", type=Path, required=True)
    db_scan = db_sub.add_parser("scan", help="index run folders (and raw plates) into the registry")
    db_scan.add_argument("paths", nargs="+", type=Path, help="run folders or folders above them")
    db_scan.add_argument("--raw", action="store_true", help="also register raw plate folders")
    db_scan.add_argument("--force", action="store_true", help="re-index unchanged runs")
    reason_commands.append(db_scan)
    db_search = db_sub.add_parser("search", help="find plates")
    for name in ("text", "barcode", "experiment", "assay", "cell-line", "compound", "instrument"):
        db_search.add_argument(f"--{name}", default="")
    db_search.add_argument("--status", default="", help="latest run status, e.g. completed")
    db_potency = db_sub.add_parser("potency", help="EC50 across runs: compound × cell line")
    db_ic50 = db_sub.add_parser("ic50s", help="every EC50 of a compound in a cell line")
    for command in (db_potency, db_ic50):
        command.add_argument("--compound", default="", required=command is db_ic50)
        command.add_argument("--cell-line", default="", required=command is db_ic50)
        command.add_argument("--readout", default="")
    db_plate = db_sub.add_parser("plate", help="one plate: acquisitions, runs, QC")
    db_plate.add_argument("barcode")
    db_norm = db_sub.add_parser("normalize", help="normalize one run's readout on its plate")
    db_norm.add_argument("run", help="run id, run folder name or plate barcode")
    db_norm.add_argument("--readout", required=True)
    db_norm.add_argument(
        "--method",
        required=True,
        choices=["percent_of_control", "percent_inhibition", "robust_z", "b_score", "uniformity"],
    )
    db_norm.add_argument("--map", default="", help="uniformity map name (method uniformity)")
    db_unif = db_sub.add_parser("uniformity", help="store a correction map from a uniformity plate")
    db_unif.add_argument("run", help="the run of a vehicle-only plate")
    db_unif.add_argument("--readout", required=True)
    db_unif.add_argument("--name", required=True)
    db_unif.add_argument("--degree", type=int, default=2, help="surface degree (default 2)")
    db_unif.add_argument(
        "--edge-rings", type=int, default=1, help="outer well rings with their own step (0–3)"
    )
    db_unif.add_argument("--instrument", default="")
    db_unif.add_argument("--assay", default="")
    db_unif.add_argument("--replace", action="store_true", help="replace a map of that name")
    reason_commands.append(db_unif)
    db_sub.add_parser("maps", help="list the uniformity maps")
    db_batch = db_sub.add_parser("batch", help="day and plate effects of the vehicle wells")
    db_batch.add_argument("--readout", required=True)
    db_batch.add_argument("--runs", nargs="*", default=[], help="runs (default: an experiment)")
    db_batch.add_argument("--experiment", default="")
    db_conc = db_sub.add_parser("concordance", help="agreement of two replicate runs")
    db_conc.add_argument("run_a")
    db_conc.add_argument("run_b")
    db_conc.add_argument("--readout", required=True)
    db_conc.add_argument("--r-min", type=float, default=0.8)
    db_conc.add_argument("--fold-max", type=float, default=3.0)
    db_chart = db_sub.add_parser("chart", help="control chart of a metric over runs")
    db_chart.add_argument(
        "--metric",
        required=True,
        choices=[
            "z_prime",
            "vehicle_mean",
            "vehicle_cv_pct",
            "positive_mean",
            "positive_cv_pct",
            "log_ec50",
        ],
    )
    db_chart.add_argument("--readout", default="")
    db_chart.add_argument("--compound", default="", help="reference compound (metric log_ec50)")
    for name in ("cell-line", "assay", "instrument", "experiment"):
        db_chart.add_argument(f"--{name}", default="")
    db_chart.add_argument("--baseline", type=int, default=20, help="runs that set the limits")
    db_chart.add_argument(
        "--rules", default="1_3s,2of3_2s,7x", help="e.g. 1_3s,2of3_2s,4of5_1s,7x,6t"
    )
    db_msr = db_sub.add_parser("msr", help="minimum significant ratio of a reference compound")
    db_msr.add_argument("--compound", required=True)
    db_msr.add_argument("--cell-line", default="")
    db_msr.add_argument("--readout", default="")
    db_msr.add_argument("--window", type=int, default=6, help="runs in the rolling MSR")
    db_bridge = db_sub.add_parser("bridge", help="bridging report: plates imaged on 2 instruments")
    db_bridge.add_argument("--instrument-a", required=True)
    db_bridge.add_argument("--instrument-b", required=True)
    db_bridge.add_argument("--readout", required=True)
    db_bridge.add_argument("--msr", type=float, default=3.0)
    db_bridge.add_argument("--output", type=Path, required=True, help="folder for the report")
    reason_commands.append(db_bridge)
    db_event = db_sub.add_parser("event", help="annotate an event (new lot, vial, service …)")
    db_event.add_argument(
        "--kind",
        required=True,
        choices=["reagent_lot", "cell_vial", "instrument_service", "instrument_change", "other"],
    )
    db_event.add_argument("--on", required=True, help="date, YYYY-MM-DD")
    for name in ("instrument", "assay", "cell-line", "description"):
        db_event.add_argument(f"--{name}", default="")
    reason_commands.append(db_event)
    for command in (db_search, db_potency, db_ic50, db_norm, db_batch, db_conc, db_chart, db_msr):
        command.add_argument("--export", type=Path, help="write the table as .csv or .xlsx")
    ilastik_cmd = sub.add_parser(
        "ilastik", help="ilastik backend: find ilastik and Fiji, check versions, save settings"
    )
    reason_commands.append(ilastik_cmd)
    ilastik_cmd.add_argument(
        "--detect", action="store_true", help="report the ilastik / Fiji installation found"
    )
    ilastik_cmd.add_argument("--set-ilastik", type=Path, help="ilastik executable (or its Python)")
    ilastik_cmd.add_argument("--set-fiji", type=Path, help="Fiji folder or launcher")
    ilastik_cmd.add_argument("--threads", type=int, help="ilastik threads (-1 = no limit)")
    ilastik_cmd.add_argument("--ram-mb", type=int, help="ilastik RAM limit in MB (0 = no limit)")
    ilastik_cmd.add_argument(
        "--max-processes", type=int, help="ilastik processes allowed at once (default 1)"
    )
    ilastik_cmd.add_argument(
        "--save", action="store_true", help="save the given settings (and the paths found)"
    )
    ilastik_cmd.add_argument("--info", type=Path, help="what an ilastik project (.ilp) holds")
    ilastik_cmd.add_argument(
        "--preset", type=Path, help="write a pipeline for this project (.ilp) to --output"
    )
    ilastik_cmd.add_argument("--output", type=Path, help="pipeline file for --preset")
    ilastik_cmd.add_argument("--class", dest="object_class", help="class that becomes objects")
    ilastik_cmd.add_argument(
        "--boundary", type=Path, help="Multicut preset: the boundary pixel-classification .ilp"
    )
    ilastik_cmd.add_argument(
        "--channels", type=int, nargs="+", help="channels sent to ilastik (from 1)"
    )
    qc_cmd = sub.add_parser(
        "qc", help="image QC of a plate from raw data (focus, saturation, artifacts …)"
    )
    qc_cmd.add_argument("input", nargs="?", type=Path, help="raw plate (folder or file)")
    qc_cmd.add_argument("--output", type=Path, help="folder that receives qc/ (not in raw data)")
    qc_cmd.add_argument("--rules", type=Path, help="QC rules YAML (default: the shipped rules)")
    qc_cmd.add_argument("--wells", default="all", help="e.g. A01-H12 or A01,B03")
    qc_cmd.add_argument("--workers", type=int, default=4, help="threads reading fields")
    qc_cmd.add_argument("--write-rules", type=Path, help="write the default rules here and exit")
    qc_cmd.add_argument("--reason", default="", help="why (recorded)")

    review = sub.add_parser("qc-review", help="accept or override an image-QC decision")
    review.add_argument("folder", type=Path, help="run folder (or QC output folder)")
    review.add_argument("--well", required=True)
    review.add_argument("--field", type=int, help="field number; omit to review the well")
    review.add_argument("--action", required=True, choices=["confirm", "include", "exclude"])
    review.add_argument("--reason", required=True, help="why (recorded)")
    qc_train = sub.add_parser("qc-train", help="train learned QC from reviewed QC folders")
    qc_train.add_argument("folders", nargs="+", type=Path)
    qc_train.add_argument("--output", type=Path, required=True, help="classifier file (.joblib)")
    qc_train.add_argument(
        "--reviewed-only", action="store_true", help="do not count unflagged fields as good"
    )
    qc_train.add_argument("--reason", default="", help="why (recorded)")

    assay = sub.add_parser("assay", help="oncology assay modules (cell cycle …)")
    assay_sub = assay.add_subparsers(dest="assay_command", required=True)
    assay_sub.add_parser("list", help="the assay modules and their channel roles")
    assay_preset = assay_sub.add_parser("preset", help="write an assay's preset pipeline")
    assay_preset.add_argument("name")
    assay_preset.add_argument(
        "--channels", default="", help="the plate's channel names, comma-separated"
    )
    assay_preset.add_argument(
        "--map",
        action="append",
        default=[],
        metavar="ROLE=CHANNEL",
        help="e.g. --map dna=DAPI --map edu=EdU",
    )
    assay_preset.add_argument("--nucleus-um", type=float, default=10.0)
    assay_preset.add_argument("--z-planes", type=int, default=1)
    assay_preset.add_argument("--output", type=Path, required=True)
    assay_run = assay_sub.add_parser("run", help="apply an assay to a finished run")
    assay_run.add_argument("folder", type=Path)
    assay_run.add_argument("name")
    assay_run.add_argument("--map", action="append", default=[], metavar="ROLE=CHANNEL")
    assay_run.add_argument("--reason", default="")

    kinetics = sub.add_parser("kinetics", help="time courses of time-lapse plates")
    kinetics_sub = kinetics.add_subparsers(dest="kinetics_command", required=True)
    kinetics_table = kinetics_sub.add_parser(
        "table",
        help="readouts per well and time point (one time-lapse run, or several "
        "acquisitions of the same plate joined on their times)",
    )
    kinetics_table.add_argument("folders", type=Path, nargs="+", help="run folder(s)")
    kinetics_table.add_argument(
        "--treated-at",
        default=None,
        help="when the compounds were added (ISO 8601; default: "
        "the layout's treatment time, else the plate.dispensed event)",
    )
    kinetics_table.add_argument("--output", type=Path, default=None, help="CSV to write")
    kinetics_analyse = kinetics_sub.add_parser(
        "analyse",
        help="growth fits, doubling times, IC50 and GR over time (one time-lapse run, "
        "or several acquisitions of the same plate)",
    )
    kinetics_analyse.add_argument("folders", type=Path, nargs="+", help="run folder(s)")
    kinetics_analyse.add_argument("--readout", default="auto")
    kinetics_analyse.add_argument("--treated-at", default=None)
    kinetics_analyse.add_argument("--interval-h", type=float, default=None)
    kinetics_analyse.add_argument(
        "--models", default=",".join(("exponential", "lag_exponential", "logistic"))
    )
    kinetics_analyse.add_argument(
        "--output", type=Path, default=None, help="folder (default: <run>/kinetics)"
    )

    painting = sub.add_parser("cellpainting", help="Cell Painting profiling (CellProfiler style)")
    painting_sub = painting.add_subparsers(dest="painting_command", required=True)
    painting_preset = painting_sub.add_parser("preset", help="write the Cell Painting pipeline")
    painting_preset.add_argument(
        "--channels", required=True, help="the plate's channel names, comma-separated"
    )
    painting_preset.add_argument(
        "--map",
        action="append",
        default=[],
        metavar="ROLE=CHANNEL",
        help="dna, er, rna, agp, mito (and cell: the channel outlining cells)",
    )
    painting_preset.add_argument("--nucleus-um", type=float, default=10.0)
    painting_preset.add_argument("--cell-um", type=float, default=12.0, help="cell growth (µm)")
    painting_preset.add_argument("--output", type=Path, required=True)
    painting_cells = painting_sub.add_parser(
        "single-cells", help="one row per cell, CellProfiler names, Metadata_ columns"
    )
    painting_cells.add_argument("run", type=Path)
    painting_cells.add_argument("--map", action="append", default=[], metavar="ROLE=CHANNEL")
    painting_cells.add_argument(
        "--output", type=Path, help="Parquet (default: <run>/profiling/single_cell.parquet)"
    )
    painting_profile = painting_sub.add_parser(
        "profile", help="well and treatment profiles (pycytominer: aggregate → consensus)"
    )
    painting_profile.add_argument(
        "sources", type=Path, nargs="+", help="run folders or single-cell Parquet files"
    )
    painting_profile.add_argument("--output", type=Path, required=True)
    painting_profile.add_argument(
        "--normalize",
        default="mad_robustize",
        choices=["mad_robustize", "robustize", "standardize", "spherize", "none"],
    )
    painting_profile.add_argument("--reference", default="vehicle", choices=["vehicle", "all"])
    painting_profile.add_argument(
        "--across-plates",
        action="store_true",
        help="normalize all plates together (default: each plate)",
    )
    painting_profile.add_argument(
        "--operations",
        default=",".join(
            [
                "variance_threshold",
                "correlation_threshold",
                "drop_na_columns",
                "blocklist",
                "drop_outliers",
            ]
        ),
    )
    painting_profile.add_argument("--corr-threshold", type=float, default=0.9)
    painting_profile.add_argument("--consensus-by", default="Metadata_Treatment,Metadata_CellLine")
    painting_profile.add_argument("--platemap", type=Path, help="plate map CSV (imported data)")
    painting_profile.add_argument(
        "--evaluate",
        action="store_true",
        help="also evaluate the profiles (as `cellpainting evaluate`)",
    )
    painting_profile.add_argument("--moa", type=Path, help="MoA annotations (with --evaluate)")
    painting_profile.add_argument("--reason", default="")
    painting_eval = painting_sub.add_parser(
        "evaluate", help="mAP activity/consistency, similarity, MoA kNN, batch checks, report"
    )
    painting_eval.add_argument("folder", type=Path, help="a `cellpainting profile` output folder")
    painting_eval.add_argument("--moa", type=Path, help="CSV/XLSX: compound, moa")
    painting_eval.add_argument("--treatment", default="Metadata_Treatment")
    painting_eval.add_argument("--threshold", type=float, default=0.05, help="FDR")
    painting_eval.add_argument("--null-size", type=int, default=10000)
    painting_eval.add_argument(
        "--any-plate-negatives",
        action="store_true",
        help="compare with vehicle wells of all plates",
    )
    painting_eval.add_argument("-k", type=int, default=5, help="neighbours for MoA suggestions")
    painting_eval.add_argument("--no-umap", action="store_true")
    painting_eval.add_argument("--reason", default="")
    painting_similar = painting_sub.add_parser("similar", help="most similar treatments")
    painting_similar.add_argument("folder", type=Path, help="an evaluated profiling folder")
    painting_similar.add_argument("--to", required=True, help="treatment (row of the matrix)")
    painting_similar.add_argument("-n", type=int, default=10)
    painting_import = painting_sub.add_parser(
        "import", help="CellProfiler SQLite / CSV folder / Parquet → single-cell Parquet"
    )
    painting_import.add_argument("source", type=Path)
    painting_import.add_argument("--output", type=Path, required=True)

    screen = sub.add_parser("screen", help="primary screens: hit calling, library maps")
    screen_sub = screen.add_subparsers(dest="screen_command", required=True)
    screen_score = screen_sub.add_parser("score", help="score plates and call hits")
    screen_score.add_argument("folders", type=Path, nargs="+", help="run folders of the screen")
    screen_score.add_argument("--config", type=Path, required=True, help="screen YAML/JSON")
    screen_score.add_argument("--output", type=Path, required=True, help="screen folder")
    screen_score.add_argument(
        "--gallery", type=int, default=0, help="images of the first N hits (hit gallery)"
    )
    screen_score.add_argument("--reason", default="")
    screen_report = screen_sub.add_parser(
        "report", help="rewrite a screen's report (and hit gallery) from its runs"
    )
    screen_report.add_argument("output", type=Path, help="screen folder")
    screen_report.add_argument("--gallery", type=int, default=0, help="first N hits")
    screen_report.add_argument("--reason", default="")
    screen_pick = screen_sub.add_parser(
        "cherrypick", help="Echo pick list and confirmation plates for the hits"
    )
    screen_pick.add_argument("output", type=Path, help="screen folder")
    screen_pick.add_argument(
        "--template",
        type=Path,
        required=True,
        help="layout with the controls of the confirmation plates",
    )
    screen_pick.add_argument(
        "--mode", choices=["replicates", "dose_response"], default="replicates"
    )
    screen_pick.add_argument("--replicates", type=int, default=3)
    screen_pick.add_argument("--concentration-um", type=float, help="replicates: final conc.")
    screen_pick.add_argument("--top-um", type=float, help="dose_response: highest conc.")
    screen_pick.add_argument("--points", type=int, default=8)
    screen_pick.add_argument("--factor", type=float, default=3.0)
    screen_pick.add_argument("--assay-volume-ul", type=float, default=40.0)
    screen_pick.add_argument("--droplet-nl", type=float, default=2.5)
    screen_pick.add_argument("--prefix", default="CONF", help="destination barcode prefix")
    screen_pick.add_argument("--backfill", metavar="PLATE:WELL", help="vehicle source well")
    screen_pick.add_argument("--format", type=Path, help="pick-list format (YAML/JSON)")
    screen_pick.add_argument("--library", type=Path, help="library map (default: registry)")
    screen_pick.add_argument(
        "--register", action="store_true", help="pre-register the confirmation layouts by barcode"
    )
    screen_pick.add_argument("--outbox", default="", help="folder for cherrypick.request events")
    screen_pick.add_argument("--scheduler-url", default="", help="POST the events here")
    screen_pick.add_argument("--reason", default="")
    screen_review = screen_sub.add_parser("review", help="include or exclude a hit by hand")
    screen_review.add_argument("output", type=Path, help="screen folder")
    screen_review.add_argument("--compound", required=True, help="compound name or key")
    screen_review.add_argument("--action", choices=["include", "exclude"], required=True)
    screen_review.add_argument("--reason", required=True, help="why (recorded)")
    screen_library = screen_sub.add_parser("library", help="store library plate maps")
    screen_library.add_argument("map", type=Path, help="library map CSV/XLSX")
    screen_library.add_argument("--name", default="", help="library name")
    screen_library.add_argument("--reason", default="")
    screen_layouts = screen_sub.add_parser(
        "layouts", help="assay-plate layouts from library maps (stamp, quadrants, Echo)"
    )
    screen_layouts.add_argument("map", type=Path, help="library map CSV/XLSX")
    screen_layouts.add_argument(
        "--template", type=Path, required=True, help="layout with the controls (JSON/CSV/XLSX)"
    )
    screen_layouts.add_argument("--rule", choices=["stamp", "quadrants", "echo"], required=True)
    screen_layouts.add_argument(
        "--dest",
        action="append",
        default=[],
        metavar="LIBRARY=ASSAY",
        help="assay barcode per library plate (stamp; quadrants: one)",
    )
    screen_layouts.add_argument("--sources", default="", help="quadrants: four library plates")
    screen_layouts.add_argument(
        "--dilution",
        type=float,
        default=1.0,
        help="final / library concentration (stamp, quadrants)",
    )
    screen_layouts.add_argument("--transfers", type=Path, help="echo: transfer file")
    screen_layouts.add_argument("--assay-volume-ul", type=float, help="echo: assay volume")
    screen_layouts.add_argument("--output", type=Path, required=True, help="folder for layouts")
    screen_layouts.add_argument(
        "--register", action="store_true", help="save as layout versions keyed by barcode"
    )
    screen_layouts.add_argument("--reason", default="")

    selftest = sub.add_parser(
        "selftest", help="check this installation on a small synthetic plate (about a minute)"
    )
    selftest.add_argument("--workers", type=int, default=2)
    selftest.add_argument("--keep", type=Path, help="keep the test plate and run in this folder")
    acceptance = sub.add_parser(
        "acceptance", help="acceptance tests on real data (PROMPT.md §10), PASS/FAIL report"
    )
    checks = acceptance.add_subparsers(dest="check", required=True)
    counts = checks.add_parser("counts", help="counts per field against manual counts (±5 %%)")
    counts.add_argument("--run", type=Path, required=True)
    counts.add_argument("--manual", type=Path, required=True, help="CSV: well, field, count")
    counts.add_argument("--object-type", default="nuclei")
    counts.add_argument("--tolerance", type=float, default=0.05)
    coloc = checks.add_parser("coloc", help="Pearson/Manders against Fiji Coloc 2 (±0.02)")
    coloc.add_argument("folders", type=Path, nargs="+", help="folders from Export for Fiji")
    coloc.add_argument("--tolerance", type=float, default=0.02)
    run_check = checks.add_parser("run", help="a run folder is complete; report works offline")
    run_check.add_argument("folder", type=Path)
    kill = checks.add_parser("kill-resume", help="kill an unattended run part-way, then resume")
    kill.add_argument("--pipeline", type=Path, required=True)
    kill.add_argument("--input", type=Path, required=True)
    kill.add_argument("--output", type=Path, required=True)
    kill.add_argument("--layout", type=Path)
    kill.add_argument("--kill-after", type=int, default=10, help="complete wells before the kill")
    kill.add_argument("--workers", default="auto")
    beads = checks.add_parser("beads", help="deconvolution narrows the FWHM of beads")
    beads.add_argument("path", type=Path, help="bead Z stack (.nd2, OME-TIFF, …)")
    beads.add_argument("--channel", type=int, default=1)
    beads.add_argument("--psf", type=Path, help="measured PSF file (default: theoretical)")
    beads.add_argument("--iterations", type=int, default=40)
    acceptance.add_argument(
        "--report", type=Path, help="write the report as JSON (default: acceptance_<check>.json)"
    )
    figure = sub.add_parser(
        "figure", help="regenerate a figure from its sidecar, or make one figure per compound"
    )
    figure.add_argument(
        "--regenerate",
        type=Path,
        metavar="SIDECAR",
        help="the .json written next to an exported figure",
    )
    figure.add_argument("--run", type=Path, help="run folder (with --per-compound)")
    figure.add_argument(
        "--per-compound",
        action="store_true",
        help="a figure per compound: concentrations × cell lines",
    )
    figure.add_argument("--compound", action="append", default=[], help="only these compounds")
    figure.add_argument("--source", default="corrected", help="raw, corrected, enhanced, step:<n>")
    figure.add_argument("--format", default="png", choices=["png", "tiff", "svg", "pdf"])
    figure.add_argument("--dpi", type=int, default=300)
    figure.add_argument("--width-mm", type=float, default=180.0)
    figure.add_argument("--split-channels", action="store_true")
    figure.add_argument("--output", type=Path, help="output file (regenerate) or folder")

    pharm = sub.add_parser(
        "pharm",
        help="dose-response curves, combination matrices and synergy scores of finished runs",
    )
    reason_commands.append(pharm)
    pharm.add_argument("runs", nargs="*", type=Path, help="run folder(s); plates are merged")
    pharm.add_argument(
        "--run",
        dest="extra_runs",
        action="append",
        type=Path,
        default=[],
        help="run folder (may be repeated)",
    )
    pharm.add_argument(
        "--readout", required=True, help="per-well readout(s), comma separated, e.g. nuclei_count"
    )
    pharm.add_argument(
        "--normalize",
        default="percent_inhibition",
        choices=["percent_of_control", "percent_inhibition", "gr"],
    )
    pharm.add_argument(
        "--models",
        default="bliss,hsa,loewe,zip",
        help="synergy models: bliss, hsa, loewe, zip (comma separated; empty = none)",
    )
    pharm.add_argument("--curve", default="4pl", choices=["4pl", "5pl", "3pl_bottom", "3pl_top"])
    pharm.add_argument("--robust", action="store_true", help="robust (soft L1) curve fitting")
    pharm.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="bootstrap intervals with this many resamples (default: covariance)",
    )
    pharm.add_argument(
        "--outliers",
        type=float,
        default=None,
        metavar="K",
        help="exclude wells beyond median ± K·MAD of their replicates",
    )
    pharm.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="PLATE:WELL",
        help="exclude a well (repeatable)",
    )
    pharm.add_argument("--z-threshold", type=float, default=0.5, help="flag plates below this Z′")
    pharm.add_argument("--doubling-time", type=float, help="GR: doubling time (h)")
    pharm.add_argument("--duration", type=float, help="GR: assay duration (h)")
    pharm.add_argument(
        "--time-zero",
        action="append",
        type=Path,
        default=[],
        help="GR: run folder of a time-zero plate",
    )
    pharm.add_argument("--surfaces", action="store_true", help="also draw 3D synergy surfaces")
    pharm.add_argument("--no-figures", action="store_true")
    pharm.add_argument("--no-pdf", action="store_true")
    pharm.add_argument(
        "--output", type=Path, help="output folder (default: <first run>/pharmacology)"
    )

    profile = sub.add_parser(
        "profile",
        help="PCA/UMAP phenotypic profiles, distance to control and dose-response of runs",
    )
    reason_commands.append(profile)
    profile.add_argument("runs", type=Path, nargs="+", help="run folder(s), one per plate")
    profile.add_argument("--object-type", default="nuclei")
    profile.add_argument("--level", choices=["cell", "well"], default="cell")
    profile.add_argument(
        "--normalisation",
        choices=["negative_controls", "all_wells", "none"],
        default="negative_controls",
    )
    profile.add_argument("--plate-correction", action="store_true", help="centre each plate")
    profile.add_argument("--max-cells", type=int, default=50_000)
    profile.add_argument("--correlation", type=float, default=0.95, help="drop features above")
    profile.add_argument("--pca-components", type=int, default=10)
    profile.add_argument("--no-umap", action="store_true")
    profile.add_argument("--neighbors", type=int, default=15, help="UMAP n_neighbors")
    profile.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist")
    profile.add_argument("--metric", default="euclidean", help="UMAP metric")
    profile.add_argument("--seed", type=int, default=0)
    profile.add_argument(
        "--dose-readouts",
        default="profile_distance_to_control",
        help="comma-separated per-well readouts for dose-response fits",
    )
    profile.add_argument("--output", type=Path, help="default: advanced/ of the first run")
    classify = sub.add_parser(
        "train-classifier", help="train an object classifier from labelled objects of a run"
    )
    reason_commands.append(classify)
    classify.add_argument("--run", type=Path, required=True, help="run folder")
    classify.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="CSV with well, field, object_id, label (and optionally timepoint)",
    )
    classify.add_argument("--object-type", default="nuclei")
    classify.add_argument(
        "--algorithm", choices=["random_forest", "hist_gradient_boosting"], default="random_forest"
    )
    classify.add_argument(
        "--normalisation", choices=["plate_robust_z", "none"], default="plate_robust_z"
    )
    classify.add_argument("--output", type=Path, required=True, help="classifier file (.joblib)")
    bench = sub.add_parser("benchmark", help="time deconvolution of a synthetic stack")
    bench.add_argument(
        "--shape", type=int, nargs=3, default=[30, 2048, 2048], metavar=("Z", "Y", "X")
    )
    bench.add_argument("--iterations", type=int, default=40)
    bench.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    bench.add_argument("--tile", type=int, default=0, help="tile size in pixels; 0 = automatic")
    bench.add_argument("--max-memory-gb", type=float, help="memory limit (default: half free RAM)")
    run = sub.add_parser("run", help="run a pipeline on a plate, unattended")
    reason_commands.append(run)
    run.add_argument("--pipeline", type=Path, help="pipeline YAML or JSON file")
    run.add_argument(
        "--input",
        type=Path,
        action="append",
        help="plate folder or image file (repeat for several)",
    )
    run.add_argument(
        "--workers", default="1", help="worker processes: a number, or 'auto' (default: 1)"
    )
    run.add_argument(
        "--resume", type=Path, help="continue an interrupted run folder (skips completed wells)"
    )
    run.add_argument(
        "--retry-failed", type=Path, help="run folder: redo only the wells that failed"
    )
    run.add_argument("--wells", default="all", help="e.g. all, A01-H12, or B02,B03 (default: all)")
    run.add_argument("--output", type=Path, help="output root (default: next to the raw data)")
    run.add_argument("--layout", type=Path, help="plate layout (CSV, XLSX, plate-map XLSX or JSON)")
    run.add_argument(
        "--no-images",
        action="store_true",
        help="do not save corrected images (masks, previews and tables are still written)",
    )
    run.add_argument(
        "--image-dtype",
        choices=["float32", "original"],
        default="float32",
        help="corrected images as float32 (default) or in the camera's bit depth (rounded, "
        "clipped; smaller files)",
    )
    run.add_argument(
        "--max-memory-gb", type=float, help="RAM the whole run may use (all workers together)"
    )
    run.add_argument(
        "--no-checksums",
        action="store_true",
        help="record input file size and time only, instead of SHA-256 checksums",
    )
    for command in reason_commands:
        command.add_argument(
            "--reason",
            default="",
            help="why (recorded with the change; required for manual exclusions)",
        )
    from hcs_studio.cli_models import add_parsers

    add_parsers(sub)
    from hcs_studio.cli_security import add_parsers as add_security_parsers

    add_security_parsers(sub)
    from hcs_studio.cli_server import add_parsers as add_server_parsers

    add_server_parsers(sub)
    from hcs_studio.cli_assistant import add_parsers as add_assistant_parsers

    add_assistant_parsers(sub)
    from hcs_studio.cli_automation import add_parsers as add_automation_parsers

    add_automation_parsers(sub)
    return parser


def _ctx(args: argparse.Namespace) -> Any:
    """The service context of a command: the logged-in user and ``--reason``."""
    from hcs_studio.services import context

    return context(getattr(args, "reason", "") or "")


def main(argv: Sequence[str] | None = None) -> int:
    import multiprocessing

    multiprocessing.freeze_support()  # no-op except in frozen (installer) builds
    args = build_parser().parse_args(argv)
    if args.command == "info":
        print(f"hcs-studio {__version__}")
        print(detect_gpu().summary)
        return 0
    if args.command == "inspect":
        from hcs_studio.inspect import describe_path

        print(describe_path(args.path))
        return 0
    if args.command == "modules":
        print(_describe_modules())
        return 0
    if args.command == "run":
        return _run(args)
    if args.command == "report":
        from hcs_studio.services.operations import write_report

        try:
            path = write_report(_ctx(args), args.folder)
        except (OSError, ValueError, KeyError) as exc:
            print(f"Error: {exc}")
            return 2
        print(f"Report: {path}")
        return 0
    if args.command == "analyze":
        return _analyze(args)
    if args.command == "qc":
        return _qc(args)
    if args.command == "assay":
        return _assay(args)
    if args.command == "cellpainting":
        return _cellpainting(args)
    if args.command in ("accounts", "audit"):
        from hcs_studio.cli_security import run_accounts, run_audit

        return run_accounts(args) if args.command == "accounts" else run_audit(args)
    if args.command in ("worker", "worker-pool", "queue", "serve"):
        from hcs_studio.cli_server import run as run_server

        return run_server(args)
    if args.command in ("run-state", "sign", "unlock"):
        from hcs_studio.cli_security import run_signoff

        return run_signoff(args)
    if args.command in ("gt", "validate-seg", "models"):
        from hcs_studio.cli_models import run_gt, run_models, run_validate

        if args.command == "models":
            return run_models(args)
        return run_gt(args) if args.command == "gt" else run_validate(args)
    if args.command == "auto":
        from hcs_studio.cli_automation import run as run_automation

        return run_automation(args)
    if args.command == "assistant":
        from hcs_studio.cli_assistant import run as run_assistant

        return run_assistant(args)
    if args.command == "screen":
        return _screen(args)
    if args.command == "kinetics":
        return _kinetics(args)
    if args.command == "qc-review":
        return _qc_review(args)
    if args.command == "qc-train":
        return _qc_train(args)
    if args.command == "annotate":
        return _annotate(args)
    if args.command == "layout":
        return _layout(args)
    if args.command == "ilastik":
        return _ilastik(args)
    if args.command == "db":
        return _db(args)
    if args.command == "selftest":
        from hcs_studio.selftest import run_selftest

        checks = run_selftest(workers=args.workers, keep=args.keep)
        failed = [c.name for c in checks if not c.ok]
        print(
            f"\n{len(checks) - len(failed)} of {len(checks)} checks passed"
            + (f"; failed: {', '.join(failed)}" if failed else "")
        )
        return 1 if failed else 0
    if args.command == "acceptance":
        return _acceptance(args)
    if args.command == "profile":
        return _profile(args)
    if args.command == "pharm":
        return _pharm(args)
    if args.command == "figure":
        return _figure(args)
    if args.command == "train-classifier":
        return _train_classifier(args)
    if args.command == "train-cellpose":
        from hcs_studio.processing.base import ModuleError
        from hcs_studio.services.operations import fine_tune_cellpose

        try:
            path, history = fine_tune_cellpose(
                _ctx(args),
                args.data,
                base_model=args.base,
                model_name=args.name,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                device=args.device,
            )
        except ModuleError as exc:
            print(f"Error: {exc}")
            return 2
        print(f"Trained on {history['images_train']} image(s) for {args.epochs} epochs")
        print(f"Model: {path}")
        return 0
    if args.command == "benchmark":
        from hcs_studio.benchmark import run_benchmark
        from hcs_studio.processing.base import ModuleError

        try:
            result = run_benchmark(
                tuple(args.shape),
                args.iterations,
                args.device,
                args.tile,
                args.max_memory_gb,
            )
        except (ModuleError, MemoryError) as exc:
            print(f"Error: {exc}")
            return 2
        print(result.summary())
        return 0
    try:
        from hcs_studio.gui.launch import run_gui
    except ImportError as exc:
        print(
            f"The GUI dependencies are not installed ({exc}).\n"
            'Install them with: pip install "hcs-studio[gui]"'
        )
        return 1
    return run_gui(smoke_test=bool(getattr(args, "smoke_test", False)))


def _describe_modules() -> str:
    from hcs_studio.processing.base import MODULES, load_builtin_modules

    load_builtin_modules()
    lines: list[str] = []
    for name in MODULES:
        module = MODULES.get(name)
        note = "" if module.preserves_linearity else "  [display only: not for measurement]"
        lines.append(f"{name}: {module.title}{note}")
        for key, info in module.Params.model_fields.items():
            if info.exclude:
                continue
            default = (
                "required"
                if info.is_required()
                else f"default {info.get_default(call_default_factory=True)!r}"
            )
            lines.append(f"    {key} ({default}): {info.description or ''}")
    return "\n".join(lines)


def _run(args: argparse.Namespace) -> int:
    import json

    from pydantic import ValidationError

    from hcs_studio.io.base import ReaderError, open_plate, reopen_plate
    from hcs_studio.io.sidecar import apply_overrides
    from hcs_studio.model.layout import load_plate_layout, run_layout_file
    from hcs_studio.pipeline.executor import parse_wells
    from hcs_studio.pipeline.schema import Pipeline
    from hcs_studio.processing.base import ModuleError

    def progress(done: int, total: int, message: str) -> None:
        print(f"[{done}/{total}] {message}", flush=True)

    folder = args.resume or args.retry_failed
    try:
        workers = None if args.workers == "auto" else int(args.workers)
        if workers is not None and workers < 1:
            raise ValueError("--workers must be at least 1")
        jobs: list[tuple[Any, Pipeline, Any, Any, Path | None]] = []
        if folder is not None:
            manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
            pipeline = Pipeline.load(folder / "pipeline.yaml")
            source = manifest["plate"].get("source") or {"path": manifest["plate"]["path"]}
            plate = apply_overrides(reopen_plate(source))
            layout_file = run_layout_file(folder)
            layout = load_plate_layout(layout_file) if layout_file else None
            if args.retry_failed:
                failed = list(manifest.get("failed_wells", {}))
                if not failed:
                    print("No failed wells in this run.")
                    return 0
                wells = parse_wells(",".join(failed), plate)
            else:
                wells = parse_wells(",".join(manifest["wells"]), plate)
            jobs.append((plate, pipeline, wells, layout, None))
        else:
            if args.pipeline is None or not args.input:
                raise ValueError("--pipeline and --input are required (or use --resume)")
            pipeline = Pipeline.load(args.pipeline)
            layout = load_plate_layout(args.layout) if args.layout else None
            for path in args.input:
                plate = apply_overrides(open_plate(path))
                jobs.append((plate, pipeline, parse_wells(args.wells, plate), layout, args.layout))
    except (OSError, ValueError, KeyError, ValidationError, ReaderError, ModuleError) as exc:
        print(f"Error: {exc}")
        return 2

    # Ctrl+C stops after the current field; the run folder can then be resumed.
    import signal
    import threading

    cancel = threading.Event()

    def interrupt(signum: int, frame: Any) -> None:
        if cancel.is_set():
            raise KeyboardInterrupt
        print("\nStopping after the current field (Ctrl+C again to quit at once)…", flush=True)
        cancel.set()

    previous_handler = signal.signal(signal.SIGINT, interrupt)
    try:
        return _run_jobs(args, jobs, workers, folder, progress, cancel)
    finally:
        signal.signal(signal.SIGINT, previous_handler)


def _run_jobs(
    args: argparse.Namespace,
    jobs: list[Any],
    workers: int | None,
    folder: Path | None,
    progress: Any,
    cancel: Any,
) -> int:
    from pydantic import ValidationError

    from hcs_studio.io.base import ReaderError
    from hcs_studio.processing.base import ModuleError
    from hcs_studio.services.operations import run_pipeline

    code = 0
    for plate, pipeline, wells, layout, layout_path in jobs:
        if len(jobs) > 1:
            print(f"== {plate.name} ==", flush=True)
        try:
            result = run_pipeline(
                _ctx(args),
                plate,
                pipeline,
                output_root=args.output,
                wells=wells,
                layout=layout,
                layout_path=layout_path,
                checksums=not args.no_checksums,
                progress=progress,
                workers=workers,
                resume=folder,
                save_images=not args.no_images,
                max_memory_gb=args.max_memory_gb,
                image_dtype=args.image_dtype,
                cancel=cancel,
            )
        except (OSError, ValueError, ValidationError, ReaderError, ModuleError) as exc:
            print(f"Error: {exc}")
            code = 2
            continue
        print(f"Output folder: {result.folder}")
        if result.cancelled:
            print(f'Interrupted. Continue with: hcs-studio run --resume "{result.folder}"')
            return 130
        if result.skipped_wells:
            print(f"Already complete (skipped): {len(result.skipped_wells)} well(s)")
        print(f"Fields written: {result.fields_written}")
        for well, error in result.failed_wells.items():
            print(f"FAILED {well}: {error}")
        if result.failed_wells:
            code = max(code, 1)
    return code


def _qc(args: argparse.Namespace) -> int:
    from hcs_studio.io.base import open_plate
    from hcs_studio.pipeline.executor import parse_wells
    from hcs_studio.qc.rules import default_rules, load_rules, save_rules
    from hcs_studio.services.operations import run_image_qc

    try:
        rules = load_rules(args.rules) if args.rules else default_rules()
        if args.write_rules:
            print(f"Rules: {save_rules(rules, args.write_rules)}")
            return 0
        if args.input is None or args.output is None:
            print("Error: give a plate and --output (or --write-rules)")
            return 2
        plate = open_plate(args.input)
        wells = parse_wells(args.wells, plate)

        def progress(done: int, total: int) -> None:
            if done == total or done % 50 == 0:
                print(f"  {done}/{total} fields", flush=True)

        result = run_image_qc(
            _ctx(args),
            plate,
            rules,
            args.output,
            wells=wells,
            workers=args.workers,
            progress=progress,
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"Error: {exc}")
        return 2
    summary = result.plate
    print(f"Fields: {summary['fields']}, excluded {summary['excluded_fields']}")
    for flag, count in summary["flag_counts"].items():
        if count:
            print(f"  {flag}: {count}")
    print(f"Wells failed: {summary['failed_wells']} of {summary['wells']}")
    print(
        "Plate: " + ("passed" if summary["passed"] else "FAILED — " + "; ".join(summary["reasons"]))
    )
    print(f"Files: {Path(args.output) / 'qc'}")
    return 0 if summary["passed"] else 1


def _assay(args: argparse.Namespace) -> int:
    from hcs_studio.assays.base import assay_names
    from hcs_studio.processing.base import MODULES

    if args.assay_command == "list":
        for name in assay_names():
            cls: Any = MODULES.get(name)
            roles = ", ".join(f"{r.key}{'' if r.required else '?'}" for r in cls.roles)
            print(f"{name}: {cls.title} — channels: {roles}")
        return 0
    try:
        mapping = dict(item.split("=", 1) for item in args.map)
    except ValueError:
        print("Error: --map takes ROLE=CHANNEL")
        return 2
    if args.name not in assay_names():
        print(f"Error: unknown assay {args.name!r} ({', '.join(assay_names())})")
        return 2
    assay_cls: Any = MODULES.get(args.name)
    if args.assay_command == "preset":
        channels = [c.strip() for c in args.channels.split(",") if c.strip()]
        pipeline = assay_cls.preset(
            mapping,
            channels=channels or None,
            nucleus_diameter_um=args.nucleus_um,
            z_planes=args.z_planes,
        )
        pipeline.save(args.output)
        print(f"Pipeline: {args.output}")
        return 0
    from hcs_studio.pipeline.schema import PipelineStep
    from hcs_studio.processing.base import ModuleError
    from hcs_studio.services.operations import analyze_run

    params = {f"{role}_channel": channel for role, channel in mapping.items()}
    try:
        analyze_run(_ctx(args), args.folder, [PipelineStep(module=args.name, params=params)])
    except (OSError, ValueError, ModuleError) as exc:
        print(f"Error: {exc}")
        return 2
    print(f"Report: {args.folder / 'assays' / args.name / 'report.html'}")
    return 0


def _kinetics(args: argparse.Namespace) -> int:
    from hcs_studio.kinetics.timecourse import assemble, build_time_course, time_info

    folders: list[Path] = list(args.folders)
    if args.kinetics_command == "analyse":
        return _kinetics_analyse(args, folders)
    try:
        if len(folders) == 1:
            table = build_time_course(folders[0], treated_at=args.treated_at)
            if table is None:
                print("Error: a single time point; give several runs of the plate to join them")
                return 2
        else:
            table = assemble(folders, treated_at=args.treated_at)
        info = time_info(folders[0], args.treated_at)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Error: {exc}")
        return 2
    output = args.output
    if output is None:
        if len(folders) > 1:
            print("Error: give --output to join several runs")
            return 2
        output = folders[0] / "tables" / "time_course.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(
        f"Times: {info['time_source'] or 'separate acquisitions'}; started "
        f"{info['acquired_at'] or 'unknown'}; treated {info['treated_at'] or 'unknown'}"
        + (f" ({info['treatment_source']})" if info["treatment_source"] else "")
    )
    print(f"Time course: {output} ({table['well'].nunique()} wells, {len(table)} rows)")
    return 0


def _kinetics_analyse(args: argparse.Namespace, folders: list[Path]) -> int:
    from hcs_studio.kinetics.analysis import KineticsParams, analyse_runs
    from hcs_studio.processing.base import ModuleError

    output = args.output or folders[0] / "kinetics"
    try:
        params = KineticsParams(
            readout=args.readout,
            treated_at=args.treated_at,
            interval_h=args.interval_h,
            models=[m.strip() for m in args.models.split(",") if m.strip()],
        )
        result, files = analyse_runs(folders, params, output)
    except (OSError, ValueError, KeyError, ModuleError) as exc:
        print(f"Error: {exc}")
        return 2
    for note in result.notes:
        print(f"Note: {note}")
    print(
        f"Readout: {result.readout}; {len(result.fits)} wells; files in {output}: "
        + ", ".join(files)
    )
    return 0


def _cellpainting(args: argparse.Namespace) -> int:
    try:
        mapping = dict(item.split("=", 1) for item in args.map) if hasattr(args, "map") else {}
    except ValueError:
        print("Error: --map takes ROLE=CHANNEL")
        return 2
    try:
        if args.painting_command == "preset":
            from hcs_studio.cellpainting.preset import preset

            channels = [c.strip() for c in args.channels.split(",") if c.strip()]
            pipeline = preset(
                mapping,
                channels=channels,
                nucleus_diameter_um=args.nucleus_um,
                cell_distance_um=args.cell_um,
            )
            pipeline.save(args.output)
            print(f"Pipeline: {args.output}")
        elif args.painting_command == "single-cells":
            from hcs_studio.cellpainting.preset import channel_map
            from hcs_studio.cellpainting.single_cell import write_single_cells

            path, cells = write_single_cells(args.run, args.output, channel_map(mapping))
            print(f"Single cells: {cells} → {path}")
        elif args.painting_command == "profile":
            from hcs_studio.cellpainting.profiles import ProfilingOptions
            from hcs_studio.services.operations import cellpainting_profiles

            options = ProfilingOptions(
                normalize=args.normalize,
                reference=args.reference,
                per_plate=not args.across_plates,
                operations=[o for o in args.operations.split(",") if o],
                corr_threshold=args.corr_threshold,
                consensus_by=[c for c in args.consensus_by.split(",") if c],
            )
            result = cellpainting_profiles(
                _ctx(args), list(args.sources), args.output, options, args.platemap
            )
            log = result.log
            print(
                f"Profiles: {log['single_cell_qc']['kept']} cells → {log['wells']} wells → "
                f"{log['treatments']} consensus profiles; {log['selected_features']} of "
                f"{log['features']} features kept → {args.output}"
            )
            if args.evaluate:
                from hcs_studio.services.operations import cellpainting_evaluation

                evaluation = cellpainting_evaluation(_ctx(args), args.output, None, args.moa)
                _print_evaluation(evaluation.log, args.output)
        elif args.painting_command == "evaluate":
            from hcs_studio.cellpainting.evaluation import EvaluationOptions
            from hcs_studio.services.operations import cellpainting_evaluation

            evaluation_options = EvaluationOptions(
                treatment=args.treatment,
                threshold=args.threshold,
                null_size=args.null_size,
                same_plate_negatives=not args.any_plate_negatives,
                k=args.k,
                umap=not args.no_umap,
            )
            evaluation = cellpainting_evaluation(
                _ctx(args), args.folder, evaluation_options, args.moa
            )
            _print_evaluation(evaluation.log, args.folder)
        elif args.painting_command == "similar":
            import pandas as pd

            from hcs_studio.cellpainting.evaluation import similar

            matrix = pd.read_csv(args.folder / "similarity_cosine.csv", index_col=0)
            print(similar(matrix, args.n, args.to).to_string(index=False))
        else:
            from hcs_studio.cellpainting.cellprofiler import read_cellprofiler

            table = read_cellprofiler(args.source)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            table.to_parquet(args.output, index=False)
            print(f"Imported {len(table)} cells, {table.shape[1]} columns → {args.output}")
    except (OSError, ValueError, KeyError) as exc:
        print(f"Error: {exc}")
        return 2
    return 0


def _screen(args: argparse.Namespace) -> int:
    from hcs_studio.processing.base import ModuleError

    try:
        if args.screen_command == "score":
            from hcs_studio.screening.screen import ScreenConfig
            from hcs_studio.services.operations import run_screen

            result = run_screen(
                _ctx(args),
                list(args.folders),
                ScreenConfig.load(args.config),
                args.output,
                gallery=args.gallery,
            )
        elif args.screen_command == "report":
            from hcs_studio.screening.screen import ScreenConfig
            from hcs_studio.services.operations import run_screen

            summary = json.loads((args.output / "screen.json").read_text(encoding="utf-8"))
            result = run_screen(
                _ctx(args),
                [Path(f) for f in summary["runs"]],
                ScreenConfig.model_validate(summary["config"]),
                args.output,
                gallery=args.gallery,
            )
        elif args.screen_command == "cherrypick":
            return _screen_cherrypick(args)
        elif args.screen_command == "review":
            from hcs_studio.services.operations import review_hit

            result = review_hit(_ctx(args), args.output, args.compound, args.action)
        elif args.screen_command == "library":
            from hcs_studio.screening.library import read_library_map
            from hcs_studio.services.registry import import_library_map

            counts = import_library_map(_ctx(args), read_library_map(args.map), args.name)
            print(
                f"Library: {counts['plates']} plate(s), {counts['wells']} wells, "
                f"{counts['new_compounds']} new compound(s)"
            )
            return 0
        else:
            return _screen_layouts(args)
    except (OSError, ValueError, KeyError, ModuleError) as exc:
        print(f"Error: {exc}")
        return 2
    for note in result.notes:
        print(f"Note: {note}")
    print(
        f"Screen: {len(result.plates)} plates, {len(result.compounds)} compounds, "
        f"{len(result.hits)} hits → {args.output / 'hits.csv'}"
    )
    return 0


def _screen_cherrypick(args: argparse.Namespace) -> int:
    from hcs_studio.io.layout_io import load_layout
    from hcs_studio.screening.cherrypick import Backfill, ConfirmationDesign, PickListFormat
    from hcs_studio.screening.library import read_library_map
    from hcs_studio.services.operations import cherry_pick_hits

    backfill = None
    if args.backfill:
        plate, well = args.backfill.split(":", 1)
        backfill = Backfill(source_plate=plate, source_well=well)
    design = ConfirmationDesign(
        mode=args.mode,
        replicates=args.replicates,
        concentration_um=args.concentration_um,
        top_um=args.top_um,
        points=args.points,
        factor=args.factor,
        assay_volume_ul=args.assay_volume_ul,
        droplet_nl=args.droplet_nl,
        barcode_prefix=args.prefix,
        backfill=backfill,
    )
    result = cherry_pick_hits(
        _ctx(args),
        args.output,
        design,
        load_layout(args.template).layout,
        library=read_library_map(args.library) if args.library else None,
        fmt=PickListFormat.load(args.format) if args.format else None,
        register=args.register,
        outbox=args.outbox,
        scheduler_url=args.scheduler_url,
    )
    for note in result.notes:
        print(f"Note: {note}")
    compounds = int((result.picks["kind"] == "compound").sum())
    print(
        f"Cherry-pick: {compounds} transfers onto {len(result.layouts)} plate(s) "
        f"({', '.join(result.layouts)}) → {args.output / 'cherrypick'}"
    )
    return 0


def _screen_layouts(args: argparse.Namespace) -> int:
    from hcs_studio.io.layout_io import load_layout, save_layout
    from hcs_studio.io.layout_store import LayoutStore, plate_key
    from hcs_studio.screening.library import assay_layouts, read_library_map, read_transfers
    from hcs_studio.services.operations import save_layout_version

    library = read_library_map(args.map)
    template = load_layout(args.template).layout
    destinations = dict(item.split("=", 1) for item in args.dest)
    transfers = read_transfers(args.transfers) if args.transfers else None
    layouts = assay_layouts(
        library,
        template,
        args.rule,
        destinations=destinations,
        sources=[s for s in args.sources.split(",") if s] or None,
        dilution=args.dilution,
        transfers=transfers,
        assay_volume_ul=args.assay_volume_ul,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    ctx = _ctx(args)
    for barcode, layout in layouts.items():
        save_layout(layout, args.output / f"{barcode}.json")
        if args.register:
            saved = save_layout_version(
                ctx,
                LayoutStore(plate_key(barcode=barcode)),
                layout,
                f"screen layout from {args.map.name}",
            )
            print(f"Registered {barcode} as version {saved.version}")
    print(f"Layouts: {len(layouts)} → {args.output}")
    return 0


def _qc_review(args: argparse.Namespace) -> int:
    from hcs_studio.services.operations import review_qc

    try:
        result = review_qc(
            _ctx(args),
            args.folder,
            "field" if args.field is not None else "well",
            args.well,
            args.action,
            field=args.field,
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"Error: {exc}")
        return 2
    plate = result.plate
    verdict = "passed" if plate["passed"] else "FAILED"
    print(
        f"Fields excluded: {plate['excluded_fields']} of {plate['fields']}; "
        f"wells failed: {plate['failed_wells']}; plate {verdict}"
    )
    return 0


def _qc_train(args: argparse.Namespace) -> int:
    from hcs_studio.analysis.classifier import file_sha256
    from hcs_studio.qc.learned import train_field_qc
    from hcs_studio.services.operations import save_classifier

    try:
        classifier = train_field_qc(args.folders, unflagged_good=not args.reviewed_only)
        save_classifier(_ctx(args), classifier, args.output)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}")
        return 2
    accuracy = classifier.cv_balanced_accuracy
    shown = "n/a" if accuracy is None else f"{accuracy:.2f}"
    print(f"Examples: {classifier.n_examples}; balanced CV accuracy: {shown}")
    print("Add to the QC rules:")
    print(
        f"learned: {{model: '{args.output}', sha256: {file_sha256(args.output)}, threshold: 0.5}}"
    )
    return 0


def _analyze(args: argparse.Namespace) -> int:
    from hcs_studio.pipeline.schema import Pipeline
    from hcs_studio.processing.base import ModuleError
    from hcs_studio.services.operations import analyze_run

    try:
        pipeline = Pipeline.load(args.pipeline)
        if not pipeline.table_steps() and not pipeline.run_steps() and args.only:
            print("Error: the pipeline has no table steps or run steps")
            return 2
        record = analyze_run(
            _ctx(args),
            args.folder,
            pipeline,
            include_run_steps=not args.only,
            new_version=args.new_version,
        )
    except (OSError, ValueError, ModuleError, PermissionError) as exc:
        print(f"Error: {exc}")
        return 2
    if args.new_version:
        print(f"New version: {record['folder']}")
    print(f"Applied {len(record['steps'])} table step(s); tables and report updated")
    for step in record.get("run_steps", []):
        outcome = step["status"] + (f": {step['error']}" if step.get("error") else "")
        print(f"Run step {step['module']}: {outcome} ({step['seconds']} s)")
    failed = any(s["status"] != "completed" for s in record.get("run_steps", []))
    return 1 if failed else 0


def _train_classifier(args: argparse.Namespace) -> int:
    import pandas as pd

    from hcs_studio.analysis.classifier import train_from_run
    from hcs_studio.processing.base import ModuleError
    from hcs_studio.services.operations import save_classifier

    try:
        model = train_from_run(
            args.run,
            pd.read_csv(args.labels),
            args.object_type,
            algorithm=args.algorithm,
            normalisation=args.normalisation,
        )
    except (OSError, ValueError, ModuleError) as exc:
        print(f"Error: {exc}")
        return 2
    save_classifier(_ctx(args), model, args.output)
    examples = ", ".join(f"{k}: {v}" for k, v in model.n_examples.items())
    print(f"Trained on {examples} ({len(model.features)} features)")
    if model.cv_accuracy is not None:
        print(
            f"{model.cv_folds}-fold cross-validated accuracy {model.cv_accuracy:.3f} "
            f"(balanced {model.cv_balanced_accuracy:.3f})"
        )
    print(f"Classifier: {args.output}")
    return 0


def _figure(args: argparse.Namespace) -> int:
    from hcs_studio.compare.export import batch_compound_figures, regenerate
    from hcs_studio.compare.spec import Comparison, ExportSettings, PanelSpec

    try:
        if args.regenerate:
            out = regenerate(args.regenerate, args.output)
            print(f"Wrote {out}")
            return 0
        if args.per_compound and args.run:
            settings = ExportSettings(
                format=args.format,
                dpi=args.dpi,
                width_mm=args.width_mm,
                split_channels=args.split_channels,
                label_template="{well}",
            )
            base = Comparison(
                settings=settings,
                panels=[PanelSpec(run=str(args.run), well="A01", source=args.source)],
            )
            folder = args.output or args.run / "figures"
            files = batch_compound_figures(args.run, folder, base, args.compound or None)
            for f in files:
                print(f"Wrote {f}")
            return 0 if files else 1
    except (OSError, ValueError, KeyError) as exc:
        print(f"Error: {exc}")
        return 2
    print("Error: use --regenerate SIDECAR, or --run RUN --per-compound")
    return 2


def _conc(value: float, unit: str) -> str:
    from hcs_studio.model import units

    if not math.isfinite(value):
        return "—"
    try:
        return units.format_concentration(value, unit)
    except (ValueError, KeyError):
        return f"{value:.3g}"


def _pharm(args: argparse.Namespace) -> int:
    from hcs_studio.pharmacology.analysis import PharmOptions
    from hcs_studio.services.operations import run_pharmacology

    runs = [*args.runs, *args.extra_runs]
    if not runs:
        print("Error: give at least one run folder")
        return 2
    excluded = []
    for item in args.exclude:
        plate, _, well = item.rpartition(":")
        if not plate or not well:
            print(f"Error: --exclude needs PLATE:WELL, got {item!r}")
            return 2
        excluded.append((plate, well))
    options = PharmOptions(
        readouts=[r.strip() for r in args.readout.split(",") if r.strip()],
        normalization=args.normalize,
        models=tuple(m.strip() for m in args.models.split(",") if m.strip()),
        curve_model=args.curve,
        robust=args.robust,
        bootstrap=args.bootstrap,
        outlier_mad=args.outliers,
        z_threshold=args.z_threshold,
        doubling_time_h=args.doubling_time,
        assay_duration_h=args.duration,
        time_zero_runs=[str(p) for p in args.time_zero],
        excluded_wells=excluded,
        figures=not args.no_figures,
        surfaces=args.surfaces,
        pdf=not args.no_pdf,
    )
    try:
        result = run_pharmacology(_ctx(args), runs, options, args.output)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Error: {exc}")
        return 2
    for readout, rr in result.readouts.items():
        print(f"== {readout} ==")
        for row in rr.qc.itertuples():
            print(f"plate {row.plate}: Z′ {row.z_prime:.2f}, SSMD {row.ssmd:.1f}  {row.flag}")
        for c in rr.curves:
            f = c.fit
            print(
                f"{c.cell_line} · {c.compound}: EC50 {_conc(f.ec50, c.unit)}, Hill "
                f"{f.hill:.2f}, R² {f.r2:.3f}" + (f"  [{'; '.join(f.flags)}]" if f.flags else "")
            )
        for combo in rr.combinations:
            scores = ", ".join(
                f"{name} {res.mean:+.1f} ({res.label})" for name, res in combo.models.items()
            )
            print(f"{combo.matrix.name}: {scores}")
        for note in rr.notes:
            print(f"Note: {note}")
    print(f"Wrote {result.folder} in {result.seconds:.1f} s")
    return 0


def _profile(args: argparse.Namespace) -> int:
    from hcs_studio.analysis.profiling import ProfileOptions
    from hcs_studio.services.operations import run_profiling

    options = ProfileOptions(
        object_type=args.object_type,
        level=args.level,
        normalisation=args.normalisation,
        plate_correction=args.plate_correction,
        max_cells=args.max_cells,
        correlation_threshold=args.correlation,
        pca_components=args.pca_components,
        umap=not args.no_umap,
        n_neighbors=args.neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        seed=args.seed,
        dose_readouts=[r.strip() for r in args.dose_readouts.split(",") if r.strip()],
    )
    try:
        result = run_profiling(_ctx(args), args.runs, options, args.output)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Error: {exc}")
        return 2
    variance = result.variance["explained_variance_ratio"].head(3).tolist()
    print(
        f"{len(result.cells)} points, {len(result.wells)} wells, {len(result.features)} features "
        f"({len(result.dropped)} dropped); PC1-3 explain "
        + ", ".join(f"{100 * v:.0f} %" for v in variance)
    )
    for note in result.notes:
        print(f"Note: {note}")
    print(f"Profiles: {result.folder}")
    return 0


def _print_evaluation(log: dict[str, Any], folder: Path) -> None:
    line = (
        f"Evaluation: {log['active']} of {log['treatments']} treatments active; held-out "
        f"vehicle wells active: {log['holdout_active']}"
    )
    if log.get("moa_accuracy") is not None:
        line += f"; MoA kNN accuracy {log['moa_accuracy']:.0%}"
    print(f"{line} → {folder / 'evaluation.html'}")


def _acceptance(args: argparse.Namespace) -> int:
    from hcs_studio import acceptance
    from hcs_studio.processing.base import ModuleError

    try:
        if args.check == "counts":
            report = acceptance.check_counts(
                args.run, args.manual, args.object_type, args.tolerance
            )
        elif args.check == "coloc":
            report = acceptance.check_coloc(args.folders, args.tolerance)
        elif args.check == "run":
            report = acceptance.check_run(args.folder)
        elif args.check == "kill-resume":
            report = acceptance.check_kill_resume(
                args.pipeline,
                args.input,
                args.output,
                kill_after_wells=args.kill_after,
                workers=args.workers,
                layout=args.layout,
            )
        else:
            report = acceptance.check_beads(
                args.path, args.channel, psf_file=args.psf, iterations=args.iterations
            )
    except (OSError, ValueError, KeyError, RuntimeError, ModuleError) as exc:
        print(f"Error: {exc}")
        return 2
    print(report.text())
    target = args.report or Path(f"acceptance_{args.check.replace('-', '_')}.json")
    report.save(target)
    print(f"Report: {target}")
    return 0 if report.passed else 1


def _annotate(args: argparse.Namespace) -> int:
    from hcs_studio.io.layout_io import load_layout
    from hcs_studio.processing.base import ModuleError
    from hcs_studio.services.operations import annotate_run

    try:
        imported = load_layout(args.layout)
        for note in imported.notes:
            print(f"Note: {note}")
        record = annotate_run(_ctx(args), args.folder, imported.layout, args.layout)
    except (OSError, ValueError, KeyError, ModuleError) as exc:
        print(f"Error: {exc}")
        return 2
    print(
        f"Layout applied ({record['wells']} wells, sha256 {record['sha256'][:12]}…); "
        "tables and report rebuilt"
    )
    return 0


def _db(args: argparse.Namespace) -> int:
    import pandas as pd

    from hcs_studio.db import queries, session_scope
    from hcs_studio.db.engine import current_revision, database_url, get_engine

    def show(frame: pd.DataFrame) -> None:
        if frame.empty:
            print("(nothing found)")
        else:
            with pd.option_context("display.width", 200, "display.max_columns", 20):
                print(frame.to_string(index=False))
        if getattr(args, "export", None):
            print(f"Wrote {queries.export_table(frame, args.export)}")

    try:
        if args.db_command == "backup":
            from hcs_studio.db.backup import backup_registry

            for path in backup_registry(args.output):
                print(f"Backed up: {path}")
            return 0
        if args.db_command == "info":
            from sqlalchemy import func, select

            from hcs_studio.db.catalog import DoseResponseFit, Plate, Run, WellResult

            url = database_url()
            print(f"Registry: {url.split('@')[-1]}")
            print(f"Schema revision: {current_revision(get_engine(url))}")
            with session_scope() as session:
                for model in (Plate, Run, WellResult, DoseResponseFit):
                    n = session.scalar(select(func.count()).select_from(model))
                    print(f"  {model.__tablename__}: {n}")
            return 0
        if args.db_command == "uniformity":
            from hcs_studio.services.registry import save_uniformity_map

            stats = save_uniformity_map(
                _ctx(args),
                args.name,
                args.run,
                args.readout,
                degree=args.degree,
                instrument=args.instrument,
                assay=args.assay,
                replace=args.replace,
            )
            print(f"Uniformity map {args.name!r} saved")
            for key in ("edge_effect_pct", "cv_pct", "row_trend_pct", "col_trend_pct"):
                print(f"  {key}: {stats[f'before_{key}']:.2f} → {stats[f'after_{key}']:.2f}")
            return 0
        if args.db_command == "bridge":
            from hcs_studio.services.registry import write_bridging_report

            files = write_bridging_report(
                _ctx(args),
                args.instrument_a,
                args.instrument_b,
                args.readout,
                args.output,
                msr=args.msr,
            )
            summary = json.loads(files[1].read_text(encoding="utf-8"))
            print(
                f"{summary['plates']} plate(s), {summary['potency_pairs']} potency pair(s); "
                f"equivalent: {'yes' if summary['equivalent'] else 'no'}"
            )
            print(f"Report: {files[0]}")
            return 0 if summary["equivalent"] else 1
        if args.db_command == "event":
            from datetime import date

            from hcs_studio.services.registry import add_event_annotation

            add_event_annotation(
                _ctx(args),
                args.kind,
                date.fromisoformat(args.on),
                instrument=args.instrument,
                assay=args.assay,
                cell_line=args.cell_line,
                description=args.description,
            )
            print(f"Event recorded: {args.kind} on {args.on}")
            return 0
        if args.db_command == "scan":
            from hcs_studio.services.registry import scan_folders

            report = scan_folders(_ctx(args), args.paths, raw=args.raw, force=args.force)
            print(report.text())
            return 1 if report.errors else 0
        with session_scope() as session:
            if args.db_command == "search":
                query = queries.PlateQuery(
                    text=args.text,
                    barcode=args.barcode,
                    experiment=args.experiment,
                    assay=args.assay,
                    cell_line=args.cell_line,
                    compound=args.compound,
                    instrument=args.instrument,
                    run_status=args.status,
                )
                show(queries.search_plates(session, query).drop(columns=["plate_id"]))
            elif args.db_command == "potency":
                show(queries.potency_table(session, args.compound, args.cell_line, args.readout))
            elif args.db_command == "ic50s":
                show(queries.ic50s(session, args.compound, args.cell_line, args.readout))
            elif args.db_command in ("chart", "msr"):
                return _db_monitoring(args, session, show)
            elif args.db_command in ("normalize", "maps", "batch", "concordance"):
                return _db_normalization(args, session, show)
            elif args.db_command == "plate":
                found = queries.search_plates(session, queries.PlateQuery(barcode=args.barcode))
                if found.empty:
                    print(f"No plate {args.barcode!r} in the registry")
                    return 2
                page = queries.plate_page(session, int(found.plate_id.iloc[0]))
                info = page["plate"]
                print(
                    f"{info['barcode'] or info['name']}: {info['plate_type']} wells, "
                    f"assay {info['assay'] or '-'}, experiment {info['experiment'] or '-'}, "
                    f"cell lines {', '.join(info['cell_lines']) or '-'}"
                )
                print("Acquisitions:")
                show(pd.DataFrame(page["acquisitions"]))
                print("Runs:")
                show(pd.DataFrame(page["runs"]))
                if page["qc"]:
                    print("Plate QC:")
                    show(pd.DataFrame(page["qc"]))
    except (OSError, ValueError, KeyError) as exc:
        print(f"Error: {exc}")
        return 2
    return 0


def _db_monitoring(args: argparse.Namespace, session: Any, show: Any) -> int:
    from hcs_studio.db import monitoring as dbm
    from hcs_studio.monitoring.charts import Rule

    if args.db_command == "msr":
        history = dbm.msr_history(
            session,
            args.compound,
            cell_line=args.cell_line,
            readout=args.readout,
            window=args.window,
        )
        print(
            f"{args.compound}: {history['n_runs']} runs, MSR {history['msr']:.3g} "
            "(Eastwood et al. 2006: 10^(2·√2·s), s = SD of log10 EC50 between runs)"
        )
        show(history["rolling"])
        return 0
    rules = [Rule.parse(r) for r in args.rules.split(",") if r.strip()]
    chart = dbm.chart(
        session,
        args.metric,
        readout=args.readout,
        compound=args.compound,
        cell_line=args.cell_line,
        assay=args.assay,
        instrument=args.instrument,
        experiment=args.experiment,
        rules=rules,
        baseline_n=args.baseline,
    )
    low, high = chart.limits(3)
    print(
        f"{chart.name}: centre {chart.centre:.4g}, SD {chart.sd:.4g} "
        f"(from {chart.baseline_n} runs), ±3 SD {low:.4g} … {high:.4g}"
    )
    for note in chart.notes:
        print(f"Note: {note}")
    for v in chart.violations:
        print(f"  RULE {v.rule}: run {chart.points['run'].iat[v.index]} ({v.value:.4g})")
    for e in chart.events.to_dict("records"):
        print(f"  event {e['on']}: {e['kind']} {e['description']}")
    flagged = set(chart.flagged)
    table = chart.points.assign(flag=[i in flagged for i in range(len(chart.points))])
    show(table)
    return 1 if chart.violations else 0


def _db_normalization(args: argparse.Namespace, session: Any, show: Any) -> int:
    import pandas as pd

    from hcs_studio.db import normalization as dbn
    from hcs_studio.normalization.plate import PositionalCorrectionBlocked

    if args.db_command == "maps":
        show(dbn.uniformity_maps(session))
        return 0
    if args.db_command == "normalize":
        try:
            data, result = dbn.normalize_run(session, args.run, args.readout, args.method, args.map)
        except PositionalCorrectionBlocked as exc:
            print(f"Blocked: {exc}")
            return 1
        print(f"{data.barcode} ({data.run}): {args.readout}, {args.method}")
        for note in result.notes:
            print(f"Note: {note}")
        show(result.frame)
        return 0
    if args.db_command == "batch":
        effects = dbn.day_effects(
            session, args.readout, runs=args.runs or None, experiment=args.experiment
        )
        print("Share of the vehicle wells' variance explained by day and by plate:")
        print(effects.summary.to_string(index=False))
        for note in effects.notes:
            print(f"Note: {note}")
        show(effects.per_plate)
        return 0
    a, b, agreement = dbn.concordance(
        session, args.run_a, args.run_b, args.readout, r_min=args.r_min, fold_max=args.fold_max
    )
    summary = agreement.summary()
    print(f"{a.barcode} vs {b.barcode}, {args.readout}: {summary['n']} sample wells")
    print(f"  Pearson r {agreement.pearson:.3f}, Spearman ρ {agreement.spearman:.3f}")
    low, high = agreement.loa_low, agreement.loa_high
    print(f"  Bland–Altman bias {agreement.bias:.4g}, 95 % limits {low:.4g} … {high:.4g}")
    print("  " + ("DISCORDANT: " + "; ".join(agreement.flags) if agreement.flags else "concordant"))
    show(agreement.ic50 if not agreement.ic50.empty else pd.DataFrame(agreement.conditions))
    return 1 if agreement.discordant else 0


def _ilastik(args: argparse.Namespace) -> int:
    """Exit code 0: ilastik usable; 1: found but blocked; 2: not found."""
    from dataclasses import replace

    from hcs_studio.segmentation.ilastik_backend.settings import IlastikSettings, detect

    if args.info or args.preset:
        return _ilastik_project(args)
    settings = IlastikSettings.load()
    changes: dict[str, object] = {}
    if args.set_ilastik:
        changes["ilastik"] = str(args.set_ilastik.expanduser().absolute())
    if args.set_fiji:
        changes["fiji"] = str(args.set_fiji.expanduser().absolute())
    for name in ("threads", "ram_mb", "max_processes"):
        if getattr(args, name) is not None:
            changes[name] = getattr(args, name)
    try:
        settings = replace(settings, **changes)  # type: ignore[arg-type]
        settings.validate()
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2
    found = detect(settings)
    print(found.text())
    if args.save:
        if found.ilastik and not settings.ilastik:
            settings = replace(settings, ilastik=found.ilastik)
        if found.fiji and not settings.fiji:
            settings = replace(settings, fiji=found.fiji)
        from hcs_studio.services.operations import save_ilastik_settings

        print(f"Saved {save_ilastik_settings(_ctx(args), settings)}")
    if not found.ilastik:
        return 2
    return 0 if found.ilastik_ok else 1


def _ilastik_project(args: argparse.Namespace) -> int:
    """``--info``: the project's workflow, classes, training axes; ``--preset``: a pipeline."""
    import json

    from hcs_studio.segmentation.ilastik_backend.project import read_project

    path = args.info or args.preset
    try:
        info = read_project(path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    if args.info:
        print(json.dumps(info.to_dict(), indent=2))
        return 0
    if not args.output:
        print("ERROR: give --output for the pipeline file")
        return 2
    from hcs_studio.segmentation.ilastik_backend.presets import multicut_preset, pixel_preset

    try:
        if info.workflow == "multicut" and args.boundary:
            pipeline = multicut_preset(args.boundary, path)
        else:
            pipeline = pixel_preset(
                path, info, object_class=args.object_class, channels=args.channels
            )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2
    import yaml

    args.output.write_text(yaml.safe_dump(pipeline.to_dict(), sort_keys=False), encoding="utf-8")
    print(f"Pipeline written to {args.output}")
    return 0


def _layout(args: argparse.Namespace) -> int:
    from hcs_studio.io.layout_io import (
        load_layout,
        save_layout,
        write_layout_figure,
        write_platemap_xlsx,
    )
    from hcs_studio.model.layout_tools import validate

    try:
        imported = load_layout(args.path)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Error: {exc}")
        return 2
    layout = imported.layout
    print(
        f"{len(layout.wells)} annotated wells on a {layout.rows}×{layout.cols} plate; "
        f"cell lines: {', '.join(layout.cell_lines()) or '-'}; compounds: "
        f"{', '.join(layout.compounds()) or '-'}"
    )
    for note in imported.notes:
        print(f"Note: {note}")
    code = 0
    if args.validate:
        issues = validate(layout)
        for issue in issues:
            print(f"{issue.level.upper()}: {issue.text()}")
        print("No problems found." if not issues else f"{len(issues)} problem(s)")
        code = 1 if any(i.level == "error" for i in issues) else 0
    if args.export:
        suffix = args.export.suffix.lower()
        if suffix in (".png", ".pdf", ".svg"):
            write_layout_figure(layout, args.export, args.layer)
        elif args.plate_map:
            write_platemap_xlsx(layout, args.export)
        else:
            save_layout(layout, args.export)
        print(f"Wrote {args.export}")
    return code
