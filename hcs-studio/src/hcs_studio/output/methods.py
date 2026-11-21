"""Methods text written from a run manifest: acquisition, every pipeline step with its
parameters, measurements, analyses, pharmacology and software — ready to adapt for a
paper. Only facts recorded in the run are stated."""

from __future__ import annotations

from typing import Any

#: What each module does, in one clause (the step's parameters are appended).
DESCRIPTIONS = {
    "flatfield": "illumination (flat-field) correction with a profile estimated from fields across the plate",
    "af_reference": "autofluorescence subtraction using unstained reference wells",
    "af_channel": "autofluorescence subtraction using a dedicated autofluorescence channel (bleed-through coefficients estimated by robust regression)",
    "unmix": "linear spectral unmixing with reference spectra",
    "background": "background removal",
    "denoise": "denoising",
    "deconvolution": "deconvolution (Richardson–Lucy with total-variation regularisation unless stated) with a PSF computed from the optical metadata",
    "register_channels": "registration of channels to correct chromatic shifts",
    "projection": "Z projection",
    "contrast": "contrast adjustment for display and segmentation only",
    "clahe": "contrast-limited adaptive histogram equalisation for display and segmentation only",
    "sharpen": "sharpening for display and segmentation only",
    "clarity": "detail enhancement for display and segmentation only",
    "segment_cellpose": "segmentation with Cellpose",
    "segment_classical": "segmentation by thresholding and watershed separation",
    "segment_sam3": "segmentation with SAM 3",
    "expand_cells": "cell regions grown from the nuclei",
    "detect_spots": "spot detection (Laplacian/difference of Gaussians)",
    "filter_objects": "object filtering",
    "measure_objects": "per-object morphology, intensity and texture measurements on the linear (corrected) images",
    "colocalization": "colocalization analysis (Pearson, Manders with Costes thresholds)",
    "qc": "field quality control (focus, saturation) and counts",
    "stitch_well": "stitching of the fields of each well (stage positions refined by phase correlation)",
    "gate_objects": "gating of objects",
    "detect_foci": "detection of DNA-damage foci inside nuclei (Laplacian-of-Gaussian band "
    "pass, threshold = median + k robust SD of the vehicle nuclei's filter response, split at "
    "local maxima, size filter; 3D with Z stacks)",
    "foci": "foci per nucleus and well (pan-nuclear nuclei — mean intensity ≥ a multiple of the "
    "vehicle median — reported apart)",
    "cell_cycle": "cell-cycle analysis from integrated nuclear DNA stain (vehicle G1 peak scaled "
    "to 2N; Dean–Jett–Fox-style fit of G1, S and G2/M; sub-G1 and > 4N by DNA content; EdU and "
    "pH3 thresholds from the vehicle wells)",
    "object_qc": "object-level QC (robust outliers in size, shape and intensity left out of "
    "feature statistics)",
    "classify_objects": "classification of objects with a trained classifier",
    "phenotypic_clusters": "phenotypic clustering of objects",
    "spatial_clusters": "spatial statistics (colonies, nearest neighbours, Ripley's H)",
    "pharmacology": "dose–response and combination analysis",
    "compound_figures": "image montages per compound",
    "experiment_report": "this report",
}


def _value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, list):
        return ", ".join(_value(v) for v in value) if value else "none"
    if isinstance(value, dict):
        return "; ".join(f"{k} {_value(v)}" for k, v in value.items())
    return str(value)


def _params(params: dict[str, Any]) -> str:
    shown = [
        f"{k.replace('_', ' ')} {_value(v)}"
        for k, v in params.items()
        if v not in (None, "", [], {})
    ]
    return "; ".join(shown)


def acquisition(manifest: dict[str, Any]) -> str:
    plate = manifest.get("plate", {})
    meta = plate.get("metadata", {})
    channels = meta.get("channels", [])
    parts = [
        f"Images of plate {plate.get('name', '')} ({plate.get('vendor', 'unknown vendor')},"
        f" {plate.get('rows', '?')} × {plate.get('cols', '?')} wells)"
    ]
    optics = []
    if meta.get("objective_name") or meta.get("magnification"):
        optics.append(
            f"objective {meta.get('objective_name') or ''} {meta.get('magnification') or ''}×".replace(
                "  ", " "
            )
        )
    if meta.get("objective_na"):
        optics.append(f"NA {meta['objective_na']:g}")
    if meta.get("pixel_size_um"):
        optics.append(f"pixel size {meta['pixel_size_um']:.4g} µm")
    if meta.get("z_step_um"):
        optics.append(f"Z step {meta['z_step_um']:.4g} µm")
    if meta.get("modality") and meta["modality"] != "unknown":
        optics.append(meta["modality"])
    if optics:
        parts.append("were acquired with " + ", ".join(optics))
    if channels:
        names = []
        for c in channels:
            wl = "/".join(str(int(v)) for v in (c.get("excitation_nm"), c.get("emission_nm")) if v)
            names.append(f"{c.get('name', '')}" + (f" ({wl} nm)" if wl else ""))
        parts.append(f"in {len(channels)} channel(s): {', '.join(names)}")
    overrides = plate.get("metadata_overrides") or {}
    text = " ".join(parts) + "."
    if overrides:
        text += f" Metadata corrected by the user: {', '.join(sorted(overrides))}."
    return text


def pipeline_text(manifest: dict[str, Any]) -> list[str]:
    from hcs_studio.processing.base import MODULES, load_builtin_modules

    load_builtin_modules()
    pipeline = manifest.get("pipeline", {})
    steps = [s for s in pipeline.get("steps", []) if s.get("enabled", True)]
    out = []
    for n, step in enumerate(steps, start=1):
        name = step["module"]
        title = MODULES.get(name).title if name in MODULES else name
        what = DESCRIPTIONS.get(name, title)
        params = _params(step.get("params", {}))
        out.append(f"{n}. {title}: {what}" + (f" ({params})." if params else "."))
    return out


def software(manifest: dict[str, Any]) -> str:
    sw = manifest.get("software", {})
    commit = sw.get("git_commit")
    text = (
        f"Images were processed with HCS Studio {sw.get('version', '?')}"
        + (f" (git {str(commit)[:10]})" if commit else "")
        + f", Python {sw.get('python', '?')}, NumPy {sw.get('numpy', '?')}, on {sw.get('platform', '?')}."
    )
    models = manifest.get("models") or []
    if models:
        text += (
            " Models: "
            + "; ".join(
                f"{m.get('name', '')} (checkpoint SHA-256 {str(m.get('sha256', ''))[:12]}…)"
                for m in models
            )
            + "."
        )
    text += (
        " Intensities were measured on linear images (after correction, before any "
        "display-only enhancement). All parameters, input files"
        + (" with SHA-256 checksums" if manifest.get("input_checksums") == "sha256" else "")
        + " and plate-level estimates are recorded in run_manifest.json."
    )
    return text


def pharmacology_text(info: dict[str, Any] | None) -> str:
    if not info:
        return ""
    o = info.get("options", {})
    labels = {
        "percent_of_control": "percent of vehicle control",
        "percent_inhibition": "percent inhibition between vehicle (0 %) and positive controls (100 %)",
        "gr": "growth-rate inhibition (GR; Hafner et al. 2016)",
    }
    model = {
        "4pl": "a four-parameter logistic",
        "5pl": "a five-parameter (asymmetric) logistic",
        "3pl_bottom": "a three-parameter logistic (bottom fixed)",
        "3pl_top": "a three-parameter logistic (top fixed)",
    }.get(o.get("curve_model"), "a logistic")
    text = (
        f"Per-well readouts ({', '.join(o.get('readouts', []))}) were normalized per plate and "
        f"cell line to {labels.get(o.get('normalization'), o.get('normalization'))}. "
        f"Plate quality was assessed with Z′ (threshold {o.get('z_threshold', 0.5)}), SSMD and "
        f"control CVs. Dose–response curves were fitted with {model} on log10 concentration "
        f"({'robust soft-L1' if o.get('robust') else 'least-squares'} fitting; 95 % confidence "
        f"intervals from {'a bootstrap' if o.get('bootstrap') else 'the covariance matrix'})."
    )
    if o.get("outlier_mad"):
        text += f" Wells beyond median ± {o['outlier_mad']:g} MAD of their replicate group were excluded."
    if o.get("excluded_wells"):
        text += f" {len(o['excluded_wells'])} well(s) were excluded by hand."
    models = o.get("models", [])
    if models:
        text += (
            f" Combination matrices were scored with {', '.join(m.upper() for m in models)} "
            "(Bliss independence, highest single agent, Loewe additivity with combination "
            "indices, and ZIP after Yadav et al. 2015), reporting the mean synergy over the "
            "combination cells and over the most synergistic 3 × 3 window, with a one-sample "
            "t test across replicate matrices."
        )
    return text


def methods_paragraphs(
    manifest: dict[str, Any], pharmacology: dict[str, Any] | None = None
) -> list[str]:
    paragraphs = [acquisition(manifest)]
    qc = manifest.get("image_qc")
    if qc:
        rules = qc.get("rules") or {}
        paragraphs.append(
            "Before processing, every field was checked on the raw images (focus, saturation, "
            "illumination, bubbles, out-of-well imaging, debris and other artifacts) with QC "
            f"rules '{rules.get('name', '')}' version {rules.get('version', '')}; "
            f"{qc.get('excluded_fields', 0)} of {qc.get('fields', 0)} fields were excluded and "
            f"{qc.get('failed_wells', 0)} wells failed QC."
        )
    paragraphs.append("Image processing and analysis steps, in order:")
    paragraphs += pipeline_text(manifest)
    analyses = manifest.get("analyses") or []
    if analyses:
        later = [s["module"] for a in analyses for s in a.get("steps", [])]
        if later:
            paragraphs.append("Applied after the run: " + ", ".join(dict.fromkeys(later)) + ".")
    pharm = pharmacology_text(pharmacology)
    if pharm:
        paragraphs.append(pharm)
    paragraphs.append(software(manifest))
    return paragraphs
