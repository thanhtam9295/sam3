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
    "segment_model": "segmentation with a registered model (pipeline steps stored in the model registry)",
    "model_qa": "fields queued for human spot checks of the segmentation and a drift check against the model's training data",
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
    "attenuation": "correction of depth-dependent intensity loss (the 90th percentile of the "
    "foreground above each plane's background, fitted as an exponential decay with depth by a "
    "Theil–Sen line through its logarithm; the signal above background multiplied by the inverse, "
    "capped at 10×)",
    "detect_spheroids": "spheroid detection (brightfield: local standard deviation of a "
    "difference of Gaussians, which ignores smooth well-edge and meniscus shading, then the outline "
    "at half height between the spheroid interior and its local background; fluorescence: Otsu)",
    "measure_spheroids": "spheroid size, shape and volume (sphere from the equivalent diameter, "
    "π/6·d³; ellipsoid π/6·L·W²), brightfield darkness, outline blur and position; radial "
    "intensity profiles in concentric shells; necrotic core outlined at half height between the "
    "rim and the centre; live/dead stain areas above background median + 5 robust SD; invasion "
    "as protrusions beyond a morphological opening of the spheroid (count, reach, skeleton)",
    "spheroid": "one spheroid per well (the largest object not touching the image border; or, in "
    "organoid mode, every organoid: count, size distribution, density and forming efficiency), with "
    "QC flags for no spheroid, several spheroids, touching the border, off-centre, out of focus, "
    "debris and irregular outlines",
    "mitotic_errors": "micronuclei (small DNA objects up to 0.2 × the median nuclear area within "
    "5 µm of a nucleus but not touching it, assigned to the nearest nucleus) and nuclei per cell "
    "(each nucleus linked to the cell segmented on a cell stain that it overlaps most)",
    "proliferation": "proliferation markers (% positive nuclei for Ki67, EdU/BrdU and pH3 with "
    "thresholds from secondary-only wells, median + 5 robust SD of log intensity, or from the "
    "vehicle nuclei: equal-likelihood point of two log-normal populations, or median + 5 robust "
    "SD for pH3), micronucleus frequency and multinucleated cells",
    "register_timepoints": "drift between time points corrected per field (phase "
    "cross-correlation of consecutive time points, sub-pixel, accumulated; jumps larger than "
    "max_shift_um left uncorrected; each field cropped to the area seen at every time point)",
    "confluence": "confluence (% of the field covered by cells: local texture of the image minus "
    "its median filter, thresholded at Otsu's level of fields pooled across the plate and time "
    "points, edges corrected for the window; or an intensity threshold, or a label set)",
    "tracking": "cell tracking (nuclei linked between time points by optimal assignment on the "
    "distance to their predicted positions, with track ends and starts allowed at the cost of a "
    "speed gate, Jaqaman et al. 2008; divisions from daughters smaller than the mother next to it; "
    "gaps of up to max_gap time points closed)",
    "kinetics": "kinetic analysis of the time course (growth curves fitted on the log scale: "
    "exponential, lag-exponential or logistic chosen by AICc, doubling time ln 2 / growth rate; "
    "AUC; IC50 at each time point from % of the same-time vehicle median; GR values from "
    "treatment with each well's own pre-treatment value as time zero and over intervals, "
    "Hafner et al. 2016; kinetic QC of acquisition gaps, focus drift, edge evaporation and "
    "vehicle growth against earlier runs)",
    "cytoplasm": "a cytoplasm compartment per nucleus (a ring between gap_um and gap_um + "
    "width_um from the nuclear edge, assigned to the nearest nucleus and, with a cell segmentation, "
    "kept inside the cell; or the cell minus its nucleus)",
    "translocation": "nuclear translocation (nuclear/cytoplasmic mean-intensity ratio per cell; "
    "cells responding above median + 3 robust SD of log N/C of the unstimulated control cells; "
    "% responding, median N/C and activation between unstimulated and stimulated controls)",
    "apoptosis": "apoptosis and cytotoxicity per nucleus (apoptosis-marker and dead-cell-dye "
    "thresholds at median + 5 robust SD of log intensity of the vehicle nuclei; condensed nuclei "
    "by size and DNA brightness relative to vehicle; viable counts, and GR from a time-zero count "
    "to separate cytostatic from cytotoxic responses)",
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
