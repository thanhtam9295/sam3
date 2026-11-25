"""Regenerate docs/processing.md from the module definitions.

Run from the hcs-studio folder: python scripts/gen_processing_docs.py
"""

from __future__ import annotations

from pathlib import Path

from hcs_studio.analysis.run_steps import RunStep
from hcs_studio.analysis.table_steps import TableModule
from hcs_studio.processing.base import MODULES, load_builtin_modules

NOTES = {
    "flatfield": "Estimates the illumination profile from many fields of the plate: each field "
    "is divided by its mean, the per-pixel median across fields removes the cells, and a "
    "smooth surface is fitted: a 2D polynomial (default, degree 4; accurate up to the image "
    "border) or a wide Gaussian (for illumination a polynomial cannot follow; it flattens the "
    "corners by a few percent). The median assumes most pixels are background: stains that "
    "cover much of each field (e.g. dense cytoplasm) bias the profile; where almost no "
    "background is left, the polynomial is not used and the channel falls back to smoothing "
    "(recorded as gaussian_fallback_channels in the manifest). The profile is "
    "normalised to mean 1, so the mean "
    "intensity of a field is preserved. This is a median-based estimate, not the BaSiC "
    "algorithm. It needs enough fields with varied content; it fails rather than guess when "
    "fewer than `min_fields` are available.",
    "af_reference": "Subtracts the autofluorescence measured in unstained or vehicle-only wells, "
    "then clips at 0. Place it after flat-field correction.",
    "af_channel": "Subtracts `k × AF channel` from each target channel, then clips at 0. `k` is "
    "estimated by robust regression (Tukey biweight), which ignores pixels carrying real "
    "signal as long as background pixels are the majority.",
    "background": "Removes background per plane, typically after projection: top-hat, rolling "
    "ball or sliding paraboloid (estimated on a 3 × 3 mean), Gaussian or median background "
    "(subtracted or divided), a polynomial or spline surface fitted to background pixels, a "
    "constant (percentile or blank wells), or BaSiC-style flat- and dark-field estimated "
    "across the plate. Negatives are clipped; output stays linear. Each field reports the "
    "change inside objects and in the background, and warns when the radius is under 2× the "
    "typical object diameter.",
    "unmix": "Solves the linear mixing model by least squares and clips negative values. The "
    "output has one channel per fluorophore.",
    "denoise": "Smoothing filters. Output is still treated as measurement-grade, but median and "
    "non-local means are not strictly linear; prefer Gaussian or none for quantitative work.",
    "contrast": "Rescales each channel to 0-1 between two percentiles and applies gamma.",
    "clahe": "Contrast-limited adaptive histogram equalisation, per plane (2D).",
    "sharpen": "Unsharp mask or difference-of-Gaussians band boost.",
    "clarity": "Mild Gaussian denoise, then an unsharp mask that does not amplify detail at the "
    "noise level (limits ringing and noise after deconvolution), then CLAHE.",
    "projection": "Collapses Z: maximum (default), mean, sum, minimum, standard deviation or "
    "extended depth of field (`variance`: sharpest plane per pixel; `pyramid`: multi-scale "
    "fusion). Planes: all, a range, or `auto` (the planes around the best focus by normalized "
    "variance or Brenner, per channel or from `focus_channel`). The planes used are recorded "
    "per field in `well_summary.json`.",
    "deconvolution": "Richardson-Lucy with total-variation regularisation (Dey et al. 2006) and "
    "Biggs-Andrews acceleration, or a one-step Wiener filter for quick previews. 3D for Z "
    "stacks (2D per plane with `force_2d`). The image is mirror-padded by the PSF half-size, "
    "a constant background is estimated (or given) and excluded from the result, and the "
    "result is non-negative. The PSF is computed from the metadata with `psfmodels` "
    "(vectorial or Gibson-Lanni; pinhole-aware for confocal) or loaded from a file saved by "
    "Tools → PSF tool, resampled to the image voxel size. Large images are split into "
    "overlapping Y/X tiles, blended linearly, whenever they would exceed the memory budget. "
    "Iterations, residual, runtime, tiles and the PSF used are recorded per channel in each "
    "well's `well_summary.json`. Runs on a CUDA GPU through PyTorch when one is available, "
    "otherwise on the CPU. **Validate against your reference software (e.g. Huygens) on bead "
    "and cell images before using it for quantitative claims** — see "
    "[validation.md](validation.md).",
    "segment_classical": "Smooths the channel, thresholds it (Otsu, Li or triangle; `local` "
    "subtracts a smooth background first, for uneven illumination), fills holes and separates "
    "touching objects by watershed on the distance transform. Fast and transparent; best for "
    "well-separated, evenly stained nuclei and as a fallback when no model is available.",
    "segment_model": "Runs a segmentation model registered as pipeline steps (for example a "
    "classical segmentation tuned on a ground-truth set, `hcs-studio models train --kind steps`) "
    "by its registry reference `registry:NAME@VERSION`. The stored steps run exactly as "
    "registered, so a pinned version always gives the same masks; the run manifest records the "
    "version and its SHA-256. See [models.md](models.md).",
    "model_qa": "Run step: after the run, queues a random sample of fields for a human spot "
    "check (accept or flag, `hcs-studio models spot-check`) and compares the run's objects (area, "
    "intensity, shape, objects per field) with the training data of the registered segmentation "
    "model it used; features more than `drift_threshold` IQR away are reported as drift in "
    "`qa/model_qa.json` and the run log. See [models.md](models.md).",
    "segment_cellpose": "Cellpose 4 (`cpsam_v2` is the default Cellpose-SAM model). Give one "
    "channel, or two (e.g. cytoplasm plus nuclei) for whole cells. The diameter is optional. "
    "Z stacks are segmented in 3D (`do_3d`), plane by plane with stitching, or plane by plane. "
    "A trained model file or a registered model (`registry:NAME@VERSION`) can be used in place "
    "of a built-in name. Uses the GPU when available; "
    "on the CPU expect about 15 s per 512 × 512 field.",
    "segment_sam3": 'Segments every instance of a text concept ("nucleus", "spheroid") with '
    "SAM 3, plane by plane, from 1-3 channels shown as a grey or colour image. Runs in a "
    "separate SAM 3 environment (Tools → SAM 3 settings). Best for organoids, spheroids and "
    "unusual objects; use Cellpose for high-throughput cell and nucleus segmentation.",
    "expand_cells": "Grows each nucleus outward by up to `distance_um`, without overlap, so "
    "each cell keeps its nucleus' ID (cell 7 contains nucleus 7). Optionally limited to pixels "
    "where a cytoplasm channel is above a threshold.",
    "detect_spots": "Finds sub-cellular spots with a Laplacian-of-Gaussian or "
    "difference-of-Gaussians filter tuned to the spot size, thresholded at a multiple of the "
    "robust noise level. Each spot is linked to the cell under its centre; spots outside cells "
    "are dropped unless `only_inside_parent` is off.",
    "filter_objects": "Removes objects touching the border, outside a size range, below a "
    "solidity, or outside an intensity range (intensity gates need linear data, so put the "
    "filter before display-only steps). Objects of child label sets lose their parent or are "
    "removed with it. Removed IDs are not renumbered.",
    "measure_objects": "Measures every object of the chosen label sets on the linear image "
    "(never on contrast/CLAHE/clarity output): morphology in µm, intensity per channel, "
    "radial distribution, Haralick texture, granularity, optional Zernike moments, counts of "
    "child objects (e.g. spots per cell), nuclear/cytoplasmic ratios and neighbours. Writes "
    "`objects.parquet` per well. Column reference: [measurements.md](measurements.md).",
    "colocalization": "Pearson, Spearman, Li's ICQ, Manders M1/M2, thresholded tM1/tM2 with "
    "Costes automatic thresholds (or manual ones), overlap coefficients and, optionally, the "
    "Costes randomization p-value, per object or for the whole field, after background "
    "subtraction. Warns about saturated pixels and chromatic shift. Optional object-based "
    "colocalization: the fraction of objects in one label set near objects of another. "
    "Writes `coloc.csv` per well.",
    "qc": "Per field: focus score (edge sharpness relative to contrast, independent of the "
    "number of cells), fraction of saturated "
    "pixels, imaged area and object counts. After the run, fields are flagged as out of focus "
    "or saturated relative to the plate, wells with low counts are flagged, and counts and "
    "densities per well leave out the excluded fields. Writes `tables/qc_flags.csv`.",
    "stitch_well": "Places the fields of each well in one coordinate system from their stage "
    "positions. When fields overlap, the way stage axes map to image axes is found from the "
    "overlaps (or set), and positions are refined by phase correlation of each overlap "
    "(sub-pixel, Hanning-windowed) and solved together by least squares; overlaps that "
    "correlate poorly or would move a field by more than `max_correction_fraction` are "
    "ignored. Objects get `well_x_um`, `well_y_um` and `in_overlap_duplicate` (the copy seen "
    "in a field whose centre is farther away), so each object is counted once; well counts "
    "and densities use the area the fields cover together. Writes `previews/mosaic.png` and "
    "the field positions to `well_summary.json`. Images are not resampled or fused.",
    "gate_objects": "Marks objects of one type whose features lie inside thresholds (min "
    "and/or max per feature, all or any rule), e.g. drawn on a histogram or scatter plot in "
    "the Classify panel. Writes `gate_<name>` per object and the count and fraction per well.",
    "detect_foci": "Assay (Phase 20b): DNA-damage foci inside nuclei. The foci channel is band-"
    "pass filtered (Laplacian of Gaussian × σ², or a white top-hat); pixels above the threshold "
    "inside nuclei are split at local maxima (seeded watershed) and kept when their size is "
    "plausible. The threshold is calibrated on the vehicle wells (median + threshold_sd × robust "
    "SD of the response over the vehicle nuclei) unless fixed. With Z planes the detection is 3D. "
    "Writes the label set `foci` (parents: nuclei); measure_objects then gives `foci_count` per "
    "nucleus. See the Assays guide.",
    "foci": "Assay (Phase 20b): per well the mean foci per nucleus, % nuclei with ≥ "
    "positive_min_foci foci, focus area and intensity, and pan-nuclear nuclei (mean intensity ≥ "
    "pan_factor × the vehicle median) reported apart; by cell-cycle phase when the cell-cycle "
    "step ran first. Writes `dd_` columns and `assays/foci/`. Warns about Z projections.",
    "attenuation": "Confocal Z stacks (Phase 21c): depth-dependent intensity loss. Per plane, the "
    "`percentile` of the foreground above the plane's background, over planes whose foreground "
    "covers ≥ min_area_fraction of the largest; exponential: robust fit of I₀·exp(−z/λ), "
    "per_plane: each plane to the first. Only the signal above background is scaled; gains capped "
    "at max_gain (warning). Reports λ and the factors per channel. Put it first.",
    "detect_spheroids": "Spheroids (Phase 21), Z stacks with method bright: 3D detection (Otsu, "
    "surface at half height between interior and local background, anisotropic distances). "
    "2D: brightfield = textured objects (local SD of a "
    "difference of Gaussians at texture_um; smooth well-edge and meniscus shading cancels out; "
    "objects not textured inside are dropped), outline refined at half height between the "
    "interior and the local background (blur does not change the size); dark = darker than the "
    "background; bright = fluorescence (Otsu). Label set `spheroids`, largest first.",
    "measure_spheroids": "Spheroids (Phase 21): use instead of measure_objects for the spheroids. "
    "Z stacks: 3D volume, surface, sphericity, Z extent and coverage (z_incomplete), core / live / "
    "dead volumes, counts of objects whose parent is the spheroids (e.g. Cellpose 3D nuclei) per "
    "spheroid and per mm³; axial_scale corrects the Z step. "
    "Area, equivalent diameter, perimeter, axes, circularity, solidity, compactness, volume "
    "(sphere π/6·d³ and ellipsoid π/6·L·W²), darkness (brightfield), edge blur (edge_sigma_um, "
    "focus), position (offcentre, touching_edge). Phase 21b extras: radial profiles in `shells` "
    "concentric shells (core, live and dead channels); necrotic core at half height between rim "
    "and centre of the core channel (dead stain: bright core; brightfield: dark core) — area, "
    "fraction, viable rim; live/dead areas and dead fraction (threshold: background median + "
    "threshold_sd robust SD, or fixed); invasion (opening by invasion_core_um with distance "
    "transforms: body vs protrusions, their count, reach, skeleton length and branches).",
    "spheroid": "Spheroids (Phase 21): mode organoids = every object ≥ min_organoid_um2: count, "
    "diameter distribution, total area/volume, density per mm², forming efficiency from the "
    "layout's seeding density (well_area_mm2 extrapolates from the imaged fields), flag "
    "no_organoids. Mode single (default): one spheroid per well — the largest object not touching "
    "the image border (≥ min_spheroid_um2). Readouts `sp_` (volume by `volume`: sphere or "
    "ellipsoid), volume vs vehicle, Z′; flags no_spheroid, multiple, touching_edge, off_centre, "
    "out_of_focus, debris, irregular (exclude_flags set sp_qc_ok = 0). Writes `spheroids/`: "
    "spheroids.csv, wells.csv, crops and a sortable gallery.html.",
    "mitotic_errors": "Assay (Phase 20e), right after the nuclear segmentation, 2D: micronuclei = "
    "small DNA objects (min_area_um2 to max_area_fraction × the median nuclear area) within "
    "search_um of a nucleus edge but not touching it; segmented ones are taken out of the nuclei, "
    "smaller ones found with threshold_fraction × the median nuclear DNA intensity. Label set "
    "`micronuclei` (parent: nearest nucleus) → `micronuclei_count` per nucleus. With `cells` "
    "(segmented on a cell stain, not grown from the nuclei) each nucleus is linked to its cell → "
    "`nuclei_count` per cell (multinucleation).",
    "proliferation": "Assay (Phase 20e): % positive nuclei for Ki67, EdU/BrdU and pH3. Thresholds "
    "per marker: secondary-only wells (role unstained, median + threshold_sd robust SD of log "
    "intensity), vehicle_mixture (two log-normal populations, equal-likelihood point), "
    "vehicle_otsu, vehicle_robust (rare positives, pH3) or fixed; auto picks secondary-only when "
    "present. Micronuclei per 100 nuclei and % with micronuclei; multinucleated and binucleated "
    "cells; fold vs vehicle; Z′. Writes `pf_` columns and `assays/proliferation/` (control "
    "histograms and per-well intensity distributions). See the Assays guide.",
    "register_timepoints": "Kinetics (Phase 22a): drift between the time points of each "
    "field, measured by phase cross-correlation of consecutive time points (mean-subtracted, "
    "Hann-windowed, sub-pixel) on `channel` and accumulated; each time point is moved back onto "
    "the first and the field is cropped to the area seen at every time point (counts then "
    "cover the same area throughout). A jump between two time points larger than max_shift_um "
    "is a stage move: not corrected, reported. Cannot be combined with stitch_well. Put it "
    "first. See the Kinetics guide.",
    "ilastik_pixel": "ilastik (Phase 16b): runs a trained Pixel Classification project with "
    "ilastik's headless command line, once per well with all its fields (ilastik starts slowly). "
    "Adds one channel per class, `ilastik_<class>`: probabilities (0–1) or 0/1 masks from Simple "
    "Segmentation; the original channels are unchanged and stay measurable. The project's SHA-256, "
    "ilastik's version and the command lines are recorded (manifest, ilastik/ folder). Needs "
    "ilastik (Tools → ilastik settings). See the ilastik guide.",
    "ilastik_autocontext": "ilastik (Phase 16b): as ilastik_pixel for an Autocontext project "
    "(the second stage's probabilities or segmentation).",
    "probabilities_to_objects": "ilastik (Phase 16b): objects from a class channel: smoothing, "
    "threshold (with optional hysteresis from low_threshold), holes filled, touching objects "
    "split by the distance transform (min_distance_px) or by seeds from another label set, "
    "size filter. Known-answer test: two touching discs become two objects.",
    "ilastik_class_areas": "ilastik (Phase 16b): % of each field where each class is the most "
    "likely (`ilastik_area_pct_<class>`; per-well means in the summary).",
    "ilastik_objects": "ilastik (Phase 16c): classifies objects with a trained Object "
    "Classification project, once per well. input labels (default): an existing label set "
    "(e.g. Cellpose nuclei) goes to ilastik as the label image and keeps its IDs; each object "
    "gets the class most of its pixels have in ilastik's Object Predictions (the agreement is "
    "kept). input prediction_maps / segmentation_image: ilastik makes its own objects from the "
    "class channels (two calls per well); they become the label set <name>_objects. Needs "
    "ilastik_object_classes to put the classes into the tables. See the ilastik guide.",
    "ilastik_object_classes": "ilastik (Phase 16c): table step; adds class_<name> (class name) "
    "and class_<name>_agreement to the objects and per-well class_<name>_<class>_count and "
    "_fraction (plate maps, reports, pharmacology readouts).",
    "ilastik_multicut": "ilastik (Phase 16d): boundary-based segmentation with a trained "
    "Multicut project, once per well: the class channels of an earlier ilastik_pixel step "
    "(trained with a boundary class) go to ilastik as --probabilities; the Multicut Segmentation "
    "becomes the label set `name` (unique IDs). Good for dense tissue and touching cells.",
    "ilastik_tracking": "ilastik (Phase 16d): tracking with a trained Tracking project; each "
    "field's whole time series goes to ilastik (a binary mask from a label set or class "
    "channel, or the class probabilities). ilastik's objects of every time point become the "
    "label set `name`; their tracks and parent tracks (CSV-Table export) feed ilastik_tracks. "
    "Needs a time-lapse plate.",
    "ilastik_tracks": "ilastik (Phase 16d): table step; ilastik's tracks in the object tables "
    "(trk_id, trk_parent, trk_divided) and the same track table (tracking/tracks.csv) and "
    "per-well trk_ values as the built-in tracking step: tracks, speed, straightness, "
    "divisions, division rate, cell-cycle time.",
    "confluence": "Kinetics (Phase 22a): % of the field covered by cells, per field and time "
    "point (`confluence_pct`, `confluence_area_um2`; per-well means in the summary). texture "
    "(brightfield/phase): local SD of the image minus its median filter, threshold calibrated "
    "on fields pooled across the plate and time points, mask edge corrected for the window "
    "(gaps narrower than ~3 × texture_um count as covered). threshold: a fluorescent stain. "
    "labels: the area of a label set. See the Kinetics guide.",
    "tracking": "Kinetics (Phase 22d): table step that tracks nuclei over the time points of each "
    "field (scipy only): frame-to-frame optimal assignment on the distance to the predicted "
    "position (persistent motion), gated by max_speed_um_per_h; divisions (daughters smaller than "
    "max_daughter_ratio × the mother, the mother's track ends); gap closing over max_gap time "
    "points. Object columns `trk_id`, `trk_parent`, `trk_divided`; `tracking/tracks.csv`; per well "
    "speed, straightness, divisions, division rate, cell-cycle time, lost fraction. Put "
    "register_timepoints first. See the Kinetics guide.",
    "kinetics": "Kinetics (Phase 22b): table step for time-lapse runs. Per well a growth fit "
    "on the log of the readout (exponential, lag_exponential, logistic; AICc) → `kn_doubling_h`, "
    "`kn_growth_rate_per_h`, `kn_lag_h`, `kn_capacity`, plus AUC, fold changes and times to a "
    "fold change or level. With compounds in the layout: IC50 per time point, GR from treatment "
    "(each well's own value before treatment as time zero) and GR over intervals (interval_h, "
    "else consecutive time points); GR50/GRmax per compound and time. Kinetic QC (Phase 22c): "
    "gaps and missing time points, focus drift, edge evaporation, vehicle growth vs the registry "
    "history (`kn_qc_flags`). Writes `kinetics/` with `report.html` (plate map with sparklines, "
    "condition × time heat map, curve overlays, QC). "
    "See the Kinetics guide.",
    "cytoplasm": "Assay (Phase 20d): a cytoplasm label set with the nucleus IDs, for nuclear "
    "translocation. ring: pixels between gap_um and gap_um + width_um from the nuclear edge that "
    "are closer to this nucleus than to any other (never inside a neighbour); with cells (grown "
    "from the nuclei) the ring stays inside its own cell. cells: the cell minus its nucleus and "
    "the gap. Put it before measure_objects.",
    "translocation": "Assay (Phase 20d): nuclear/cytoplasmic ratio of a reporter per cell "
    "(nucleus vs the cytoplasm label set), log2 ratio, and responding cells above a threshold "
    "from the unstimulated controls (median + threshold_sd robust SD of log N/C; a percentile; "
    "Otsu over unstimulated and stimulated cells; or fixed). mode agonist: vehicle = "
    "unstimulated; inhibitor: positive controls = unstimulated. Per well % responding, median "
    "N/C, activation between the controls, Z′. Writes `tl_` columns and `assays/translocation/`. "
    "EC50/IC50 with the pharmacology step on tl_responding_pct. See the Assays guide.",
    "apoptosis": "Assay (Phase 20c): apoptosis and cytotoxicity on the nuclei. An apoptosis "
    "marker (caspase-3/7, cleaved caspase-3, or annexin V measured on the cells) and a dead-cell "
    "dye (PI, SYTOX, DRAQ7) are thresholded from the vehicle nuclei (median + threshold_sd robust "
    "SD of log intensity; or a percentile; or Otsu over vehicle and positive controls); condensed "
    "nuclei (small and bright vs vehicle) or a classifier's classes count as apoptotic; very small "
    "bright objects are fragments, not cells. Categories viable / early / late apoptotic / "
    "necrotic, viable count, Z′ of the positive control, and GR from a time-zero count (0 h wells "
    "or time_zero_count): cytotoxic below 0, cytostatic below no_effect_gr. Writes `ap_` columns "
    "and `assays/apoptosis/`. See the Assays guide.",
    "cell_cycle": "Assay (Phase 20a): cell cycle from the integrated nuclear intensity of a DNA "
    "stain, scaled so the vehicle wells' G1 peak is 2N; per well a Dean–Jett–Fox-style fit "
    "(Gaussian G1 and G2/M peaks with a shared CV, uniform broadened S phase) gives % G1, S and "
    "G2/M, with sub-G1 and > 4N by DNA content; optional EdU (2D gating) and pH3 (mitotic "
    "index), thresholds from the vehicle wells. Writes `cc_dna_n`, `cc_phase` per nucleus, the "
    "`cc_` readouts per well, and `assays/cell_cycle/` (calibration, per-well CSV, report). "
    "Warns about max projections and single confocal planes. See the Assays guide.",
    "object_qc": "Object-level QC: learns the median and robust spread (1.4826 × MAD) of size "
    "(log area), shape (solidity) and the mean intensity of the first channel (log) from "
    "objects sampled across the plate, and flags objects beyond the robust-Z threshold in any "
    "of them (merged cells, fragments, out-of-focus or debris-like objects). Writes "
    "`qc_outlier`, `qc_excluded` and `qc_reason` per object and the outlier count and "
    "fraction per well. Excluded objects are left out of per-well feature statistics and of "
    "later table steps; counts are unchanged. See the Image QC guide.",
    "classify_objects": "Applies an object classifier trained in the Classify panel (or with "
    "`hcs-studio train-classifier`) on example objects: random forest or histogram gradient "
    "boosting on the measured features. With `plate_robust_z` normalisation each plate's "
    "features become robust Z-scores of that plate first, which removes plate-to-plate "
    "intensity offsets but assumes most objects on a plate are alike. Writes the class, its "
    "probability and every class probability per object, and class counts and fractions per "
    "well; the classifier file's SHA-256, classes and cross-validated accuracy go into "
    "`tables/analysis_summary.json`.",
    "phenotypic_clusters": "Groups objects by their features without labels: k-means, "
    "HDBSCAN, or Leiden (needs the optional `leidenalg` and `igraph`). Features are robust "
    "Z-scores clipped to ±10, optionally reduced by PCA. Clusters are learned on objects "
    "sampled across the plate; each object is then assigned (k-means: nearest centre; "
    "HDBSCAN and Leiden: majority of the 5 nearest sampled objects, an approximation). "
    "Cluster 1 is the largest, -1 is noise. Writes `cluster_<name>` per object and fractions "
    "per well; the features that set each cluster apart are listed in the analysis summary.",
    "spatial_clusters": "Spatial organisation of objects in the whole well (needs `stitch_well`) "
    "or in each field: colonies by DBSCAN or HDBSCAN, nearest-neighbour distances, the "
    "Clark-Evans ratio (Donnelly edge correction), Voronoi cell area and Delaunay neighbours "
    "(NaN where the cell is not closed inside the imaged area), Ripley's K, L and H = L − r "
    "with border correction, and the pair correlation g(r), compared with envelopes from "
    "simulated complete spatial randomness in the same area. `spatial_pattern_<r>um` is +1 "
    "(clustered), -1 (regular) or 0 (consistent with randomness at the 5 % pointwise level "
    "with 39 simulations). Full curves per well are in `wells/<W>/analysis.json`.",
    "pharmacology": "Normalizes per-well readouts per plate and cell line (% of control, % "
    "inhibition or GR), checks plate QC (Z′, SSMD, CVs), fits dose–response curves "
    "(4PL/5PL/3PL in GraphPad Prism's form) and scores combination matrices (Bliss, HSA, "
    "Loewe, ZIP). Writes `pharmacology/` (tables, figures, HTML and PDF report) and adds "
    "curves and heat maps to `index.html`. Methods: docs/pharmacology.md.",
    "compound_figures": "One montage per compound of the plate layout: concentrations as "
    "columns (vehicle first), cell lines as rows, the representative field of each well, "
    "linked contrast and scale bars; each with a JSON sidecar to regenerate it. Written to "
    "`figures/`.",
    "experiment_report": "One report per run (`experiment_report/`, HTML and PDF): provenance, "
    "plate layout figures, QC, dose–response, combinations, representative montages and "
    "methods text generated from the manifest. Runs pharmacology first when the layout has "
    "drugs and `pharmacology/` does not exist yet.",
    "register_channels": "Corrects chromatic shift by translating channels, either by given "
    "shifts in µm or by shifts measured on an image of multicolour beads. Linear (sub-pixel "
    "interpolation), so measurements stay valid. Place it before segmentation and "
    "colocalization.",
}


def render() -> str:
    load_builtin_modules()
    lines = [
        "# Processing modules",
        "",
        "Generated by `scripts/gen_processing_docs.py` from the module definitions "
        "(`hcs-studio modules` prints the same reference).",
        "Channel numbers start at 1. Modules marked **display only** must not be used for "
        "intensity measurement.",
        "",
        "These algorithms are verified against synthetic images with a known answer. Validate "
        "them on your own data, and against your reference software, before relying on them "
        "for quantitative claims.",
        "",
    ]
    for name in MODULES:
        module = MODULES.get(name)
        flag = "" if module.preserves_linearity else " — **display only**"
        lines += [f"## {module.title} (`{name}`){flag}", "", NOTES[name], ""]
        if issubclass(module, TableModule):
            lines += [
                "Table step: runs on the object tables after all wells are processed, and can "
                "be applied to a finished run with `hcs-studio analyze`.",
                "",
            ]
        elif issubclass(module, RunStep):
            lines += [
                "Run step: runs once on the finished run (after all wells, the tables and "
                "the report); it does not change images, is not part of the pipeline hash, "
                "and can be applied to a finished run with `hcs-studio analyze`.",
                "",
            ]
        elif module.needs_fit:
            lines += [
                "Learns a plate-level estimate before processing; the estimate is recorded in "
                "`run_manifest.json`.",
                "",
            ]
        lines += ["| Parameter | Default | Meaning |", "|---|---|---|"]
        for key, info in module.Params.model_fields.items():
            if info.exclude:
                continue
            default = (
                "required"
                if info.is_required()
                else repr(info.get_default(call_default_factory=True))
            )
            lines.append(f"| `{key}` | {default} | {info.description or ''} |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    target = Path(__file__).resolve().parent.parent / "docs" / "processing.md"
    target.write_text(render(), encoding="utf-8")
    print(f"Wrote {target}")
