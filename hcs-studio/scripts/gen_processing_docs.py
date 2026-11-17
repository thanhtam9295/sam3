"""Regenerate docs/processing.md from the module definitions.

Run from the hcs-studio folder: python scripts/gen_processing_docs.py
"""

from __future__ import annotations

from pathlib import Path

from hcs_studio.processing.base import MODULES, load_builtin_modules

NOTES = {
    "flatfield": "Estimates the illumination profile from many fields of the plate: each field "
    "is divided by its mean, the per-pixel median across fields removes the cells, and a wide "
    "Gaussian keeps the smooth shading. The profile is normalised to mean 1, so the mean "
    "intensity of a field is preserved. This is a median-based estimate, not the BaSiC "
    "algorithm. It needs enough fields with varied content; it fails rather than guess when "
    "fewer than `min_fields` are available.",
    "af_reference": "Subtracts the autofluorescence measured in unstained or vehicle-only wells, "
    "then clips at 0. Place it after flat-field correction.",
    "af_channel": "Subtracts `k × AF channel` from each target channel, then clips at 0. `k` is "
    "estimated by robust regression (Tukey biweight), which ignores pixels carrying real "
    "signal as long as background pixels are the majority.",
    "background": "Removes smooth background per plane. The radius must be larger than the "
    "largest object, or object intensity is removed too.",
    "unmix": "Solves the linear mixing model by least squares and clips negative values. The "
    "output has one channel per fluorophore.",
    "denoise": "Smoothing filters. Output is still treated as measurement-grade, but median and "
    "non-local means are not strictly linear; prefer Gaussian or none for quantitative work.",
    "contrast": "Rescales each channel to 0-1 between two percentiles and applies gamma.",
    "clahe": "Contrast-limited adaptive histogram equalisation, per plane (2D).",
    "sharpen": "Unsharp mask or difference-of-Gaussians band boost.",
    "clarity": "Mild Gaussian denoise, then an unsharp mask that does not amplify detail at the "
    "noise level (limits ringing and noise after deconvolution), then CLAHE.",
    "projection": "Collapses Z. `edf` picks, per pixel, the plane with the highest local "
    "sharpness.",
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
    "segment_cellpose": "Cellpose 4 (`cpsam_v2` is the default Cellpose-SAM model). Give one "
    "channel, or two (e.g. cytoplasm plus nuclei) for whole cells. The diameter is optional. "
    "Z stacks are segmented in 3D (`do_3d`), plane by plane with stitching, or plane by plane. "
    "A trained model file can be used in place of a built-in name. Uses the GPU when available; "
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
        if module.needs_fit:
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
