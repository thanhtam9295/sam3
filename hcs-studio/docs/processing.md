# Processing modules

Generated from the module definitions (`hcs-studio modules` prints the same reference).
Channel numbers start at 1. Modules marked **display only** must not be used for intensity measurement.

These algorithms are verified against synthetic images with a known answer. Validate them on your own data, and against your reference software, before relying on them for quantitative claims.

## Autofluorescence: AF channel subtraction (`af_channel`)

Subtracts `k × AF channel` from each target channel, then clips at 0. `k` is estimated by robust regression (Tukey biweight), which ignores pixels carrying real signal as long as background pixels are the majority.

Learns a plate-level estimate before processing; the estimate is recorded in `run_manifest.json`.

| Parameter | Default | Meaning |
|---|---|---|
| `channels` | None | Channel numbers to process, starting at 1. Empty = all channels. |
| `af_channel` | required | Number of the autofluorescence / empty channel. |
| `coefficients` | None | Manual bleed-through coefficient per channel number; empty = estimate. |
| `max_fields` | 20 | Fields sampled for the estimate. |
| `pixels_per_field` | 20000 | Pixels sampled per field. |

## Autofluorescence: reference wells (`af_reference`)

Subtracts the autofluorescence measured in unstained or vehicle-only wells, then clips at 0. Place it after flat-field correction.

Learns a plate-level estimate before processing; the estimate is recorded in `run_manifest.json`.

| Parameter | Default | Meaning |
|---|---|---|
| `channels` | None | Channel numbers to process, starting at 1. Empty = all channels. |
| `reference_wells` | [] | Unstained / vehicle-only wells, e.g. [A01, A02]. Empty = wells marked 'unstained' in the plate layout. |
| `mode` | 'scalar' | scalar: one value per channel; image: smooth per-pixel AF profile. |
| `max_fields` | 50 | Maximum reference fields sampled. |
| `smoothing_fraction` | 0.02 | Smoothing of the AF image (fraction of width). |

## Background subtraction (`background`)

Removes smooth background per plane. The radius must be larger than the largest object, or object intensity is removed too.

| Parameter | Default | Meaning |
|---|---|---|
| `channels` | None | Channel numbers to process, starting at 1. Empty = all channels. |
| `method` | 'tophat' | tophat: fast morphological opening; rolling_ball: slower, smoother. |
| `radius_um` | 25.0 | Radius in µm; choose larger than the largest object. |

## CLAHE (local contrast) (`clahe`) — **display only**

Contrast-limited adaptive histogram equalisation, per plane (2D).

| Parameter | Default | Meaning |
|---|---|---|
| `channels` | None | Channel numbers to process, starting at 1. Empty = all channels. |
| `tile_um` | 50.0 | Tile size in µm. |
| `clip_limit` | 0.01 | Contrast limit; higher = stronger. |

## Clarity preset (`clarity`) — **display only**

Mild Gaussian denoise, then an unsharp mask that does not amplify detail at the noise level (limits ringing and noise after deconvolution), then CLAHE.

| Parameter | Default | Meaning |
|---|---|---|
| `channels` | None | Channel numbers to process, starting at 1. Empty = all channels. |
| `amount` | 0.8 | Strength of detail enhancement. |
| `denoise_sigma_px` | 0.7 | Mild pre-smoothing (pixels). |
| `detail_sigma_px` | 3.0 | Scale of enhanced detail. |
| `noise_gate` | 3.0 | Detail weaker than this many noise SDs is not amplified (limits ringing). |
| `tile_um` | 60.0 | CLAHE tile size in µm. |
| `clip_limit` | 0.005 | CLAHE contrast limit. |

## Contrast (percentile rescale, gamma) (`contrast`) — **display only**

Rescales each channel to 0-1 between two percentiles and applies gamma.

| Parameter | Default | Meaning |
|---|---|---|
| `channels` | None | Channel numbers to process, starting at 1. Empty = all channels. |
| `low_percentile` | 0.1 | Maps to black. |
| `high_percentile` | 99.9 | Maps to white. |
| `gamma` | 1.0 | Gamma applied after rescaling to 0-1. |

## Denoise (`denoise`)

Smoothing filters. Output is still treated as measurement-grade, but median and non-local means are not strictly linear; prefer Gaussian or none for quantitative work.

| Parameter | Default | Meaning |
|---|---|---|
| `channels` | None | Channel numbers to process, starting at 1. Empty = all channels. |
| `method` | 'gaussian' | nlm = non-local means (slow, preserves edges). |
| `sigma_px` | 1.0 | Gaussian sigma in pixels. |
| `median_size_px` | 3 | Median window in pixels (odd). |
| `nlm_strength` | 1.0 | Non-local means strength, in units of the noise level. |

## Flat-field correction (`flatfield`)

Estimates the illumination profile from many fields of the plate: each field is divided by its mean, the per-pixel median across fields removes the cells, and a wide Gaussian keeps the smooth shading. The profile is normalised to mean 1, so the mean intensity of a field is preserved. This is a median-based estimate, not the BaSiC algorithm. It needs enough fields with varied content; it fails rather than guess when fewer than `min_fields` are available.

Learns a plate-level estimate before processing; the estimate is recorded in `run_manifest.json`.

| Parameter | Default | Meaning |
|---|---|---|
| `dark_offset` | 0.0 | Camera offset (counts) subtracted before correction. |
| `max_fields` | 200 | Maximum number of fields sampled across the plate. |
| `min_fields` | 8 | Fewer fields than this cannot separate shading from cells; the step fails. |
| `smoothing_fraction` | 0.05 | Gaussian smoothing of the profile, as a fraction of the image width. |

## Z projection (`projection`)

Collapses Z. `edf` picks, per pixel, the plane with the highest local sharpness.

| Parameter | Default | Meaning |
|---|---|---|
| `method` | 'max' | edf = extended depth of field (sharpest plane per pixel). |

## Sharpen (unsharp mask / DoG) (`sharpen`) — **display only**

Unsharp mask or difference-of-Gaussians band boost.

| Parameter | Default | Meaning |
|---|---|---|
| `channels` | None | Channel numbers to process, starting at 1. Empty = all channels. |
| `method` | 'unsharp' | unsharp mask, or difference-of-Gaussians band boost. |
| `sigma_px` | 2.0 | Blur sigma (unsharp) in pixels. |
| `sigma_large_px` | 6.0 | Large sigma (dog) in pixels. |
| `amount` | 1.0 | Strength of the added detail. |

## Linear spectral unmixing (`unmix`)

Solves the linear mixing model by least squares and clips negative values. The output has one channel per fluorophore.

| Parameter | Default | Meaning |
|---|---|---|
| `mixing_matrix` | required | Reference spectra: one row per detection channel, one column per fluorophore (signal of that fluorophore in each channel). |
| `fluorophores` | required | Name of each fluorophore (matrix column). |

