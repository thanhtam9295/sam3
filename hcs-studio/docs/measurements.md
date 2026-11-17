# Measurement columns

All intensities are measured on the **linear** image (after flat-field, autofluorescence,
background, registration and deconvolution; never after contrast, CLAHE, sharpen or
clarity). Lengths are in µm, areas in µm², volumes in µm³.

Every row also carries `well`, `row`, `col`, `field`, `timepoint` and `object_type` (the
label set: `nuclei`, `cells`, `spots`, …). `object_id` is the label in the mask image of
that field; `parent_id` is the ID of the parent object (0 = none).

## Morphology

| Column | Meaning |
|---|---|
| `centroid_x_um`, `centroid_y_um` (`centroid_z_um`) | centre of mass within the field |
| `area_um2` (2D) / `volume_um3` (3D) | size |
| `perimeter_um` (2D) / `surface_area_um2` (3D) | boundary length / surface |
| `equivalent_diameter_um` | diameter of a circle (sphere) of the same size |
| `major_axis_um`, `minor_axis_um` | axes of the ellipse (ellipsoid) with the same moments |
| `eccentricity`, `orientation_deg` (2D) | 0 = circle, → 1 elongated; angle of the major axis |
| `form_factor` (2D) | 4π·area / perimeter²; 1 for a circle |
| `solidity`, `extent` | area / convex area; area / bounding-box area |
| `zernike_00` … (2D, optional) | Zernike shape moments |

## Intensity, per channel (`<channel>_…`)

Channel names are lower-cased, with other characters replaced by `_`
(e.g. "Alexa 488" → `alexa_488`).

| Column | Meaning |
|---|---|
| `_mean`, `_median`, `_sd`, `_min`, `_max` | statistics of the object's pixels |
| `_integrated` | sum of the object's pixels |
| `_p05`, `_p25`, `_p75`, `_p95` | percentiles |
| `_radial_outer`, `_radial_middle`, `_radial_inner` | fraction of the integrated intensity in three rings, from the edge to the centre (by distance from the edge) |
| `_center_periphery_ratio` | inner / outer fraction (empty when the outer ring is dark) |
| `_nc_ratio`, `_cytoplasm_mean` | for cells with a nucleus: nuclear mean / cytoplasmic mean (cell minus its nucleus) |
| `_haralick_<feature>_s<d>` | 13 Haralick texture features at pixel distance d, averaged over directions (64 grey levels within the object) |
| `_granularity_r<r>` | % of the object's signal removed by grey opening at radius r px, minus the previous radius: the size spectrum of bright structures |

## Contained objects and neighbours

| Column | Meaning |
|---|---|
| `<child>_count` | number of child objects (e.g. `spots_count` per cell) |
| `<child>_mean_area_um2` | mean size of the children |
| `<child>_<channel>_total_intensity`, `…_mean_intensity` | summed / mean intensity of the children |
| `nearest_neighbour_um` | distance to the nearest object of the same type (centroids) |
| `neighbours_within_radius` | objects of the same type within `neighbor_radius_um` |

## Colocalization (`coloc.csv`)

| Column | Meaning |
|---|---|
| `pearson`, `spearman` | correlation of the two intensities inside the object (or field) |
| `icq` | Li's intensity correlation quotient (−0.5 … +0.5) |
| `m1`, `m2` | Manders: fraction of A where B > 0, and of B where A > 0 |
| `tm1`, `tm2` | Manders above the Costes (or manual) thresholds, as Fiji Coloc 2's tM1/tM2 |
| `threshold_a`, `threshold_b`, `threshold_method`, `costes_slope` | thresholds used (after background subtraction) |
| `overlap`, `overlap_above_thresholds` | Manders' overlap coefficient |
| `p_value`, `random_r_mean`, `random_r_sd` | Costes randomization test; small p = unlikely by chance (Coloc 2 reports 1 − p) |
| `n_pixels`, `warnings` | pixels used; saturation, chromatic shift, too few pixels |
| `fraction_<A>_near_<B>`, `n_<A>` | object-based colocalization, when configured |

## Per-well summary (`tables/per_well_summary.csv`)

For each well: fields used and excluded (out of focus, saturated, edge), imaged area,
`<label set>_count` and `<label set>_per_mm2` (excluded fields left out), `low_count`, and
for every numeric feature `<label set>_<feature>_mean / _median / _std`, plus the median
of each colocalization statistic per channel pair.
