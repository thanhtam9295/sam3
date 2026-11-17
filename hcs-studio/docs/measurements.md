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

## Whole-well position (`stitch_well`)

| Column | Meaning |
|---|---|
| `well_y_um`, `well_x_um` | object centroid in the well (stitched field position + centroid in the field) |
| `in_overlap_duplicate` | True for the copy of an object seen in a second field, where fields overlap; left out of counts and statistics |
| `field_y_um`, `field_x_um` (`fields.csv`) | stitched top-left corner of the field in the well |

## Table-step columns (`objects_annotations.parquet`, merged into `per_object_all.parquet`)

Rows of other object types, and overlap duplicates, are empty.

| Column | Step | Meaning |
|---|---|---|
| `gate_<name>` | gate | inside the gate |
| `class_<name>` | classify | most likely class (`unassigned` below `min_probability`) |
| `class_<name>_probability` | classify | probability of that class |
| `class_<name>_p_<class>` | classify | probability of each class |
| `cluster_<name>` | phenotypic clusters | cluster, 1 = largest, -1 = noise |
| `spatial_colony`, `spatial_colony_size` | spatial | colony (1 = largest, -1 = none) and its number of objects |
| `spatial_nn_um` | spatial | distance to the nearest object |
| `spatial_voronoi_area_um2`, `spatial_local_density_per_mm2` | spatial | Voronoi cell area and its inverse; empty at the edge of the imaged area |
| `spatial_delaunay_neighbours` | spatial | Delaunay neighbours; empty at the edge |

## Per-well summary (`tables/per_well_summary.csv`)

For each well: fields used and excluded (out of focus, saturated, edge), imaged area,
`<label set>_count` and `<label set>_per_mm2` (excluded fields left out), `low_count`, and
for every numeric feature `<label set>_<feature>_mean / _median / _std`, plus the median
of each colocalization statistic per channel pair. In stitched wells, counts include each
object once and the area is that of the fields together (overlaps once).

Table steps add:

| Column | Meaning |
|---|---|
| `gate_<name>_count`, `_fraction` | objects inside the gate, and their fraction |
| `class_<name>_<class>_count`, `_fraction` | objects per class |
| `cluster_<name>_<k>_count`, `_fraction` | objects per cluster (`noise` for -1) |
| `spatial_objects`, `spatial_density_per_mm2` | objects analysed and their density in the imaged area |
| `spatial_colonies`, `spatial_colony_size_mean / _median`, `spatial_colony_area_um2_mean`, `spatial_fraction_in_colonies` | colonies (area = convex hull) |
| `spatial_nn_mean_um`, `spatial_nn_median_um` | nearest-neighbour distances |
| `spatial_clark_evans_R` | Clark–Evans ratio with Donnelly's edge correction: 1 random, < 1 clustered, 2 square lattice, 2.15 hexagonal |
| `spatial_H_<r>um` | Ripley's H(r) = L(r) − r with border correction; 0 for randomly placed objects |
| `spatial_H_<r>um_csr_low / _high` | pointwise envelope of H from simulated random placement (39 simulations: 5 % two-sided) |
| `spatial_g_<r>um` | pair correlation at r (1 = random, > 1 more neighbours at that distance) |
| `spatial_pattern_<r>um` | +1 clustered, -1 regular, 0 within the envelope |
| `spatial_groups` | with per-field scope or several timepoints: the number of patterns averaged |
