# Spheroids and organoids

Per-spheroid size, shape and volume with QC flags, a sortable gallery and dose–response on
volume. Built so far (21a): **one spheroid per well** (ULA 96/384 plates) on brightfield or a
whole-spheroid fluorescence stain, in 2D. Coming: viability (live/dead), necrotic core, invasion
and organoid mode (21b); confocal 3D (21c). Growth curves over time need Phase 22; ilastik
detection waits for Phase 16 (b).

```bash
hcs-studio assay preset spheroid --map brightfield=BF --channels BF --output spheroids.yaml
hcs-studio run --pipeline spheroids.yaml --input <plate> --workers auto
```

The preset has three steps: `detect_spheroids` → `measure_spheroids` → `spheroid`. For Z stacks
put a projection first (e.g. `max` for brightfield, or the best-focus plane).

## Detection (`detect_spheroids`)

| `method` | How |
|---|---|
| `brightfield` (default) | Spheroids are **textured** (cells), the background is not. The local standard deviation of a difference of Gaussians at the `texture_um` scale (6 µm) marks textured regions (Otsu on log SD, never below 4 × the background noise); smooth dark regions — the **well edge and meniscus shadow** — cancel out in the difference of Gaussians and are not detected, and objects that are not textured inside are dropped. The outline is then placed at **half height** between the spheroid's interior and its local background: the edge of a blurred step lies at half height, so blur does not change the measured size. |
| `dark` | Darker than the background (Otsu), outline at half height. |
| `bright` | Fluorescence (e.g. calcein, Hoechst): Otsu's threshold. |

Objects smaller than `min_area_um2` (300 µm²) are dropped. Every remaining object is kept
(largest first), so debris and second spheroids can be flagged. SAM 3 with a "spheroid" text
prompt (`segment_sam3`, output `spheroids`) can replace this step.

## Measurements (`measure_spheroids`)

Use it **instead of** `measure_objects` for the spheroids (both would write the same objects).

| Column | Meaning |
|---|---|
| `area_um2`, `equivalent_diameter_um`, `perimeter_um`, `major_axis_um`, `minor_axis_um` | size |
| `volume_sphere_um3` | π/6 · d³ from the equivalent diameter |
| `volume_ellipsoid_um3` | π/6 · L · W² from the major (L) and minor (W) axes |
| `circularity` | 4π·area / perimeter² (≈ 0.9 for a digitised circle) |
| `solidity`, `compactness` | area / convex area; spread around the centroid (1 for a disc) |
| `darkness` | 1 − interior / local background (brightfield) |
| `edge_sigma_um` | blur of the outline (Gaussian σ explaining contrast / steepest gradient): focus |
| `offcentre`, `touching_edge` | centroid distance from the field centre / half the field; object cut by the image border |

## One spheroid per well (`spheroid`)

The well's spheroid is the largest object (≥ `min_spheroid_um2`, 2000 µm²) that does not touch
the image border (else the largest). Its values are reported per well as `sp_…`
(`sp_volume_um3` uses `volume`: `sphere` or `ellipsoid`), plus `sp_volume_vs_vehicle_pct` and
Z′ of volume (vehicle vs positive control).

**QC flags** (`sp_flags`):

| Flag | When |
|---|---|
| `no_spheroid` | nothing ≥ `min_spheroid_um2` |
| `multiple` | another object ≥ `multiple_fraction` (0.25) × the spheroid's area |
| `touching_edge` | the spheroid is cut by the image border (its size is underestimated) |
| `off_centre` | `offcentre` > `max_offcentre` (0.5) |
| `out_of_focus` | `edge_sigma_um` > `max_edge_sigma_um` (4 µm) |
| `debris` | smaller objects cover > `debris_fraction` (2 %) of the spheroid's area |
| `irregular` | solidity < `min_solidity` (0.95): attached debris, invasion, a broken spheroid |

Flags in `exclude_flags` (default: no_spheroid, multiple, touching_edge, out_of_focus) set
`sp_qc_ok = 0`. Exclude those wells in the pharmacology module before fitting.

**Outputs** (`spheroids/`): `spheroids.csv` (every object, `sp_main` marks the spheroid),
`wells.csv`, `crops/<well>.png` (the spheroid with its outline, from the run's preview) and
`gallery.html`: one card per well with its flags, sortable by any readout; linked from the run's
`index.html`. Brightfield channels (names such as Brightfield, BF, Phase, TL) are shown in gray.

## Dose–response

`sp_volume_um3` (and `sp_area_um2`, `sp_diameter_um`) default to **% of control** in the
pharmacology module:

```bash
hcs-studio pharm runs/plate1 --readout sp_volume_um3 --normalize percent_of_control
```
