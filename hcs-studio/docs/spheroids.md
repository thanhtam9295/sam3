# Spheroids and organoids

Per-spheroid size, shape and volume with QC flags, a sortable gallery and dose–response on
volume (21a); viability (live/dead), necrotic core, radial profiles, invasion and organoid mode
(21b) — on brightfield or fluorescence, in 2D; and confocal Z stacks in 3D (21c). Growth curves
over time need Phase 22; ilastik detection waits for Phase 16 (b).

```bash
hcs-studio assay preset spheroid --map brightfield=BF --channels BF --output spheroids.yaml
hcs-studio run --pipeline spheroids.yaml --input <plate> --workers auto
```

With live/dead stains, invasion or organoids (Python: `preset(mapping, channels=…,
invasion=True, organoids=True)`):

```bash
hcs-studio assay preset spheroid --map brightfield=BF --map live=Calcein --map dead=EthD1 \
    --channels BF,Calcein,EthD1 --output spheroids_live_dead.yaml
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

## Viability, necrotic core and radial profiles (`measure_spheroids`)

| Option | What it measures |
|---|---|
| `live_channel`, `dead_channel` | live / dead stain areas inside the spheroid: pixels above the background (outside all objects) median + `threshold_sd` (5) robust SD, or `live_threshold` / `dead_threshold`; `dead_fraction` = dead / (live ∪ dead) stained area (pixels with both count as dead) |
| `core` (on) | **necrotic core**: on `core_channel` (default: the dead stain if mapped — a bright core — else the brightfield channel — a dark core), smoothed by `core_smooth_um` (6 µm), the core is outlined at **half height between the rim** (10–30 % of the way in) **and the centre** (deepest 15 %). `core_area_um2`, `core_fraction` (of the spheroid's area), `core_diameter_um`, `rim_um` (viable rim: equivalent radius minus core radius). No core when the rim–centre contrast is below `min_core_contrast` (0.15). |
| `shells` (10) | **radial profiles**: mean intensity in concentric shells from the edge (shell 1) to the centre, by distance from the edge relative to the deepest point — so irregular spheroids work too. Columns `radial_<core/live/dead>_sNN`; `spheroids/radial_profiles.csv` (long format) and `radial_profiles.png` (mean profile per role). |

## Invasion (`invasion: true`)

For spheroids in Matrigel or collagen. Detection must follow the invading cells:
`detect_spheroids` → `protrusion_reach_um` (the preset uses 300 µm with `invasion=True`). The
spheroid **body** is the object after a morphological opening with radius `invasion_core_um`
(20 µm; computed with distance transforms); everything thinner that sticks out is protrusion.

| Column | Meaning |
|---|---|
| `invasion_core_area_um2`, `invasion_area_um2`, `invasion_fraction` | body area; protrusion area; protrusion / total area |
| `protrusions`, `protrusion_length_um`, `max_invasion_um` | protrusions reaching `min_protrusion_um` (10 µm) from the body; their mean reach; the farthest reach |
| `skeleton_length_um`, `skeleton_branches` | skeleton of the protrusions: length and end points |

Invading spheroids are also flagged `irregular` (low solidity) — expected in this assay.

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

## Organoid mode (`mode: organoids`)

Many organoids per well (in matrix, several fields per well). Every object ≥ `min_organoid_um2`
(200 µm²) is an organoid; those cut by the image border are counted but left out of the size
distribution.

| Readout | Meaning |
|---|---|
| `sp_org_count`, `sp_org_count_complete` | organoids; not cut by the border |
| `sp_org_diameter_{mean,median,p10,p90}_um` | size distribution |
| `sp_org_area_total_um2`, `sp_org_volume_total_um3` | summed area and volume |
| `sp_org_per_mm2` | organoids per imaged mm² (fields × field area) |
| `sp_ofe_pct` | **organoid-forming efficiency**: organoids / seeded cells (`seeding_density` in the layout); with `well_area_mm2` the count is extrapolated from the imaged fields to the well |
| `sp_dead_fraction`, `sp_core_fraction` | area-weighted over the organoids |

`sp_volume_vs_vehicle_pct` uses the total organoid volume; the flag `no_organoids` fails a well.
`spheroids/organoid_sizes.csv` and `size_distribution.png` (box per well) go with the gallery,
whose cards show the field with the most organoid area.

## Confocal Z stacks (3D)

```python
preset(
    {"fluorescence": "Hoechst", "dead": "EthD1", "nuclei": "Hoechst"},
    channels=["Hoechst", "EthD1"],
    z_planes=60,
)
```

gives `attenuation` → `detect_spheroids` (3D) → [`segment_cellpose` 3D nuclei with
`parent: spheroids` → `measure_objects` on the nuclei] → `measure_spheroids` → `spheroid`.
Brightfield stacks are projected (extended depth of field) and analysed in 2D.

**Depth attenuation** (`attenuation`): deeper planes of a thick sample are dimmer (scattering,
absorption). Per plane the step takes the 90th percentile of the foreground above the plane's
background (planes whose foreground covers at least half the largest), fits an exponential decay
I₀·exp(−z/λ) (Theil–Sen line through the logarithm) and multiplies the signal above background by
exp(z/λ) from the first plane with signal on. `per_plane` normalises each plane instead. Gains are
capped at `max_gain` (10×) with a warning; a stack that does not get dimmer with depth is left
alone (warning). The report gives λ and the factor per plane. **Without it the dim deep half of
a spheroid falls below the segmentation threshold** (on the test spheroid with λ = 80 µm: 44 %
of the volume lost).

**3D detection**: with method `bright` on a Z stack, objects are found with Otsu's threshold in 3D
and their surface is placed at half height between the bright interior and the local background
(distances in µm, so the coarser Z step is accounted for); holes (an unstained necrotic core) are
filled.

**3D measurement** (`measure_spheroids` on a Z stack):

| Column | Meaning |
|---|---|
| `volume_3d_um3`, `equivalent_diameter_3d_um` | voxels × voxel volume; diameter of a sphere of that volume |
| `surface_area_um2`, `sphericity` | smoothed marching-cubes surface; π^⅓ (6V)^⅔ / surface (1 = sphere) |
| `z_extent_um`, `z_incomplete`, `z_coverage` | depth of the segmentation; touches the first or last plane; Z extent / projected diameter (share inside the stack, assuming a spheroid as deep as wide) |
| `core_volume_um3`, `core_fraction`, `rim_um` | necrotic core in 3D (half height between the rim and the centre) |
| `live_volume_um3`, `dead_volume_um3`, `dead_fraction` | stained volumes |
| `<child>_count`, `<child>_per_mm3` | objects whose parent is the spheroid (nuclei from Cellpose 3D) and their density |
| `area_um2`, `circularity`, … | from the projection of the 3D mask |

`axial_scale` corrects the nominal Z step for refractive-index mismatch (e.g. ~0.8 for an air
objective imaging into an aqueous sample); it scales volumes and depths. `sp_volume_um3` is the 3D
volume (`volume: auto`). The flag **`incomplete_z`** (the stack does not cover the whole spheroid;
its volume is underestimated) fails the well by default.

Stain thresholds (2D and 3D): half height between the background and the stain's plateau inside
the object (99th percentile), never below the background median + 5 robust SD — blur does not
grow the stained region, and an unstained spheroid stays at 0.

## Dose–response

`sp_volume_um3` (and `sp_area_um2`, `sp_diameter_um`) default to **% of control** in the
pharmacology module:

```bash
hcs-studio pharm runs/plate1 --readout sp_volume_um3 --normalize percent_of_control
```
