# Cell Painting profiling

Morphological profiling in the style of Cell Painting (Bray et al. 2016): rich per-cell features
→ single-cell tables in CellProfiler's naming → (Phase 24b) normalized well and treatment
profiles → (Phase 24c) phenotypic activity, similarity and clustering. This page covers the
features and single-cell tables (Phase 24a).

pycytominer and copairs come with the `profiling` extra:

```bash
uv pip install --python .venv/bin/python -e ".[gui,dev,segmentation,profiling]"
```

## The Cell Painting preset

Five stains by default — DNA (Hoechst), ER (concanavalin A), RNA (SYTO 14), AGP (phalloidin +
WGA) and Mito (MitoTracker); map them to your plate's channel names:

```bash
hcs-studio cellpainting preset --channels Hoechst,ConA,SYTO14,Phalloidin,MitoTracker \
  --map dna=Hoechst --map er=ConA --map rna=SYTO14 --map agp=Phalloidin --map mito=MitoTracker \
  --output cell_painting.yaml
```

Steps: nuclei (Otsu on DNA, touching nuclei split), cells grown from the nuclei up to
`--cell-um` within the cell stain (AGP, else RNA, or `--map cell=…`), cytoplasm = cell minus
nucleus (same ID as its nucleus and cell), then every feature group on the three compartments
and image QC. Custom channel sets: any roles, as long as `dna` is given.

## Features

`measure_objects` on nuclei, cells and cytoplasm: AreaShape (area, perimeter, axes,
eccentricity, solidity, extent, form factor; Zernike optional), Intensity (mean, median, SD,
min, max, integrated, quartiles and 5th/95th percentiles), RadialDistribution (fraction of each
channel in rings from the centre, centre/periphery ratio), Texture (13 Haralick features at
pixel distances 1, 3 and 5, averaged over directions), Granularity (spectrum of openings),
**Correlation** (Pearson correlation of every channel pair over the object's pixels — new; the
preset turns it on with `correlation: true`), Neighbors and Location.

## Single-cell tables in CellProfiler naming

```bash
hcs-studio cellpainting single-cells runs/CP1__cell_painting__… --map dna=Hoechst --map er=ConA …
```

writes `profiling/single_cell.parquet`: one row per cell, the three compartments side by side
(joined on the shared object ID), feature names as CellProfiler writes them —
`<Compartment>_<Category>_<Feature>_<Channel>[_<Scale>]`, e.g.
`Nuclei_Intensity_MeanIntensity_DNA`, `Cells_Texture_Contrast_ER_3`,
`Cytoplasm_Correlation_Correlation_ER_RNA` — so pycytominer finds the features by their
prefixes without manual mapping. Channels are named by the role map (`--map`), others keep
their name. Columns:

| Metadata | From |
|---|---|
| `Metadata_Plate`, `Metadata_Well`, `Metadata_Site` (field), `Metadata_TimePoint`, `Metadata_ObjectNumber` | the run |
| `Metadata_Role`, `Metadata_Compound` (DMSO for vehicle wells), `Metadata_Concentration` (M), `Metadata_CellLine`, `Metadata_Treatment`, `Metadata_ReplicateGroup` | the plate layout |
| `Metadata_QCExcluded` | object QC (Phase 18): kept and flagged; dropped before aggregation |

Summaries of one compartment carried by another (a nucleus's "cells" counts and means) are left
out: the compartment is in the table itself. The full name table is in
`cellpainting/naming.py`.

## Importing CellProfiler data

```bash
hcs-studio cellpainting import SQ00014613.sqlite --output sq14613_single_cells.parquet
```

reads CellProfiler's ExportToDatabase / cytominer-database SQLite files, ExportToSpreadsheet CSV
folders (`Image.csv`, `Nuclei.csv`, `Cells.csv`, `Cytoplasm.csv`) or already-merged Parquet
(CytoTable, pycytominer), and joins the compartments as pycytominer does (each cytoplasm to its
cell and nucleus, within its image), with the image metadata as `Metadata_` columns. Existing
CellProfiler data can then be profiled with the same pipeline as HCS Studio's own.

## Profiles: single cells → wells → treatments (Phase 24b)

```bash
hcs-studio cellpainting profile <run> [<run> …] --output profiles/
hcs-studio cellpainting profile sq14613_single_cells.parquet --output profiles/ \
    --platemap platemap.csv --reference all
```

Sources are run folders (their `profiling/single_cell.parquet`, written first when missing) or
single-cell Parquet files (own or imported). Every stage calls pycytominer, so the profiles are
pycytominer's; HCS Studio chooses the inputs and records the choices:

| Stage | What happens | Options |
|---|---|---|
| single-cell QC | cells flagged by object QC and cells missing more than 10 % of the features are dropped (counted) | `max_missing_fraction` |
| aggregate | median (or mean) of each well's cells, with `Metadata_Object_Count`; well-level `Metadata_` columns carried over | `aggregate` |
| annotate | a plate map joined on the well (columns get `Metadata_`; its well column is `well_position`) — for imported data | `--platemap` |
| normalize | `mad_robustize` = (x − median) / (1.4826 · MAD) against the **vehicle wells of each plate** | `--normalize mad_robustize\|robustize\|standardize\|spherize\|none`, `--reference vehicle\|all`, `--across-plates` |
| feature selection | in order, each on what is left: variance (near-constant), correlation (> 0.9, the most correlated of a pair goes), NA (> 5 % missing), CellProfiler's blocklist, outliers (\|value\| > 500) | `--operations`, `--corr-threshold` |
| consensus | median (or modz) per treatment × cell line; without treatments, one profile per well | `--consensus-by` |

`spherize` (ZCA-cor whitening against the vehicle wells) is fitted on all plates together and is
meant to remove batch effects; it needs more vehicle wells than independent features, so use it
on selected features or many plates.

Files in the output folder: `aggregated.parquet`, `normalized.parquet`,
`feature_selected.parquet`, `consensus.parquet`, `feature_selection.json` (features in/kept, what
each operation removed) and `profiling.json` (options, cell/well/feature counts, pycytominer
version). Each run is recorded as a service event.

The stages are checked against pycytominer's own functions on real JUMP CellProfiler data
(identical values) and against known answers on synthetic plates (docs/validation.md).

## Evaluating profiles (Phase 24c)

```bash
hcs-studio cellpainting evaluate profiles/ --moa moa.csv
hcs-studio cellpainting profile <run> --output profiles/ --evaluate --moa moa.csv   # both
hcs-studio cellpainting similar profiles/ --to "cmpdA 1e-06 | U2OS" -n 10
```

| Question | How | File |
|---|---|---|
| Which treatments do anything? (**activity**) | copairs mean average precision: per well, do the treatment's other wells rank above the vehicle wells of the same plate (cosine)? mAP per treatment, p from a permutation null, Benjamini–Hochberg FDR; active = FDR < 0.05 | `activity.csv`, `figures/activity.png` |
| Is the test calibrated? | 2–4 vehicle wells per plate (at most a third) are held out and scored as a treatment, `vehicle (held out)`: it should not be active | row in `activity.csv` |
| Do compounds with the same MoA look alike? (**consistency**) | consensus profiles of *active* treatments; positives = other compounds of the same MoA, negatives = other MoAs; MoAs with ≥ 2 compounds | `consistency.csv` |
| What is similar to what? | cosine and Pearson matrices of consensus profiles; heat map clustered by average linkage on 1 − cosine; the 10 nearest of each | `similarity_cosine.csv`, `similarity_pearson.csv`, `figures/similarity_heatmap.png`, `similar.csv` |
| What MoA might an unannotated compound have? | the `k` (5) most similar annotated treatments of other compounds vote with their cosine; top-1 accuracy over the annotated ones (leave one compound out) says how far to trust it | `moa_suggestions.csv` |
| Batch effects? | do vehicle wells cluster by plate (and `Metadata_Date` / `Metadata_Batch`)? mAP and silhouette (cosine) before normalization (each feature standardised over all wells) and after | `batch.json` |
| Overview | UMAP of the well profiles (seeded) by treatment and by plate | `umap.csv`, `figures/umap_*.png` |

Everything is summarised in `evaluation.html` (options in `evaluation.json`).

The MoA file is CSV/TSV/XLSX with a compound column (`compound`, `broad_sample`, `pert_iname`,
`name` or `treatment`, with or without `Metadata_`) and a `moa` (or `mechanism`, `target`)
column; it is matched on `Metadata_Compound`.

**Power.** p-values depend on how many wells there are: with 3 replicate wells and only 4
vehicle wells per plate, even a perfect separation (mAP = 1) has p ≈ 0.05. Use ≥ 8 vehicle wells
per plate (JUMP uses many more) and ≥ 3 replicates.

## Synthetic data

`hcs_studio.synthetic.cellpainting.PaintedCells` draws five-channel cells (nucleus, perinuclear
ER, nucleoli, cell edge, mitochondrial puncta) with phenotypes per well — larger nuclei, ER or
AGP intensity, fragmented mitochondria, nucleolar RNA — as multipliers of the vehicle cell.
`profile_cells` writes single-cell tables directly (no images) with seeded shifts per compound,
plate gain/offset and per-plate feature offsets (batch effects) and optional MoA labels.
