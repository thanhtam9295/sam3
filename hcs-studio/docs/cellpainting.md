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

## Synthetic data

`hcs_studio.synthetic.cellpainting.PaintedCells` draws five-channel cells (nucleus, perinuclear
ER, nucleoli, cell edge, mitochondrial puncta) with phenotypes per well — larger nuclei, ER or
AGP intensity, fragmented mitochondria, nucleolar RNA — as multipliers of the vehicle cell.
