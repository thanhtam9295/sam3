# Comparison and figure export

The **Compare** tab shows fields of a finished run side by side, with linked zoom, pan,
Z and contrast, and exports publication figures, the panels' data and movies. Every
export can be regenerated exactly from the small JSON written next to it.

## Choosing wells

- **Results** tab: outline wells on the plate (Ctrl+click), then **Compare wells side by
  side** (or just the selected well).
- **Pharmacology** tab: click a curve point or a matrix cell, then **Compare**.
- **Profiles** tab: lasso wells, then **Add last selection** in the Compare tab.
- The filter bar in the Compare tab: e.g. *MCF7, Drug X, all concentrations*.

## Arranging panels

| Arrangement | Panels |
|---|---|
| grid | the wells in order; rows × columns (automatic or given) |
| by concentration × cell line | columns ascending by total drug concentration, rows by cell line, headers from the annotation (one well per cell) |
| combination matrix | a drug pair as an image grid: drug A down (highest at the top, as in the heat maps), drug B across; the 0 row/column are the single agents and the corner the matrix's own vehicle well |
| one compound | that compound's single-agent wells and each cell line's vehicle wells |

## Showing panels

- **Image**: `corrected` (the saved linear image), `enhanced` (display only, if saved),
  `raw` (read again from the raw data) or `step:<n>`: the raw field processed by the run's
  pipeline up to step n with the run's saved plate-level estimates (e.g. after AF
  correction, deconvolution, projection or background removal).
- **Field**: the first field, or the **representative field**: the one whose readout (a
  column of the well's `fields.csv`, e.g. `n_nuclei`, or an object feature, by its median
  per field; empty = object count) is closest to the well's median.
- **Z**: a plane (the slider moves all panels) or the maximum over Z.
- **Linked contrast** (default): one low/high/gamma and colour per channel for every
  panel, so images are directly comparable. **Auto contrast** uses the pooled 0.1–99.9th
  percentiles of all panels. **Unlink clicked panel** gives one panel its own contrast; it
  is then marked *⚠ contrast unlinked* on screen and in exported figures.
- Zoom and pan are linked; **Lock / unlock zoom of clicked panel** frees one panel.
- **Outlines** of label sets (masks); in figures, `<labels>:<column>` colours objects by a
  category column (`class_…`, `cluster_…`, `gate_…`).
- **Labels**: a template from the annotation: `{well}`, `{cell_line}`, `{treatment}`,
  `{role}`, `{field}`, `{run}` and custom fields.
- **Crop to the view** crops every panel to the region shown (synchronized crop, e.g. for
  insets); **No crop** restores the whole field.

## Figure export

PNG or TIFF (300 or 600 dpi), SVG or PDF. Images are embedded unresampled; text stays
text (editable; TrueType in PDF) and scale bars are vector shapes.

| Setting | Meaning |
|---|---|
| width | single column 85 mm, 1.5 column 120 mm, double column 180 mm |
| scale bar | length in µm (auto: 1, 2 or 5 × 10^k near 20 % of the panel), position, colour, thickness, label; on the last or every panel |
| split channels | each channel in grayscale with a merge, as in a typical figure panel |
| headers, legend | row/column headers from the annotation; channel names in their colours |
| font size, background, spacing | as in the export settings |
| stamp | Z plane (or "max Z") in each panel |

**Presets** save and reuse export settings (`~/.hcs_studio/export_presets/`).
**Batch: a figure per compound** writes one figure per compound of the layout
(concentrations as columns, cell lines as rows).

## Data and movies

- **Export data**: one OME-TIFF per panel (the shown plane, cropped; float32 or uint16)
  and `panels_readouts.csv` with the shown wells' per-well readouts.
- **Export movie (Z)**: GIF or MP4 scrolling through Z with all panels in step (the
  sources must have several planes; run outputs keep the first timepoint only).

## The sidecar

Each export writes `<file>.json`: the comparison (panels, fields, sources, display,
crop, settings), the source files with SHA-256, the processing steps, each panel's
position in the output (and the scale bar's length in output pixels) and the software
version. `hcs-studio figure --regenerate <file>.json` re-renders the figure; a PNG comes
out pixel-identical as long as the source files are unchanged (it refuses otherwise).

```bash
hcs-studio figure --regenerate figure.png.json --output figure_again.png
```

```bash
hcs-studio figure --run <run folder> --per-compound --format svg --output figures/
```
