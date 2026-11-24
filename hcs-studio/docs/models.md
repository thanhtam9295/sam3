# Training, validating and managing models

Segmentation and classification models are only as good as their fit to *your* cells. This
page covers ground-truth sets, how models are scored against them, the model registry and the
ongoing checks that keep track of real-world accuracy.

## Ground-truth sets

A ground-truth set is a versioned folder of annotated fields in
`<settings>/annotations/<name>/vNNNN/`: images, instance masks (segmentation) or labelled
objects (classification), and `set.json` with the provenance of every field — plate, well,
field, source run, cell line, stain, annotator, date, how the labels were made, pixel size.

```bash
hcs-studio gt import <folder of *_img.tif / *_masks.tif> --name hct116-nuclei \
    --cell-line HCT116 --stain Hoechst --annotator anna --pixel-size 0.65 [--provenance p.csv]
hcs-studio gt split hct116-nuclei --by well --fractions 0.6,0.2,0.2 --seed 0
hcs-studio gt show hct116-nuclei          # items per split and cell line, content hash
hcs-studio gt list
hcs-studio gt export hct116-nuclei@2 --split train --output pairs/
```

The folder is what the segmentation editor's *Save as training data* writes (Cellpose
`*_img.tif` / `*_masks.tif` pairs); a provenance CSV (column `id` or `file`, plus any of `plate`,
`well`, `field`, `z`, `run`, `cell_line`, `stain`, `annotator`, `date`, `source`,
`pixel_size_um`) fills in what the file names do not say (a well like `B03` is read from the
name).

- **Versions never change.** Corrected masks, new fields or a new split make a new version
  (`--based-on NAME@VERSION` keeps the earlier items). The content hash covers images, masks and
  provenance — not the split — so a model records exactly what it was trained on.
- **Splits are by plate or well** (or cell line), never by field: fields of one well share cells,
  focus and staining, and a field-level split overstates accuracy. Whole groups are assigned to
  train / validation / test; `--test-group` sends a plate, a well (`PLATE|WELL`) or a cell line to
  test whatever the fractions (e.g. to test on a cell line the model has never seen).

## Validating a segmentation

```bash
hcs-studio validate-seg --set hct116-nuclei --split test --pipeline seg.yaml [--output val/]
```

The pipeline's image steps up to its segmentation run on every field of the split (steps that
need a plate-level fit, and measurements, are not allowed / skipped), and the result is
compared with the masks:

| Metric | Definition |
|---|---|
| AP at IoU t | TP / (TP + FP + FN), objects matched one-to-one with IoU ≥ t (as Cellpose and StarDist report it); counts summed over fields |
| mAP | mean AP over IoU 0.50, 0.55 … 0.95 |
| precision, recall, F1 | at IoU 0.5 |
| merges | a predicted object holding more than half of ≥ 2 true objects (one per extra object) |
| splits | a true object holding more than half of ≥ 2 predicted objects (one per extra piece) |
| count error | predicted − true objects per field (and %) |

Everything is reported overall and per cell line (`metrics.json`, per field `fields.csv`).

Classifiers are scored with balanced accuracy, per-class precision and recall, the confusion
matrix and calibration (reliability bins, expected calibration error, Brier score).
