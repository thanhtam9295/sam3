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

## Training

```bash
hcs-studio models train --kind steps --name hct116-nuclei --set hct116-nuclei --channel 1
hcs-studio models train --kind cellpose --name hct116-cp --set hct116-nuclei --base-model cpsam_v2 \
    --epochs 100 --learning-rate 1e-5 [--device gpu]
hcs-studio models train --kind classifier --name phase-clf --set phase-labels [--algorithm logistic_regression]
```

Every trainer learns on the set's **train** split, is scored on the **validation** split next to
its base model, and is registered as a *candidate* with the training set (name, version and
content hash), the hyperparameters, the validation metrics of both models and a summary of the
training objects (area, intensity, shape and density quantiles — used for drift alerts).

| Kind | What is learned | Where it runs |
|---|---|---|
| `steps` | a classical segmentation (threshold + watershed) tuned by grid search: threshold method, smoothing, minimum distance between centres (`--grid '{"method": ["otsu", "local"], …}'` to change the search) — scored by AP@0.5 summed over the train fields | any CPU, seconds |
| `cellpose` | Cellpose fine-tuned from a base model (`cpsam_v2`), epochs, learning rate, weight decay | a CUDA GPU in practice (a CPU works for tiny tests only) |
| `classifier` | an object classifier (random forest or logistic regression) from a classification set (`objects.csv`: `item`, `label` and numeric features) | any CPU |

In the app, training is submitted to the job queue (its own process; GPU jobs run alone); the
command line trains in the foreground. Image-QC classifiers (`hcs-studio qc-train`) and ilastik
projects (trained in ilastik) are registered with `hcs-studio models register`.

## The model registry

```bash
hcs-studio models list [--name N] [--kind K]
hcs-studio models show hct116-nuclei@3
hcs-studio models register model.joblib --name qc-field --kind qc_classifier [--training-set S@V]
hcs-studio models status hct116-nuclei@3 validated --reason "AP@0.5 0.91 on test plates"
hcs-studio models status hct116-nuclei@3 production --reason "replaces v2" --yes
hcs-studio models pin pipeline.yaml
```

Each version is copied into `<settings>/models/<name>/vNNNN/` (read-only, never changed) and
recorded with its kind, base model, training set and hash, hyperparameters, metrics, author,
date, status and notes. Statuses: **candidate → validated → production → retired** (a validated
version can go back to candidate; a retired one can be reconsidered as a candidate). Promotion
to production needs a reason and a confirmation (`--yes`; a dialog in the app) and moves the
previous production version back to *validated*. Every change is a service event (who, when,
why).

### Models in pipelines

Steps name registered models as `registry:NAME@VERSION`:

| Step | Parameter |
|---|---|
| Cellpose (`segment_cellpose`) | `model` |
| Registered segmentation (`segment_model`) — runs a `steps` model exactly as stored | `model` |
| Classify objects (`classify_objects`) | `model_path` |
| Learned image QC (pipeline QC rules) | `learned.model` |

`registry:NAME@production` (or no version) is allowed while editing, but `hcs-studio models pin`
replaces it by the exact version before the pipeline is used for runs — a saved pipeline then
always gives the same masks. A reference to a missing, retired or altered model (checksum
mismatch) stops the run with a message.

The run manifest (`run_manifest.json` → `models`) lists every model the pipeline used: step,
module, parameter, value, file (as a data-root path), SHA-256 and, for registered models, the
name, version, kind, status and training set. Built-in Cellpose models are hashed from the
Cellpose model folder when the weights are present.

## Comparing a candidate on held-out fields

```bash
hcs-studio models compare hct116-nuclei@3 [--baseline hct116-nuclei@2] [--set S@V] [--split test]
```

The candidate and a baseline — the current production version of the same name, else the
model it was trained from — segment the **test** split of the candidate's training set (fields
training never saw; another set can be given). The report (`comparison.html` in the model's
`reports/` folder, or `--output`) shows the metrics side by side, overall and per cell line, AP
against the IoU threshold, count errors per field and the worst fields with the true and
predicted outlines. The candidate's registry metrics get the test AP and the report path.

## Downstream impact on a reference plate

```bash
hcs-studio models impact <reference run> hct116-nuclei@3 --output impact/ [--step 1] [--msr 3]
```

The reference run's plate is processed again with the candidate in place of the model it
used (the steps that reference another version of the same name, or `--step N`), with the same
pipeline and layout otherwise, and the two runs are compared:

- **counts per well** (`<labels>_count`): % differences, Bland–Altman mean and limits of
  agreement (mean ± 1.96 SD), largest change;
- **potency**: IC50 of every compound × cell line in both runs (4PL on % inhibition of the
  count); a ratio beyond the compound's **MSR** — from the registry's potency history (Phase 17)
  when there is one, else `--msr` (3) — is a significant change.

The verdict is *equivalent* when no potency changes significantly and the mean count difference
is within ±10 % (`--count-tolerance`); `impact.html`, `impact.json`, `counts.csv` and
`potency.csv` are written to the output folder.

## Ongoing QA: spot checks and drift

Add the run step `model_qa` to production pipelines. After each run it

- queues a random sample of fields (3 by default) for a **human spot check**: in the Models
  tab, each pending field shows its preview and outlines; accept it, or flag it with what is
  wrong. Accept rates per model show real-world accuracy over time;
- checks **drift**: the run's objects (area, intensity of the segmented channel, eccentricity,
  objects per field) are compared with the same features of the model's own masks on its
  training fields. A feature whose median moved more than 1 IQR (`drift_threshold`), or whose
  spread changed more than twofold, is reported in `qa/model_qa.json` and in the run log, with
  the suggestion to annotate a few fields of that plate and check or fine-tune the model.

```bash
hcs-studio models spot-check queue <run> -n 3      # for runs without the model_qa step
hcs-studio models spot-check list [--status pending]
hcs-studio models spot-check review 12 flagged --reason "clusters merged"
hcs-studio models spot-check summary
hcs-studio models drift <run> [--model NAME@VERSION] [--threshold 1]
```

## The Models tab

Lists every model version (status, validation and test scores, training set, author). Select a
version to change its status (a reason is required; production asks for confirmation), compare
it on held-out ground truth (in the background; the report opens with *Open report*), and review
the pending spot checks below.
