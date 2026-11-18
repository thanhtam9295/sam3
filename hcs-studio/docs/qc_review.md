# QC review of Phases 0–15 and test-data plan

Date: 2026-11-18. Covers what was checked in the cloud container, what is still open in
each phase, what real data is needed to close it, and how to send that data.

## 1. QC run in the container

### Automated tests, phase by phase

Each phase's test files run separately under `xvfb-run` (Linux, 4 CPU cores, no GPU).

| Phase | Area | Tests | Result |
|---|---|---|---|
| 0 | Scaffold, main window, GPU detection | 15 | pass |
| 1 | Readers, data model, browser | 42 | pass |
| 2 | Correction, enhancement, pipeline schema | 42 | pass |
| 3 | Deconvolution, PSF tool | 24 | pass |
| 4 | Segmentation | 25 (+2 skipped: need the SAM 3 env) | pass |
| 5 | Measurements, colocalization | 23 | pass |
| 6 | Batch, jobs, pipeline builder | 20 | pass |
| 7 | Results plate, `index.html` | 7 | pass |
| 8 | Gates, classifier, clustering, stitching | 30 | pass |
| 9 | Profiles, PCA/UMAP | 9 | pass |
| 10 | Hardening, acceptance commands | 17 | pass |
| 11 | Plate annotation | 23 (+1 added in this review) | pass |
| 12 | Projection, background | 19 | pass |
| 13 | Pharmacology | 18 (+2 skipped: no Prism / SynergyFinder files) | pass |
| 14 | Comparison, figure export | 19 | pass |
| 15 | End-to-end regression plate | 5 | pass |

Full CI (ruff, ruff format, mypy --strict, all tests): 338 passed, 4 skipped.

### The command-line tools, end to end on synthetic data

| Phase | Tool run | Result |
|---|---|---|
| 0 | `hcs-studio info`, `selftest`, `gui --smoke-test` | all pass (9/9 self-test checks) |
| 1 | `inspect` on Operetta, ImageXpress (Z series), OME-Zarr and a 96-well plate | all read; channels, pixel size, Z, wells correct; notes shown for missing Z step / modality |
| 3 | `acceptance beads` (5 beads, measured PSF) | PASS: FWHM x/y 261 → 145 nm, z 648 → 285 nm |
| 3 | `benchmark --shape 16 512 512` | 1.14 s per iteration on CPU, 1.1 GB |
| 6 | `run` 96 wells, flat-field → segmentation → cells → measure → colocalization → QC | 96/96 wells, 121 s with 3 workers |
| 7 | `acceptance run`, `report` | PASS: all items; 480 linked previews exist; report self-contained |
| 8 | `analyze` (gate, phenotypic clusters, spatial clusters) | applied, tables and report rebuilt |
| 9 | `profile --level well` | 96 wells, 71 features, PC1–3 = 73/10/5 % |
| 10 | `acceptance kill-resume` (killed after 20 wells) | PASS: resumed, 96/96 complete, no partial files |
| 11 | `layout --validate --export --plate-map`, `annotate` | **failed before the fix below**; now passes |
| 12 | `run` projection + rolling-ball background on Operetta and ImageXpress Z stacks | completed, no warnings |
| 13 | `run` + `pharm` on the 384-well known-answer plate | EC50 within 7 % of truth for all 6 curves; A×B synergistic (Bliss +11.6, Loewe +12.3, ZIP +11.4), A×C additive (Bliss +0.4) |
| 14 | `figure --per-compound`, `figure --regenerate` | 4 figures; regenerated PNG pixel-identical |
| 15 | `analyze --only` with the three run steps; `acceptance run` | all run steps completed; PASS (308 wells) |

### Fixed in this review

- **Layout validation failed every single-cell-line plate.** A layout that records no cell
  line anywhere (common for one-line screens and for older layout files) was reported as an
  *error* and `hcs-studio layout --validate` exited 1. It is now a *warning* ("all wells are
  analysed as one cell line"); a plate where only some wells lack a cell line is still an
  error. Test: `test_plate_without_any_cell_line_is_a_warning_not_an_error`.

## 2. Open issues and what is needed, phase by phase

Everything above is verified on **synthetic data with a known answer only**. No module has
been compared with vendor or reference software on real instrument data yet. The items
below are what stands between "built" and "validated".

### Phase 0 — Scaffold and installation

Issues: the Windows installer and the CUDA build have never been run here (Linux container,
no GPU).

Need from you: on the analysis workstation, the output of `hcs-studio info` and
`hcs-studio selftest` (and `hcs-studio gui --smoke-test`); OS, GPU model, driver/CUDA
version, RAM. If you use the installer: build it with the `hcs-studio-windows` GitHub
workflow (or `packaging\build_windows.ps1`), install, and send the same three outputs from
the installed `hcs-studio.exe`.

### Phase 1 — Readers and browser

Issues: readers are tested on synthetic vendor layouts only (no real `.nd2` was reachable).
RGB `.nd2` is not supported. ImageXpress does not record confocal vs widefield, its field
positions are nominal, and its Z step may be missing from the TIFF tags. Operetta
flat-field profiles are not read.

Need from you, per instrument (Nikon, Operetta CLS, ImageXpress):
- A small but **complete-structure** export: Nikon — one `.nd2` file (a Z stack, several
  channels); Operetta — the whole `Index.idx.xml` (or `Index.xml`) plus the image files of
  2–3 wells; ImageXpress — the `.HTD` file plus the TIFFs of 2–3 wells, including the
  `ZStep_*` / `TimePoint_*` folders if used.
- The values the vendor software shows for that plate: wells, fields per well, channel
  names and wavelengths, pixel size, Z step and planes, objective, NA, immersion. A
  screenshot is fine.

### Phase 2 — Correction and enhancement

Issues: flat-field and the three autofluorescence methods are verified on synthetic
shading only; flat-field accuracy drops on dense cytoplasm stains (RMS 4–5 %). Every field
is processed whole in memory (fine for 2048 × 2048).

Need from you: one plate (or 20–50 fields) with sparse cells **and** either unstained wells
or an AF channel; a flat-field reference image (e.g. fluorescent slide) if you have one.

### Phase 3 — Deconvolution

Issues: CPU deconvolution of full-size stacks is slow (1.1 s per iteration for
512 × 512 × 16; a 2048 × 2048 × 30 stack needs a CUDA GPU in practice, not yet measured).
The automatic pinhole conversion is unchecked against NIS-Elements/Harmony Airy units.
No depth-variant PSF; no automatic stopping rule.

Need from you: a bead Z stack (100–200 nm beads, isolated, Nyquist-sampled, same objective
and settings as the cell images) with NA, emission wavelengths, immersion and pinhole;
optionally Huygens results on the same stack; `hcs-studio benchmark --device gpu` output
from the GPU workstation. Run `hcs-studio acceptance beads <stack>` and send the JSON.

### Phase 4 — Segmentation

Issues: Cellpose counts are not yet checked against manual counts on real fields
(acceptance ±5 %). SAM 3 was checked on CPU only, is unreliable on dense small nuclei and
sensitive to prompt wording, and is 2D only. `cpdino` models need the `dinov3` package
(not included). The spot detector threshold is unchecked on PLA images.

Need from you: manual counts on 10 fields (`counts.csv`: `well,field,count`, see
`docs/acceptance.md` §4) from a finished run; for SAM 3, a few spheroid/organoid and PLA
fields with manual outlines (Fiji ROI sets) and the phrases you would use; manual spot
counts in a few PLA cells.

### Phase 5 — Measurements and colocalization

Issues: colocalization is not yet compared with Fiji Coloc 2 on real fields (acceptance:
Pearson and Manders within ±0.02); the Costes threshold search and p-value convention are
documented differences. The focus-QC threshold is unchecked on real out-of-focus fields.

Need from you: 3 real fields exported with *Export for Fiji…*, with `coloc2_compare.ijm`
run in Fiji (each folder then holds the inputs and Fiji's log); then
`hcs-studio acceptance coloc <folders>`. A plate with a few known out-of-focus fields
(tell me which).

### Phase 6 — Pipelines and batch

Issues: a full 384-well run has been timed on synthetic fields only (48 s for small
fields; 3.4 GB plate profiled in Phase 10).

Need from you, on the workstation: `hcs-studio acceptance kill-resume` on a real 384-well
plate with your pipeline (`docs/acceptance.md` §3). Send the JSON report and the run's
`run_manifest.json` (it contains the timings).

### Phase 7 — Results plate and report

Issues: the offline click-through of `index.html` on another computer is a manual step.

Need from you: `hcs-studio acceptance run <run folder>` JSON, then open the zipped run on
another computer with the network off and tell me anything that does not open.

### Phase 8 — Gates, classifier, clustering, stitching

Issues: stitching corrects translation only and is not compared with vendor stitching;
classifier accuracy is unchecked on a real screen; plate robust-Z assumes most objects are
alike (use no normalisation on plates dominated by one phenotype).

Need from you: one well stitched by Harmony or NIS-Elements plus the raw fields of that
well; for the classifier, a run with labelled objects (`well,field,object_id,label`) or
control wells with known phenotypes, keeping some wells back for testing.

### Phase 9 — Profiles, PCA/UMAP

Issues: only synthetic profiles; with several plates, profiles are saved with the first
plate's run only.

Need from you (optional): a plate with reference compounds of known mechanism in
replicate, and which ones you expect to group together.

### Phase 10 — Hardening and packaging

Issues: see Phase 0 (installer, CUDA) and Phase 6 (real 384-well timing). An intermittent
hang of the GUI test suite was seen once and did not recur in three repeated runs; pytest
now prints every thread's stack after 180 s, so a recurrence will show where it hangs.

Need from you: as in Phases 0 and 6.

### Phase 11 — Plate annotation

Issues: Echo/D300 dispense-file import is deferred until a sample file exists.

Need from you: your real layout files for one dose–response plate and one combination
plate (CSV/XLSX, long or plate-map format, as you use them today); an Echo or D300 dispense
file (CSV) with the stock concentrations and the assay volume per well.

### Phase 12 — Projection and background

Issues: the BaSiC-style flat/dark-field estimate is a per-pixel regression, not the
published L1/low-rank method; a dark-field shaped like the vignetting cannot be separated.
Not yet compared with Fiji or `basicpy`; runtime on 2048 × 2048 fields not measured.

Need from you: 3–5 real Z stacks (Operetta confocal or Nikon) and, from Fiji on the same
fields, *Z Project → Max* and *Subtract Background* (25 µm, rolling ball and sliding
paraboloid) saved as 32-bit TIFF; optionally a BaSiC result (Fiji plugin or `basicpy`)
for one plate.

### Phase 13 — Pharmacology

Issues: fits and synergy scores are verified on synthetic data only. MuSyC needs the
optional `synergy` package (not added; your decision). Cosmetic: SSMD shows `nan` when both
control groups have zero spread (noise-free synthetic data only).

Need from you: 5 dose–response curves fitted in GraphPad Prism (XY table + Prism's logEC50
and Hill slope) and one 6 × 6 matrix scored in SynergyFinder (matrix + Bliss/HSA/Loewe/ZIP
+ version and options). The formats are in `tests/data/pharmacology/README.md`; the
comparison tests run as soon as the files are there. Tell me whether to add MuSyC.

### Phase 14 — Comparison and figure export

Issues: no figures from real plates checked against journal requirements; channel
colours are user-chosen (colour-blind safety is not enforced).

Need from you: the target journal(s) or a figure you consider correct, and your preferred
channel colours; a time-lapse plate if you want time movies checked on real data.

### Phase 15 — Integration and reports

Issues: the full chain (run steps, experiment report, methods text) has only run on the
synthetic regression plate.

Need from you: one complete real experiment (plate + layout) to run end to end, and your
corrections to the methods text and the report contents.

## 3. How to send data

Use the route that fits the size. Do not paste tokens or passwords into the chat.

| What | Size | Route |
|---|---|---|
| Text results: `acceptance_*.json`, `inspect` output, `counts.csv`, logs, `run_manifest.json`, Prism/SynergyFinder exports, layout files, Echo/D300 file | KB–MB | Paste text in the chat, or commit them to the repository (`thanhtam9295/sam3`) on any branch and tell me the path. Reference files can go straight into `hcs-studio/tests/data/pharmacology/`. |
| Image subsets: 2–3 wells of a plate, one `.nd2`, bead stack, 3–10 fields | MB–10 GB | A **private Hugging Face dataset** (this session's environment already allows huggingface.co with your token). See below. |
| Full plates, GPU timing, the Windows installer | 10 GB and more, or needs your hardware | Run the acceptance command on your workstation and send the JSON reports (they contain no image data), **or** connect the workstation to a session with Remote Control so I can run the commands there. |

**Hugging Face upload** (on your computer, with the `huggingface_hub` package):

```bash
hf auth login
```

```bash
hf repo create hcs-test-data --repo-type dataset --private
```

```bash
hf upload <your-hf-user>/hcs-test-data ./operetta_subset operetta_subset --repo-type dataset
```

Then tell me the repository name and the folder. The token saved in this environment must
be allowed to read that repository (if it is a fine-grained token, add the repository to
it). Keep each batch under about 10 GB: this session has about 15 GB of free disk.

**Remote Control**: in the Claude desktop app on the workstation, Settings → Claude Code →
Remote Control, add the folder with the data (not the home folder). The folder then
appears to this session and I ask to use it; you approve the request on your side.

Do not send images that contain patient identifiers. Raw data is only ever read; outputs go
to new run folders.

## 4. Suggested order

1. Phase 1 readers (one subset per instrument + vendor values): everything else depends on
   reading the files correctly.
2. Phase 4 counts and Phase 5 Coloc 2 comparison (the §10 acceptance numbers).
3. Phase 13 Prism and SynergyFinder references; Phase 11 layout files and Echo/D300 file.
4. Phase 3 beads and GPU benchmark; Phase 12 Fiji comparison.
5. Phases 0, 6 and 10 on the workstation: installer, full 384-well kill-resume and timing.
6. Phases 8, 9, 14 and 15 on a complete real experiment.
