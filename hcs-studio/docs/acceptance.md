# Acceptance tests on real data

PROMPT.md §10 lists what HCS Studio must do on real instrument data. Each criterion has a
command that checks it and prints PASS or FAIL per item; the report is also saved as JSON
(`acceptance_<check>.json`, or `hcs-studio acceptance --report <file> <check> …`). The
commands only read raw data and run folders. Exit code 0 means PASS, 1 FAIL, 2 the check
could not run.

| §10 criterion | Command |
|---|---|
| Plates from Nikon, Operetta, ImageXpress open with the right metadata | `hcs-studio inspect <plate>`, compared by eye with the vendor software (step 1) |
| Deconvolution narrows beads (Phase 3) | `hcs-studio acceptance beads <bead stack>` |
| 384-well plate runs unattended, resumes after a forced kill, §7 output folder | `hcs-studio acceptance kill-resume …` |
| Counts within ±5 % of manual counts on 10 fields | `hcs-studio acceptance counts --run … --manual counts.csv` |
| Pearson / Manders within ±0.02 of Fiji Coloc 2 | `hcs-studio acceptance coloc <exported folders>` |
| Clicking a well in `index.html` opens its previews offline | `hcs-studio acceptance run <run folder>`, then by hand (step 6) |

## 1. Readers and metadata

For each plate (`.nd2`, Operetta `Images/Index.idx.xml` export, ImageXpress folder):

```bash
hcs-studio inspect "<plate folder or file>"
```

Compare with the vendor software: plate format and wells, fields per well, channel names
and wavelengths, pixel size, Z step and number of planes, objective and NA. Then open the
plate in HCS Studio, open a field in the viewer and check the channels look right and the
scale bar matches. Anything that differs: send me the `inspect` output and the vendor's
values (do not change the raw files).

## 2. Beads and deconvolution

On the bead Z stack of the Nikon confocal (set the metadata first in the Metadata panel if
the file lacks NA, wavelengths or modality):

```bash
hcs-studio acceptance beads "<beads.nd2>" --channel 1
```

It finds isolated beads, deconvolves the stack (theoretical PSF from the metadata, or
`--psf <file>` saved by Tools → PSF tool from a separate bead stack) and compares the FWHM
of each bead at the same position before and after, without resampling. PASS when the
median FWHM is smaller after deconvolution along X, Y and Z. The numbers are what to
compare with Huygens on the same stack.

## 3. Full plate, unattended, killed and resumed

Save the pipeline you will use (Pipeline dock → Save) and run, on the analysis
workstation:

```bash
hcs-studio acceptance kill-resume --pipeline pipe.yaml --input "<384-well plate>" --output "<output root>" --layout layout.csv --kill-after 50
```

It starts the run as a separate process, kills it (and its workers) after 50 complete
wells, resumes it and checks the finished folder: status, every well complete, no failed
wells, no partial files, the §7 layout (`run_manifest.json`, `pipeline.yaml`,
`index.html`, `logs/run.log`, `wells/<W>/…`, `tables/…`), the report and the two attempts
in the run history. `run_manifest.json` → `performance` then holds the timings for your
machine.

## 4. Counts against manual counts

Count the nuclei (or cells) by hand on 10 fields of a finished run — for example in Fiji
with the multi-point tool on `wells/<W>/previews/fNNN_raw.png` or the raw image — and
write `counts.csv`:

```
well,field,count
B03,1,212
B03,2,198
...
```

`field` is the field number as in the run (`f001` = 1). Then:

```bash
hcs-studio acceptance counts --run "<run folder>" --manual counts.csv --object-type nuclei
```

PASS per field when |automatic − manual| ≤ 5 % of manual; the report also gives the mean
error, which shows whether the segmentation splits or merges systematically.

## 5. Colocalization against Fiji Coloc 2

In the Measurements dock, choose a field and region (object or whole field) on 3 real
fields and press **Export for Fiji…** for each (one folder each). In Fiji, run the
`coloc2_compare.ijm` of each folder (Plugins → Macros → Run…); it runs Coloc 2 on the
exported channels and mask and saves Fiji's Log as `coloc2_log.txt`. If your Fiji version
names Coloc 2's options differently, record one run with the Macro Recorder and replace the
`run("Coloc 2", …)` line (the macro explains how). Then:

```bash
hcs-studio acceptance coloc "<export 1>" "<export 2>" "<export 3>"
```

PASS when Pearson's R, M1, M2, tM1 and tM2 each agree within ±0.02. Li's ICQ and Spearman
are listed for information. Costes p is not compared (Coloc 2 reports 1 − p and uses a
different random block scheme).

## 6. The report, offline

```bash
hcs-studio acceptance run "<run folder>"
```

checks that `index.html` has no external resources and that every image it links exists.
Then by hand: zip the run folder, unzip it on another computer (or with the network off),
open `index.html`, click several wells and switch between enhanced, corrected and raw.

## What to send back

The JSON reports (`acceptance_*.json`), the `inspect` outputs, and for any FAIL the run
folder's `logs/run.log` and `run_manifest.json`. They contain no image data.
