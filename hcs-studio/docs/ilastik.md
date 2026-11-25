# ilastik in HCS Studio

ilastik trains pixel and object classifiers interactively from a few brush strokes, and it
copes well with difficult stains, organoids and textured phenotypes. You train a project
(`.ilp`) in the ilastik program; HCS Studio then applies it to whole plates in batch, with the
same job queue, resume, provenance and analysis as Cellpose or SAM 3.

**Status:** Phase 16 is built: settings and detection, Pixel Classification, Autocontext,
Object Classification (including Cellpose objects), Multicut, Tracking, the Fiji runner and
Fiji macros. Tested with stand-in programs; see *Validation status* at the end.

## Installing ilastik

- **Windows / macOS / Linux:** install the latest **stable** release from ilastik's download
  page (tested: **1.4.2**). On Windows it installs to `C:\Program Files\ilastik-1.4.2\`.
- **Linux, conda packages** (as in HCS Studio's cloud tests): ilastik is on the `ilastik-forge`
  channel, not on conda-forge. Version 1.4.2's package misses one dependency and pulls an
  incompatible tracking library, so pin both:

  ```bash
  micromamba create -y -p ~/envs/ilastik --channel-priority flexible -c ilastik-forge -c conda-forge "ilastik=1.4.2" yapsy "hytra=1.1.5"
  ```

  Then choose `~/envs/ilastik/bin/python` as the ilastik program (HCS Studio runs
  `python -m ilastik`).

ilastik always runs as a separate program. It is never installed into HCS Studio's own
environment: its pinned dependencies conflict.

### Versions

| Version | In HCS Studio |
|---|---|
| 1.4.2 or newer stable | supported (1.4.2 is the tested release) |
| 1.4.0, 1.4.1 | supported |
| pre-releases (`a`, `b`, `rc`), e.g. 1.5.0b1 | allowed, with a warning |
| 1.3.3post3 | allowed, with a warning (old) |
| 1.3.3b2, 1.3.3rc2, 1.3.3, 1.3.3post1, 1.3.3post2, 1.4.0b27 | **blocked** (also incompatible with Fiji's ilastik4ij plugin) |
| older than 1.3.3post3 | **blocked** |

### Fiji (optional)

Only the Fiji runner needs Fiji: install Fiji, then *Help → Update → Manage update sites*,
tick **ilastik**, apply and restart Fiji. HCS Studio finds the `ilastik4ij*.jar` plugin in
`Fiji.app/jars` or `Fiji.app/plugins`.

## Settings

**Tools → ilastik settings…**

| Setting | Meaning |
|---|---|
| ilastik program | `ilastik.exe`, `run_ilastik.sh` / `run_ilastik.bat`, or the Python of an ilastik environment. Empty: found automatically in the usual install folders. |
| Threads | threads per ilastik process; *no limit* lets ilastik decide (passed as `LAZYFLOW_THREADS`) |
| RAM limit | memory per ilastik process; *no limit* or at least 500 MB (ilastik refuses less; passed as `LAZYFLOW_TOTAL_RAM_MB`) |
| ilastik processes at once | default 1: ilastik is already multithreaded and memory-hungry |
| Default runner | headless (ilastik directly; recommended) or fiji (through ilastik4ij; see below) |
| Set ilastik's path, threads and RAM in the plugin | Fiji runner: run the plugin's *Configure ilastik executable location* first (default on) |
| Fiji | the Fiji folder (optional) |

**Detect / check installation** starts ilastik once to read its version (this takes a few
seconds) and shows which runners are available. Settings are saved in
`~/.hcs_studio/ilastik.json`.

The same from the command line:

```bash
hcs-studio ilastik --detect
```

```bash
hcs-studio ilastik --set-ilastik "C:\Program Files\ilastik-1.4.2\ilastik.exe" --threads 8 --ram-mb 16000 --save
```

The exit code is 0 when ilastik is usable, 1 when its version is blocked, 2 when it is not
found. `HCS_ILASTIK` and `HCS_FIJI` (environment variables) point to an installation without
saving settings.

## Data handed to ilastik

ilastik reads HDF5 block by block, so HCS Studio converts each processing stage to HDF5
before calling ilastik:

- one dataset `/data`, gzip-compressed, chunked (one timepoint, all channels, up to 32 planes
  and 256 × 256 pixels per chunk);
- an `axistags` attribute that tells ilastik the axis order, in the same form ilastik writes
  itself (`{"axes": [{"key": "y", "typeFlags": 2, "resolution": 0, "description": ""}, …]}`).

HCS Studio's images are (T, C, Z, Y, X). ilastik's axis order uses the letters `t z y x c`,
for example `yxc` (2D), `zyxc` (3D), `tyxc` (2D time-lapse), `tzyxc`. Axes of length 1 are
left out, except the channel axis. A project only accepts data with the same dimensionality
and the same channels, in the same order, as its training data: a 2D project needs a Z
projection step first for Z stacks.

ilastik's own results come back as HDF5 (`exported_data`) and are read with their axis order.

Checked with ilastik 1.4.2: files written by HCS Studio in all four layouts (2D single
channel, 2D multichannel, 3D multichannel, 2D + time) go through ilastik's Data Conversion
workflow and come back identical, with the same axis order (`tests/test_ilastik.py`).

## Pixel Classification and Autocontext in a pipeline

Train a Pixel Classification (or Autocontext) project in ilastik on images at the **same
processing stage** the pipeline will send (e.g. after Z projection and background removal),
then add the steps:

| Step | What it does |
|---|---|
| `ilastik_pixel` / `ilastik_autocontext` | runs the project on every field; adds one channel per class, `ilastik_<class>` (probabilities 0–1, or 0/1 masks with `export: segmentation`) |
| `probabilities_to_objects` | a class channel → objects: threshold (optional hysteresis), holes filled, touching objects split by distance transform or by seeds (e.g. nuclei), size filter |
| `measure_objects` | as usual; the class channels can be measured too (mean class probability per object) |
| `ilastik_class_areas` | % of each field per class (`ilastik_area_pct_<class>`, averaged per well; a plate readout for pharmacology) |

```
hcs-studio ilastik --info nuclei.ilp                     # workflow, classes, training axes
hcs-studio ilastik --preset nuclei.ilp --class nuclei --channels 1 --output ilastik.yaml
hcs-studio run --pipeline ilastik.yaml --input <plate> --wells A01-P24
```

- **One ilastik call per well.** ilastik takes many seconds to start, so all the fields (and
  time points) of a well go to one headless call (`--raw_data` with several files). Steps before
  the ilastik step run first for the whole well; measurement steps must come after it.
- **Channels:** `channels` chooses what is sent, in the trained order (default: all). A project
  trained on 2D data refuses Z stacks (add a projection step) and a different number of
  channels, when the project states it.
- **Class names** come from the project; give `class_names` if they are not found.
- **Projects** may be files or model-registry references (`registry:NAME@VERSION`). Headless
  ilastik opens projects read-only; HCS Studio checks the file's SHA-256 before and after every
  call and stops if it changed.
- **Resume:** a well that failed (e.g. ilastik ran out of memory) is redone on resume; finished
  wells are not.

Run folder:

```
<run>/ilastik/projects_used.json   # project path, SHA-256, workflow, ilastik version, runner
<run>/ilastik/logs/<well>.log      # the exact command lines and ilastik's output
<run>/ilastik/inputs/              # HDF5 sent to ilastik (deleted after each well unless keep_inputs)
<run>/wells/<W>/ilastik/f001_t0_Probabilities.h5 …   # ilastik's own outputs, per field
<run>/wells/<W>/ilastik/class_summary.json            # class areas per field, command, time
```

The run manifest's fitted state records the project SHA-256, ilastik's version and the runner.

## Object Classification (including Cellpose objects)

Classify objects with a trained ilastik **Object Classification** project — for example
phenotypes of Cellpose nuclei:

| Step | What it does |
|---|---|
| `ilastik_objects` | runs the project once per well; each object gets the class most of its pixels have in ilastik's "Object Predictions" |
| `ilastik_object_classes` (table step) | adds `class_ilastik` (class name) and `class_ilastik_agreement` to the objects; per well `class_ilastik_<class>_count` and `_fraction` |

```
hcs-studio ilastik --preset phenotype.ilp --channels 1 2 --output phenotype.yaml
# Cellpose nuclei → ilastik Object Classification → measure objects → QC → class fractions
```

- **input: labels** (default) — for projects trained *from a label image*: the step sends an
  existing label set (`labels`, e.g. Cellpose `nuclei`) as ilastik's label image, so the
  objects keep their HCS Studio IDs and the classes join the object tables directly.
- **input: prediction_maps / segmentation_image** — for projects trained on probabilities or a
  binary mask: put `ilastik_pixel` first; ilastik makes its own objects from the class channels
  (`source_channels`). This needs two ilastik calls per well (object identities, then
  predictions); the objects become the label set `ilastik_objects`.
- The class fractions are ordinary per-well readouts: plate map, report, and pharmacology
  (e.g. `--readout class_ilastik_mitotic_fraction`; dose–response on the share of a phenotype).
- Several ilastik steps in one pipeline (pixel classification, then objects) each run once per
  well, in order; measurement steps come after the last one.
- Per well: `wells/<W>/ilastik/object_classes_ilastik.parquet` (field, time point, object,
  class, agreement) and `.json` (class totals), and ilastik's prediction images.

## Multicut (touching cells, dense tissue)

Train a Pixel Classification project with a **boundary** class (membranes, cell borders) and
a Multicut project on its boundary probabilities, then:

```
hcs-studio ilastik --preset multicut.ilp --boundary boundaries.ilp --output multicut.yaml
# ilastik_pixel (boundaries) → ilastik_multicut → measure objects → QC
```

`ilastik_multicut` sends the class channels of the pixel step as `--probabilities` (choose with
`source_channels`) and turns ilastik's "Multicut Segmentation" into the label set `cells`
(unique IDs), measured like any other segmentation.

## Tracking (time-lapse plates)

```
hcs-studio ilastik --preset tracking.ilp --output tracking.yaml
# Cellpose nuclei → ilastik_tracking → measure tracked objects → ilastik_tracks
```

- `ilastik_tracking` sends each field's **whole time series** in one call, with a binary mask
  (default: from the `nuclei` label set; or a class channel ≥ 0.5) or the class probabilities
  (`input: prediction_maps`), as the project was trained. Two exports per well:
  "Object-Identities" (the objects of every time point → label set `tracked`) and the
  "CSV-Table" plugin (each object's track and parent track).
- `ilastik_tracks` turns them into the same results as HCS Studio's own tracking step (Phase
  22d): `tracking/tracks.csv` (one row per track: frames, speed, straightness, how it ends,
  parent, cycle time) and per-well `trk_tracks`, `trk_speed_um_per_h`, `trk_straightness`,
  `trk_divisions`, `trk_division_rate_per_h`, `trk_cycle_h`, `trk_lost_fraction` — usable in the
  kinetics report and pharmacology.
- ilastik's tracking CSV columns are matched by their usual names (`frame`, `labelimage_oid`,
  `track_id1`, `parent_track_id`, `lineage_id` and common variants); if a real export names them
  differently the step says which column it misses.

## The ilastik dock and "Train in ilastik"

The **ilastik** dock (next to *Segmentation*) works on the field shown in the viewer:

- **Project**: a `.ilp` file, or a project from the **library** (the model registry, kind
  `ilastik`; see the Models tab). The dock shows the workflow, classes, the ilastik version that
  saved it and the training data's layout, and a badge: ✓ when the project fits this field
  (dimensionality and channel count), ✗ with the reason otherwise. *Channels* are the image
  channels sent to ilastik, in the order the project was trained.
- **Preview on this field** runs a Pixel Classification or Autocontext project on the field and
  shows one layer per class (`ilastik_<class>`).
- **Use in pipeline** loads the project's preset (ilastik → objects → measurements) into the
  pipeline builder. **Export Fiji macro…** and **Add to library…** as below.
- **Train in ilastik…** writes representative fields of the selected wells (all wells when none
  are selected) as ilastik-ready HDF5 with a `training_set.json`, then opens ilastik's own window
  on the project — a new project of the chosen workflow when the project field is empty. Add the
  files as raw data in ilastik, train, save and close ilastik: HCS Studio then offers to add the
  saved project to the library, with its classes, training layout and the training set it came
  from.

Representative fields cover every cell line with each control role and the top dose of every
compound (from the plate layout); without a layout they are spread over the plate.

**Train on what the pipeline will send.** A project trained on raw images gives poor results on
background-subtracted or projected images. From a finished run, export the training set at the
stage its ilastik step sees (recomputed with the run's own plate-level estimates):

```bash
hcs-studio ilastik --training-set <run folder> --stage ilastik --output <folder>
```

(`--stage raw` or `step:<n>` for another stage; from a raw plate only `raw`;
`--layout-file` adds cell lines and controls for raw plates.)

**Run a project on a plate in one line** (its preset pipeline; `--runner fiji` for one run
through Fiji without changing the settings):

```bash
hcs-studio ilastik --run nuclei.ilp --input <plate> --wells A01-P24 --channels 1 --runs <folder>
```

## The Fiji runner (ilastik4ij)

For labs that already run ilastik from Fiji macros, the same pipeline steps can run through
Fiji's ilastik4ij plugin: choose **fiji** as the default runner in Tools → ilastik settings
(or `hcs-studio ilastik --runner fiji --save`). Results are the same as with the headless
runner; the run folder records which runner ran each project
(`ilastik/projects_used.json`, `runner`), and `ilastik/logs/<well>.log` holds the exact macro.

How it runs: HCS Studio writes a macro (`wells/<W>/ilastik/fiji_<export>.ijm`) and starts
`<Fiji> --headless --console -macro <file>`. The macro sets ilastik's location, threads and
RAM in the plugin, then for each field imports the HDF5 input, runs the prediction command and
exports the result as HDF5. Things to know:

- The plugin starts ilastik once **per image** (the headless runner once per well), so the
  Fiji runner is slower. Use it for parity with existing Fiji workflows, not for speed.
- The plugin needs the ilastik program (`ilastik.exe`, `run_ilastik.sh`), not the Python of a
  conda environment.
- Without a RAM limit in the settings, the plugin's own default (4096 MB) is used: the plugin
  has no "unlimited".
- Running *Configure ilastik executable location* changes the plugin's saved options in that
  Fiji. If that command does not work without a window, untick the setting and configure the
  plugin once in Fiji (Plugins → ilastik → Configure ilastik executable location).
- What the plugin cannot do runs headless instead, with a note in the log: Object
  Classification on existing objects (Cellpose/SAM label images: the plugin offers only
  probabilities or a binary mask), and the Tracking step (HCS Studio needs ilastik's object
  identities and tracking table; the plugin returns the lineage image only).
- Fiji's exit code does not report macro errors; HCS Studio checks that the macro reached its
  end and that every output exists, and shows the end of Fiji's output otherwise.

## Fiji macros for collaborators

```bash
hcs-studio ilastik --macro nuclei.ilp --output nuclei.ijm
```

```bash
hcs-studio ilastik --macro objects.ilp --second segmentation --export identities --axes zyxc --output objects.ijm
```

The macro runs a project on every `.h5` file of a folder with the ilastik4ij plugin and saves
`<name>_<export>.h5` in an output folder. Opened in Fiji's script editor it asks for the
project and the folders; for batch use remove the `#@` lines and set the variables. Options:
`--export` (pixel/autocontext: `probabilities`, `segmentation`; objects: `predictions`,
`probabilities`, `identities`), `--second` (objects/tracking: `probabilities` or
`segmentation`; the second input of `<name>.h5` is `<name>_<second>.h5`), `--axes` (axis order
of the files; default: the project's training axes). When the ilastik settings name the
ilastik program, the macro also configures the plugin.

Inputs for the macro (or for training in ilastik) from a plate's raw fields:

```bash
hcs-studio ilastik --export-h5 <plate> --wells A01-B12 --channels 1 2 --output <folder>
```

writes `<well>_f<field>_t<t>.h5` (`--time-stack`: one file per field with every time point),
ilastik-ready and never inside the raw data. Processed inputs (after background subtraction,
projection, …) are the files an ilastik step keeps with `keep_inputs: true`
(`ilastik/inputs/<well>/`).

## Validation status

Tested with stand-in programs: `tests/fake_ilastik_headless.py` accepts ilastik's command line
and writes known answers, and `tests/fake_fiji.py` runs HCS Studio's Fiji macros the way the
ilastik4ij plugin does (its flags, its temporary files, its axis orders, as read in the
plugin's source). They cover one call per well, resume after a failed well, the read-only
check, and the parity test: the same jobs (pixel probabilities and segmentation, object
classification, multicut; 2D and 3D fields) through both runners give identical arrays, and a
whole pipeline gives identical per-well tables. The done-when test runs a projected 384-well
plate (24 wells in CI, all 384 with `HCS_SLOW=1`) through projection → ilastik → objects →
measurements, kills the run part-way, resumes it (finished wells are not sent to ilastik again)
and checks the counts against the synthetic truth and the run-folder structure.

**Not yet run with a real trained project, real ilastik or real Fiji**: please send one or two
`.ilp` files (Pixel Classification, Autocontext) with a few matching images. Then

```bash
HCS_ILASTIK=<ilastik program> pytest tests/test_ilastik.py tests/test_ilastik_steps.py
```

```bash
HCS_ILASTIK=<ilastik program> HCS_FIJI=<Fiji folder> HCS_ILASTIK_PIXEL_PROJECT=<project.ilp> pytest tests/test_ilastik_fiji.py -k real
```

check the real programs (the second is the headless-vs-Fiji parity test on a real project;
`HCS_ILASTIK_PIXEL_IMAGE=<file.h5/data>` uses your own image), and the project reader's guesses
(where the training axes and channels are stored) can be confirmed.
