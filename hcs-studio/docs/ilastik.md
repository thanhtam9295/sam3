# ilastik in HCS Studio

ilastik trains pixel and object classifiers interactively from a few brush strokes, and it
copes well with difficult stains, organoids and textured phenotypes. You train a project
(`.ilp`) in the ilastik program; HCS Studio then applies it to whole plates in batch, with the
same job queue, resume, provenance and analysis as Cellpose or SAM 3.

**Status:** step (a) of Phase 16 is built: settings, detection, and the HDF5 and axis-order
conventions ilastik needs. Running projects (pixel classification, autocontext, object
classification, multicut, tracking) and the Fiji bridge follow in the next steps.

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
| Default runner | headless (ilastik directly; recommended) or fiji (through ilastik4ij) |
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
