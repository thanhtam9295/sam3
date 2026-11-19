# Validation notes

What has been verified, how, and what still has to be checked on real data.

## Status

All modules are verified only against **synthetic images with a known answer**
(`tests/`). None has yet been compared with vendor or reference software on real
instrument data. Treat quantitative results as provisional until the checks below
are done.

## Deconvolution (Phase 3)

### Verified on synthetic data

| Check | Result | Test |
|---|---|---|
| A blurred point source is re-concentrated, total flux conserved | flux within 1 %, > 90 % of it back in a 5×5 neighbourhood | `test_rl_reconcentrates_a_blurred_point_and_conserves_flux` |
| Bead stack (6 beads, Poisson noise, background 100): FWHM reduced in X, Y and Z | mean lateral FWHM 4.67 → 2.37 px after 40 iterations; every bead and axis below 0.6 × raw; per-bead flux within 10 % | `test_bead_stack_fwhm_is_reduced_in_3d` |
| Background estimate | within 5 counts of the true 100 | same test |
| Noise-free data converges to the exact answer | peak within 2 % after 200 iterations | `test_acceleration_converges_faster` |
| Biggs-Andrews acceleration | 25 accelerated iterations fit better than 50 plain ones | same test |
| TV regularisation limits noise amplification on a flat noisy image | SD < 0.6 × unregularised, mean preserved within 2 % | `test_tv_regularisation_limits_noise_amplification` |
| Tiled processing matches whole-image processing | max difference < 5 % of the peak, flux within 1 % | `test_tiled_result_matches_whole_image`, `test_automatic_tiling_respects_the_memory_budget` |
| PyTorch backend matches the numpy backend | within 2 % (run on CPU here; CUDA not tested) | `test_torch_backend_matches_numpy` |
| Theoretical widefield PSF, NA 1.4 oil, 520 nm | lateral FWHM 201 nm (Airy theory ≈ 190 nm); axial 484 nm (theory 0.88 λ / (n − √(n² − NA²)) ≈ 490 nm) | `test_theoretical_widefield_psf_matches_diffraction_theory`, `test_gui_psf.py` |
| Confocal PSF is narrower than widefield, laterally and axially | yes | `test_confocal_psf_is_narrower_than_widefield` |
| Measured PSF from a bead stack recovers the true PSF | FWHM within 6 % in X, Y and Z; edge and crowded beads rejected | `test_measured_psf_from_beads_recovers_the_true_psf` |

### Runtime

`hcs-studio benchmark --shape 30 2048 2048 --iterations 40 --device cpu` on the
development container (4 x86-64 CPU cores, 15 GB RAM, no GPU):

| Stack | Backend | Tiles | Runtime | Per iteration | Peak memory |
|---|---|---|---|---|---|
| 2048 × 2048 × 30, PSF 79 × 79 × 19 | numpy, CPU | 4 (automatic) | 1826 s (30.4 min) | 45.7 s | 7.0 GB |

The whole stack needs about 19 GB in one piece, so it was tiled automatically to stay
within half the free RAM. A 384-well plate with one such stack per well would take
about 8 days on this machine; this step needs a CUDA GPU (not yet measured) or
fewer iterations for plate-scale use. Run the benchmark with `--device gpu` on the
analysis workstation to get the real figure.

### Still to do on real data

1. **Bead stack**: deconvolve your bead PSF stack with (a) the measured PSF and
   (b) the theoretical PSF, and compare the bead FWHM with the raw stack and with
   Huygens on the same stack.
2. **Cell images**: deconvolve three confocal cell fields here and in Huygens with
   matched settings, and compare line profiles and integrated intensity per cell.
3. **GPU**: run `hcs-studio benchmark --device gpu` and one real field with
   `device: gpu`; compare with the CPU result (expected within 2 %).
4. Check the automatic pinhole conversion (pinhole µm ÷ magnification ÷ Airy
   diameter) against the Airy units shown in NIS-Elements or Harmony; the exact
   magnification at the pinhole plane differs between instruments.

### Known limitations

- The background is a single constant per channel and field. Uneven background
  should be removed first (flat-field correction; background subtraction only
  when it does not remove real signal).
- The PSF is assumed to be the same across the field and through the depth: no
  depth-variant PSF and no correction of spherical aberration with depth.
- The confocal model is psfmodels' excitation × pinhole-filtered emission model,
  not an instrument-specific model (e.g. spinning-disk pinhole crosstalk is not
  modelled).
- Richardson-Lucy has no automatic stopping rule tied to noise. Choose the
  iteration count on a representative field and keep it fixed across a plate.

## Segmentation (Phase 4)

### Verified on synthetic data

| Check | Result | Test |
|---|---|---|
| Threshold + watershed finds every nucleus, one ID each, correct area | 5/5, area within 15 % | `test_classical_finds_every_nucleus` |
| Watershed separates two overlapping nuclei | 2 objects (1 without watershed) | `test_watershed_separates_touching_nuclei` |
| Local threshold copes with a strong background ramp | 5/5 (global Otsu fails) | `test_local_threshold_handles_uneven_background` |
| Cells from nuclei keep the nucleus ID, grow by the requested distance, stop at the cytoplasm mask | yes, radius within 1.5 px | `test_cells_by_expansion_keep_nucleus_ids` |
| Spots found and linked to the right cell; spots outside cells dropped | 4/4 linked, 1 dropped | `test_spots_are_found_and_linked_to_cells` |
| Edge, size and intensity filters; children removed with their parent | yes | `test_filters` |
| **Real Cellpose `cpsam_v2`** (CPU) on 25 separated synthetic nuclei | 25/25, one ID per nucleus, areas within 20 % | `test_real_cellpose_counts_separated_nuclei` (runs when the weights are downloaded) |
| **Real Cellpose fine-tuning** (1 epoch, CPU) and use of the new model | trains in 44 s; the new model finds 36/36 nuclei | manual run, see the changelog |
| **Real SAM 3 code** through the worker (random weights, CPU): image encoding, text, box and point prompts | all four calls work; ~10 s per image, 1-2 s per prompt | `test_real_sam3_api_with_random_weights` (set `HCS_SAM3_PYTHON`) |
| SAM worker protocol, errors and crashes reported | yes (stand-in model) | `test_sam_worker_*` |

### SAM 3 with its trained weights (CPU)

Access to `facebook/sam3` was granted after Phase 4 was first delivered. Results on two
real fluorescence images shipped with scikit-image (no ground-truth masks; Cellpose
`cpsam_v2` is the reference):

| Image | Prompt | SAM 3 objects | Match a Cellpose nucleus (IoU > 0.5) | Median IoU |
|---|---|---|---|---|
| `cells3d` nuclei, plane 30 (256², 24 nuclei by Cellpose) | "cell nucleus" | 24 | 24 | 0.96 |
| same | "nucleus" | 0 | – | – |
| same | "cell" | 26 | 24 | 0.96 |
| same | "mitotic cell" | 22 | 22 (ordinary nuclei) | 0.96 |
| `human_mitosis` (512², 333 small nuclei by Cellpose) | any of the above | 0-2 | 0 | – |
| same, in 256-px tiles | "cell" | 192 | 184 | – |

Point and box prompts (synthetic nuclei, known masks): one click on a nucleus gives
IoU 0.93 with the true mask; a box used as an exemplar returns the boxed nucleus and
similar ones. On flat synthetic discs, "white circle" scores 0.94 but "nucleus" 0.002,
so text prompts are semantic, not shape detectors.

Conclusions:

- Use **Cellpose** for batch nucleus and cell segmentation. SAM 3's concept mode misses
  most objects in dense fields of small nuclei, even when tiled.
- Use **SAM 3** for interactive correction (point clicks are excellent), for larger
  objects, and for objects Cellpose was not trained on (spheroids, organoids).
- Concept prompts are **very sensitive to wording** ("nucleus" found nothing where "cell
  nucleus" found every nucleus). Try several phrasings on a representative field and keep
  the one that works.
- "mitotic cell" returns ordinary nuclei too: SAM 3 concepts do not classify phenotypes.
  Use the object classifier (Phase 8) for that.
- On CPU: model load ~20 s, image encoding ~5-7 s, each prompt ~1 s.

Regression test: `test_real_sam3_agrees_with_cellpose_on_real_nuclei` (set
`HCS_SAM3_PYTHON` and `HCS_SAM3_WEIGHTS=1`).

### Still to do on real data

1. **Cell counts**: compare Cellpose counts per field with manual counts on 10 annotated
   fields (acceptance: within ±5 %).
2. **SAM 3** on your own images: spheroids/organoids and PLA fields, with a few prompt
   phrasings, compared with manual outlines.
3. Check the spot detector threshold (`threshold_sd`) on PLA images against manual spot
   counts in a few cells.

### Known limitations

- SAM 3 needs a CUDA GPU upstream. HCS Studio's worker adds a small CPU compatibility
  layer: it redirects three hard-coded CUDA allocations and `pin_memory` to the CPU and
  runs under bfloat16 autocast. Results on CPU have not been compared with GPU results.
- SAM 3 segments 2D planes only.
- `cpdino` and `cpdino-vitb` need the `dinov3` package (installed from GitHub), not
  included here.

## Measurements and colocalization (Phase 5)

### Verified on synthetic data

| Check | Result | Test |
|---|---|---|
| Pearson of identical channels = 1, of inverted channels = −1; ICQ = 0.5; M1 = M2 = tM1 = tM2 = 1 | exact | `test_identical_and_anticorrelated_channels` |
| Pearson and Spearman equal scipy's | within 1e-12 | `test_statistics_match_scipy` |
| Manders with known overlap (A: 4 regions, B: 1 shared + 1 own) | M1 = 0.25, M2 = 0.5 exactly | `test_manders_known_overlap` |
| Costes threshold equals an independent brute-force search (first crossing from the top) | within one scan step (1/256 of the range); regression slope 0.7 recovered | `test_costes_threshold_matches_brute_force_and_separates_background` |
| Costes randomization: shared texture vs independent textures | p = 0.01 (99 randomizations) vs p > 0.05 | `test_randomization_separates_real_from_chance_correlation`, `test_coloc_module_per_cell_and_per_field` |
| Statistics restricted to the mask | r = 1 inside, −1 outside | `test_mask_restricts_the_statistics` |
| Chromatic shift of (2.3, −1.6) px detected and corrected, from given shifts or a bead image | within 0.15 px | `test_channel_shift_detection_and_registration` |
| Saturation and shift warnings | raised | `test_coloc_warnings_saturation_and_shift` |
| Object-based colocalization | 2/4 spots on nuclei; 3/4 within 20 µm | `test_coloc_parameters_and_object_based` |
| Disc and ellipse morphology in µm | area within 2 %, perimeter 5 %, axes 3-5 %, eccentricity 0.02 | `test_morphology_in_physical_units` |
| Sphere in an anisotropic stack | volume within 3 %, surface area 8 % | `test_3d_volume_and_surface` |
| Integrated intensity, percentiles, nuclear/cytoplasmic ratio (4.0), radial distribution | exact / > 90 % in the right ring | `test_intensity_radial_and_nuclear_cytoplasmic` |
| Spots per cell (1, 2, 3, 4), spot intensity, neighbour distances (20 µm grid) | exact | `test_spots_per_cell_and_neighbours` |
| Haralick contrast (rough ≫ smooth), granularity peak at the granule size | yes | `test_texture_and_granularity` |
| Focus score: blurred field ≪ sharp; independent of brightness | yes | `test_focus_score_and_plate_flags` |
| Out-of-focus, saturated, edge and low-count flags; counts exclude flagged fields; density per mm² | exact | `test_focus_score_and_plate_flags`, `test_run_writes_measurement_tables` |

### Choices that may differ from Fiji Coloc 2

- **Costes threshold search**: HCS Studio takes the *first* threshold, lowering from the
  maximum, at which pixels below both thresholds stop correlating (Costes et al. 2004).
  Coloc 2 bisects, which can stop at a different crossing when the correlation of the
  background fluctuates around zero. Both land inside the background's intensity range;
  tM1/tM2 can then differ by a few percent. "Below" means below both thresholds by
  default (`costes_below: and`); `or` is available if Coloc 2 turns out to differ.
- **Regression**: orthogonal (total least squares), as Coloc 2.
- **p-value**: reported as the probability of a random r at least as high as the observed
  one; Coloc 2's "Costes P-Value" is 1 − p.
- **Background**: subtracted before Manders (median outside the objects by default). For
  the comparison, export the inputs (Measurements dock → Export for Fiji) so Coloc 2 sees
  the same background-subtracted pixels.

### Still to do on real data

1. **Fiji Coloc 2 comparison on 3 real fields** (acceptance: Pearson and Manders within
   ±0.02): export each field with *Export for Fiji…*, run `coloc2_compare.ijm`, and send
   both result sets.
2. Check the focus threshold (half the plate median) on a plate with known out-of-focus
   fields.

## Batch execution (Phase 6)

| Check | Result | Test |
|---|---|---|
| 2 worker processes give the same objects and masks as 1 | identical tables and masks | `test_parallel_run_matches_sequential` |
| Cancel, then resume: completed wells skipped, saved plate-level estimates reused, result equal to an uninterrupted run | identical | `test_cancel_then_resume_skips_completed_wells` |
| CLI run killed with SIGKILL (workers included), then `--resume` | completes; finished wells untouched; equal to a clean run | `test_forced_kill_then_cli_resume` |
| **384-well plate**, `--workers auto` (3 here), killed after 122 wells, resumed | resumed run skipped 122 wells, processed 262, identical to a clean full run (2,300 objects, 1,150 colocalization rows, 384 well summaries); clean run 48 s | manual run, small synthetic fields |
| An unreadable image fails only its well; plate-level estimates skip it; `--retry-failed` reruns that well only | yes | `test_retry_failed_wells`, `test_job_queue_runs_retries_and_resumes` |
| Resume refuses a changed pipeline | yes | `test_resume_refuses_a_changed_pipeline` |
| GPU steps take the shared lock; SAM 3 forces one worker | yes | `test_gpu_steps_are_serialised` |
| Jobs in their own process: progress, pause, cancel, resume, interrupted job detected | yes | `test_job_process_runs_and_cancels`, `test_pause_cancel_resume_and_interrupted` |

Still to do: time a real 384-well plate on the analysis workstation (Phase 10).

## End-to-end regression plate (Phase 15)

`tests/test_regression.py` runs a synthetic 384-well plate with images through the whole
system in CI (about 45 s): segmentation, measurement, QC, the pharmacology, figure and
experiment-report run steps, and the clickable `index.html` in a browser (offline).
Each well's nuclei count follows the known truth (`pharmacology/synthetic.py`): 48 nuclei
in vehicle wells, 2 in positive controls, and 48 − 46 · I/100 for a true inhibition I;
nuclei are discs at least 3 px apart, so counts are exact.

| Check | Result (this version) |
|---|---|
| Nuclei count in all 384 wells | equal to the truth in every well |
| EC50 of six curves (four drugs, two cell lines) | within 7 % of the truth (limit 10 %); Hill slopes within 0.2 |
| HeLa vs MCF7 EC50 shift for Drug D (10×) | Δlog EC50 −1.00, F test p < 10⁻⁶ |
| Synergistic combination (A × B) | Bliss +11.6, Loewe +12.3, ZIP +11.4: "likely synergistic", p < 0.05 |
| Bliss-additive combination (A × C) | Bliss +0.45, ZIP +0.36: "likely additive" |
| Z′ | 1.0 (counts have no noise; real plates will be lower) |
| Run steps | all completed; the pipeline hash is the same without them |
| index.html | curve points and matrix cells open their wells; colouring by annotation; links |

`tests/data/regression/baseline.json` stores these values; the suite fails when a change
moves any of them by more than 2 % (regenerate deliberately with
`HCS_UPDATE_BASELINE=1 pytest tests/test_regression.py`).

## Comparison and figure export (Phase 14)

| Check | Result |
|---|---|
| 2 × 6 comparison (64 px fields at 0.5 µm), PNG 300 dpi, 10 µm scale bars on every panel | measured bar length in the PNG = 20 image px × the panel's output scale, within 1.5 output px, for all 12 panels |
| Same comparison as SVG | scale-bar paths 20/64 of the panel width (in points) within 1 %; labels kept as text |
| Regenerating a PNG from its sidecar | pixel-identical (also via `hcs-studio figure --regenerate`) |
| Linked contrast | every panel drawn with the same channel display; an unlinked panel is labelled in the figure |
| Sources | `raw` equals the raw field; `step:1` recomputes the first step with its labels |
| Crop (µm) | output panel size equals the crop at the pixel size |
| Movies | GIF and MP4 with one frame per Z plane; time-lapse: one frame per timepoint from an OME-Zarr plate with 3 timepoints |
| Memory | the panel cache stays under its byte limit; a figure over 250 MP at the chosen dpi is refused before drawing |
| Regeneration from a folder source (OME-Zarr) | the store is fingerprinted (paths, sizes, times); a changed chunk blocks regeneration |

Hardening review (after Phase 15) also fixed: annotation labels followed a stale layout
cache after the layout file changed; concentrations of different unit kinds (µM and
µg/mL) were ordered as one numeric series; each `raw`/`step:` panel reopened the plate.

Not yet checked: figures from real plates in journal templates; colour-blind-safe channel
colours are the user's choice.

## Pharmacology (Phase 13)

### Verified on synthetic data

| Check | Result |
|---|---|
| 4PL on 10 concentrations × 3 replicates (noise SD 2 %) | LogEC50 within 0.05, Hill within 0.1, Top/Bottom within 3 %; absolute IC50 matches the analytic value |
| 95 % intervals (covariance, t distribution), 200 simulated curves | coverage 0.90–0.99 for LogEC50 and Hill (300 curves: 94 % and 93 %) |
| 5PL (S = 3) | Prism's LogEC50 recovered within 0.02 |
| Robust fit with one wild well | EC50 closer to the truth than least squares |
| F test of EC50 shift (0.5 log units, 3 replicates) | p < 10⁻⁶; identical curves p > 0.05 |
| GR from a known GR curve (doubling time path) | GEC50 and GR50 within 2 %, GRinf within 0.02 |
| Z′ and SSMD | equal to the formulas on simulated controls |
| Bliss-additive 6 × 6 matrix | Bliss 0 (exact), ZIP < 0.5 |
| Sham combination (a drug with itself) | Loewe < 0.05, combination index 1.000 |
| Synthetic 384-well plate (synthetic.py, noise ≈ 1.5 %): EC50s of six curves | within 6 % of the truth (EC50s 0.1–2 µM); HeLa/MCF7 EC50 shift of 10× found (p ≈ 10⁻¹⁹) |
| Same plate: A × B built synergistic, A × C Bliss-additive | A × B > 10 on all four models ("likely synergistic"); A × C Bliss 0.0, ZIP 0.0 |
| Full report (2 readouts, HTML + PDF + 12 figures as PNG and SVG) from that plate | 10–20 s (requirement: < 1 min) |

### Choices to know

- Prism's parametrisation and sign convention (falling curves have a negative Hill slope).
- Loewe uses HSA where no Loewe solution exists (as SynergyFinder does); such points are
  counted in a note.
- Bliss and HSA use the observed single agents; SynergyFinder 3 can apply a baseline
  correction (`correction = TRUE`), which changes scores.
- Single-agent fits for Loewe/ZIP include the 0 point as a dose six decades below the
  lowest (the log-logistic limit of a zero dose).
- Outliers: the MAD of a replicate group is floored by the vehicle wells' spread; with
  triplicates k = 3 still flags about 2 % of ordinary wells.

### Still to do with reference data

- GraphPad Prism: 5 curves (EC50 within 5 %, Hill within 0.1) — put the files in
  `tests/data/pharmacology/prism/` (format in the README there); the regression test runs
  automatically.
- SynergyFinder: a reference 6 × 6 matrix with its Bliss, HSA, Loewe and ZIP scores and
  the SynergyFinder version/options — `tests/data/pharmacology/synergyfinder/`.
- MuSyC (needs the `synergy` package; not added).

## Projection and background removal (Phase 12)

### Verified on synthetic data

| Check | Result |
|---|---|
| Max projection of 20-plane stacks with a known maximum per pixel | exact |
| mean, sum, min, std projections | equal to NumPy (rtol 1e-5) |
| EDF (variance and pyramid) of a stack with different in-focus halves | correlation with the sharp texture > 0.97 (higher than the MIP) |
| Auto plane selection (normalized variance, Brenner) on stacks with known best focus | best plane found; planes reported per channel |
| Background removal on a 512 × 512 gradient (≈ 50 % across the field + a broad bump) with 40 Gaussian spots and noise (SD 5): integrated spot intensity (radius 6.5 px) | within ± 5 % for every spot and every method: tophat, rolling ball, paraboloid, gaussian, median, polynomial, spline (worst: rolling ball / paraboloid ≈ 4–5 %, from the noise floor left by the opening) |
| Divide by a polynomial surface under 40 % multiplicative shading | background CV < 1 %, spots equal within 2 % |
| BaSiC-style flat- and dark-field from 40 fields with different backgrounds | flat-field within 3 %, dark-field within 15 counts (of 60–110); corrected background CV < 3 % |
| Radius check | warns for radius < 2× object diameter; objects of 8 µm lose > 80 % (tophat) or > 20 % (rolling ball) with a 3 µm radius, < 30 % with 20 µm |
| 384 wells × 3 channels × 20 planes (32 × 32 px fields, OME-Zarr) in the batch queue | cancelled after 100 wells and resumed: no well redone, all 384 equal to projection + background applied directly |

### Choices to know

- Morphological backgrounds (tophat, rolling ball, paraboloid) are estimated on a 3 × 3
  mean, as ImageJ does by default. Without it, an opening of a noisy image follows the
  noise minima and leaves a floor of about 2–3 noise SDs, which added +5–15 % to spot
  sums in the test above.
- Rolling ball and paraboloid use the intensity scale for the ball's height, as ImageJ
  does. On steep slopes (g counts per pixel) the ball stays R·(√(1 + g²) − 1) below the
  surface: negligible for real shading (g ≪ 1), noticeable on very steep synthetic
  ramps. Large radii are computed on a shrunk image (block mean, not ImageJ's block
  minimum, which added another noise-minimum bias).
- The Gaussian background ignores bright objects (pixels > 2.5 robust SDs above the blur
  are replaced and the image re-blurred); a plain blur raised each spot's own background
  and lost 8 % of its intensity.
- BaSiC-style dark-field: D is only defined up to a multiple of the flat-field (a
  constant added to every field's baseline moves into D). The multiple that leaves D
  uncorrelated with S is used, which is exact for uniform (camera offset) and linear
  dark-fields; a dark-field shaped like the vignetting cannot be separated. With field
  backgrounds that vary by < 5 %, no dark-field is estimated (a note says so).
- This BaSiC-style estimate is a per-pixel regression with Gaussian smoothing, not the
  published L1 / low-rank optimisation of Peng et al. (2017).

### Still to do on real data

- Compare projection + background with Fiji (Z Project; Subtract Background with rolling
  ball and sliding paraboloid) on the same fields.
- BaSiC on a real plate against the BaSiC Fiji plugin or `basicpy`.
- Typical radius and runtime on full-size (2048 × 2048) fields.

## Performance and robustness (Phase 10)

### Full 384-well plate

Synthetic ImageXpress plate written by `scripts/make_test_plate.py`: 384 wells × 2 sites,
2 channels, 1080 × 1080 px (3.4 GB raw), 30–400 cells per field (149,187 nuclei).
Pipeline `scripts/pipelines/perf_384.yaml`: flat-field, threshold + watershed nuclei,
cells grown by 6 µm, all object features (morphology, intensity, radial, Haralick
texture, granularity), per-cell colocalization, QC. Cloud machine with 4 CPU cores and
15 GB RAM, `--workers auto` (3 workers); other jobs ran on the machine during part of the
run, so the times are upper bounds.

| Measure | Result |
|---|---|
| Wall time | 22 min (fit 7 s, wells 21 min, tables and report 40 s); 36 fields per minute |
| CPU time per field, summed over steps | 4.9 s: object features 3.1 s (61 %), colocalization 0.7 s, writing 0.5 s, segmentation 0.3 s, QC 0.2 s, cells 0.1 s, reading 0.01 s |
| Peak memory | 463 MB per worker; 2.7 GB for the whole machine |
| Output | 7.5 GB (2.2 × the raw data; float32 corrected images dominate). Without corrected images (`--no-images`) a run folder is about a quarter of that |
| Failures | none; all 768 fields, 384 well summaries, report |

Optimisations made from this profile:

| Change | Effect |
|---|---|
| Granularity by OpenCV morphology with cached disk footprints (was SciPy, footprint rebuilt per call) | object features of a dense field 6.6 → 4.0 s (profiled); openings identical to SciPy's (max difference 0) |
| Plate summary computed well by well (was: whole plate object table in memory) | on this plate 132 → 46 s and 1,403 → 357 MB peak; identical per-well summary (384 wells × 672 columns); memory no longer grows with plate size |
| Classifier training from the CLI reads the labelled wells and a sample (was: whole plate table) | bounded memory |

Haralick texture (mahotas, about 2,000 calls per field) is now the largest single cost;
switch texture off in *Measure: objects* when it is not needed.

### Flat-field accuracy (found while profiling)

The first synthetic plate shaded only the cells, not the background; its flat-field
estimate was correctly flat, but it showed that the test data were unrealistic. With
background shaded as in real optics (vignette 0.74–1.13 of the mean):

| Data | Gaussian smoothing (before) | 2D polynomial (now the default) |
|---|---|---|
| Background only (30–50 fields) | max error 3.8 % (corners flattened), RMS 0.4 % | max error 0.2 %, RMS 0.06 % |
| Nuclei, 30–400 per field | RMS 1.8 % | RMS 1.1–1.5 % |
| Dense cytoplasm stain | RMS 5.3 % | RMS 4–5 % (the median assumes mostly background) |

Tests: `test_flatfield_profile_is_accurate_up_to_the_border` (off-centre, tilted vignette:
polynomial within 0.5 % everywhere); `test_flatfield_without_background_falls_back_to_smoothing`
(almost no background: the polynomial is not trusted, Gaussian smoothing is used for that
channel and listed as `gaussian_fallback_channels` in the manifest; no background at all:
the step refuses).

### Robustness

| Check | Result | Test |
|---|---|---|
| Disk-space check refuses a run that cannot fit, before creating the run folder | yes | `test_refuses_to_start_without_disk_space` |
| Disk full during a run: the run stops after the failing well (not every well failing), the folder resumes once space is freed | yes | `test_disk_full_stops_the_run_and_it_resumes` |
| Ctrl+C: stops after the current field, exit code 130, prints the resume command; resume completes | yes | `test_ctrl_c_stops_cleanly_and_resumes` |
| Runs without corrected images give identical tables | yes | `test_runs_without_saving_images` |
| Memory limit caps the worker count and the deconvolution budget | yes | `test_memory_limit_caps_workers`, `test_memory_limit_reaches_deconvolution` |
| Timings per well and per run recorded | yes | `test_run_records_timing_and_performance` |
| Unexpected GUI errors are logged and shown; the app keeps running | yes | `test_gui_error_handler_logs_and_keeps_running` |
| `hcs-studio selftest` in the development environment | 10 of 10 checks (registry added in Phase 17a) | `test_selftest_passes` |
| **Frozen bundle (PyInstaller, Linux, without PyTorch)**: `info`, `selftest` (worker processes, numba, figures inside the bundle), GUI smoke test | all pass, after fixing two bundling gaps the self-test found (matplotlib backends loaded by name; napari's resources package) | manual, `packaging/hcs_studio.spec` |

Not yet done: the Windows installer itself (built and self-tested by the
`hcs-studio-windows` workflow on a Windows runner; it cannot be built in the Linux
container), the CUDA variant, and timing a real 384-well plate on the analysis
workstation.

## Profiles, PCA/UMAP and dose-response (Phase 9)

Synthetic screen: two plates of 22 wells (4 negative controls, 4 positive controls, Drug X
and an inactive Drug Y at 7 concentrations), 150 objects per well. The positive control
and Drug X shift two independent features (Drug X following a 4-parameter logistic with
EC50 = 1 µM, Hill 1); plate 2 is offset in every raw feature.

| Check | Result | Test |
|---|---|---|
| Robust Z relative to control rows, per plate | exact; a plate offset is removed | `test_robust_z_relative_to_controls` |
| Missing, near-constant and correlated (|r| > 0.95) features dropped, with reasons | exact | `test_feature_selection_and_correlation_filter`, `test_profiling_two_plates` |
| 4PL fit of noisy data (EC50 2.5, Hill 1.3) | EC50 within 10 %, Hill within 15 %, R² > 0.99; decreasing curves; < 4 concentrations refused | `test_four_parameter_logistic_fit` |
| PC1 carries the two shifted features | yes | `test_profiling_two_plates` |
| Distance to control, positive control (theory √(4² + 4²) = 5.66 object SD) | 5.6–5.9 | same |
| Distance of negative-control and inactive-compound wells | < 0.5 (0.0–0.23) | same |
| EC50 of Drug X from the measured feature on both plates | within 35 % of 1 µM, R² > 0.95 (exact from the true shift) | same |
| Two plates with a raw offset agree after control normalisation | correlation of well profiles > 0.95 | same |
| UMAP and PCA reproducible with the same seed | identical | `test_profiling_is_reproducible` |
| Report: PCA/UMAP panel in Chromium offline; click a point opens its well; rectangle marks wells on the plate | yes | `test_cli_profile_results_and_report` |
| Profiles tab: lasso marks wells on the results plate; point opens the object in the viewer; plate click highlights points; SVG/PNG export | yes | `tests/test_gui_profiles.py` |

### Choices to know

- Well profiles and distances are scaled by the control **objects** (median and MAD of
  hundreds of objects), not by the spread of control-well medians: with 4 control wells
  that spread is too unstable (an inactive well scored 7× the control spread in an early
  version). Distances are therefore in units of object-to-object variation.
- `profile_distance_ratio` divides by the control wells' own typical distance; with few
  control wells it varies a lot (0.7–2.1 for the controls themselves here). Prefer
  `profile_distance_to_control`.
- Normalised values are clipped to ±10 before PCA, UMAP and distances.
- UMAP with a fixed seed runs single-threaded (reproducible but slower).

## Stitching, classification, clustering and spatial statistics (Phase 8)

### Verified on synthetic data

| Check | Result | Test |
|---|---|---|
| Stage orientation (stage y up or down) found from the overlaps | correct in all cases | `test_detects_stage_orientation`, run test |
| 3 × 3 fields, stage error SD 1.5 µm: position error after refinement | max 0.34 µm at 0.5 µm/px (7.4 µm without) | `test_refinement_reaches_subpixel_accuracy` |
| Featureless overlaps and fields without overlap keep stage positions, with a note | yes | `test_featureless_overlaps_are_rejected`, `test_no_overlap_falls_back_to_stage_positions` |
| **Whole run, 2 × 2 Operetta fields with 25 % overlap, one field 2 µm off**: field offsets within 1 px of the truth; nuclei seen in two or four fields counted once (25 of 25 per well); well area within 1 % of the true union | yes | `test_run_with_stitching_and_table_steps` |
| Gate counts with min/max, all/any, missing values | exact | `test_gate_counts_objects_inside_thresholds` |
| Two separable classes: cross-validated accuracy | random forest 99.2 %, gradient boosting 97.5 %; confusion matrix sums to the examples | `test_classifier_cross_validation_and_round_trip` |
| A second plate 3 SD brighter in every feature | 98.8 % correct with plate robust Z, 59 % without | `test_plate_normalisation_removes_a_plate_offset` |
| Active learning suggests objects at the class boundary, never already-labelled ones | yes | `test_active_learning_suggests_boundary_objects` |
| Three well-separated groups: k-means and HDBSCAN recover them (adjusted Rand index) | > 0.95, fractions 1/3 each | `test_phenotypic_clusters_recover_blobs` |
| Random (CSR) points, 1 × 1 mm window, 1500 points: L(r) − r | −0.35 to −0.65 µm at 20–80 µm; g 0.97–1.01; inside the 39-simulation envelope; Clark–Evans R 1.01 | `test_ripley_csr_is_flat_and_inside_the_envelope` |
| Thomas cluster process (σ 12 µm) | H 19–37 µm, far above the envelope (< 1 µm); R 0.38 | `test_ripley_detects_clustering_and_regularity` |
| Square lattice (20 µm) | R 1.98 (theory 2); Voronoi area 400 µm² (exact); K(10 µm) = 0 | same |
| Hexagonal lattice | 6 Delaunay neighbours, Voronoi area √3/2·s² within 1 %, edge cells left empty | `test_voronoi_and_delaunay_on_a_hexagonal_lattice` |
| DBSCAN / HDBSCAN colonies of 40 and 15 objects plus 2 isolated objects | exact | `test_dbscan_colonies` |
| Classify dock: gate on histogram and scatter, examples labelled by clicking objects, training, uncertain objects, apply to the run, colour by class; mosaic | yes | `tests/test_gui_classify.py` |

### Known limitations

- Plate robust-Z normalisation assumes most objects on a plate are alike. In the test where
  one well held only one class, 1 of its 10 objects changed class; on plates dominated by
  one phenotype use no normalisation.
- HDBSCAN and Leiden clusters are assigned to new objects by their 5 nearest sampled
  neighbours (an approximation); k-means assignment is exact.
- Ripley's K uses border correction, which discards centres near the edge of the imaged
  area at large radii (`n_inner` in the curves); choose radii well below half the well
  width. Clark–Evans uses Donnelly's correction, derived for rectangles.
- Stitching corrects translation only (no rotation or scale), and matches downsampled
  fields when they are larger than `image_max_size`.

### Still to do on real data

- Stitch a real Operetta and Nikon well and compare with the vendor's stitched image.
- Train a classifier on a real screen and check the accuracy against held-out wells.

## Results and report (Phase 7)

| Check | Result | Test |
|---|---|---|
| Robust Z and Z′ against hand-computed values | exact | `test_robust_z_and_z_prime` |
| Readouts, % positive, control wells, robust Z against negative controls on a real run | exact | `test_run_results` |
| `index.html` contains no external resources; every preview it links exists, by relative path | yes | `test_report_is_self_contained` |
| **Acceptance: a zipped and unpacked run folder, opened from `file://` in Chromium with all network requests blocked: clicking a well opens its enhanced previews**; stages, outlines, readout filter, robust Z (equal to Python's), keyboard and `#well` links work; no script errors | yes | `test_report_works_offline_in_a_browser` |
| In-app results plate, % positive, well panel, viewer and side-by-side comparison | yes | `tests/test_gui_results.py` |

### Correction found while building this phase

The field focus score of Phase 5 grew with the number of cells in a field (0.031 for one
nucleus, 0.073 for five), so sharp but sparse fields, such as wells where a compound
killed cells, were flagged out of focus and left out of the counts. It is replaced by the
sharpness of object edges relative to object contrast, which is independent of cell
number (0.327 for 1 and 5 nuclei) and brightness and falls steadily with blur (0.39× at
σ = 3 px). Fields without clear objects get no score and are never flagged. Runs made
before this change may have excluded sparse fields; rerun them to be sure.
