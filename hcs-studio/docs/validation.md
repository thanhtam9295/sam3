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
