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

## Normalization across plates (Phase 17b)

| Check | Result | Test |
|---|---|---|
| Robust Z against a hand calculation | exact | `test_robust_z_matches_a_hand_calculation` |
| Median polish of an additive row + column pattern | residuals 0, fitted effects exact | `test_median_polish_recovers_additive_row_and_column_effects` |
| 384-well screen, 16 seeded hits (−45 %), row gradient −40 % and column gradient +25 %, noise 4 % | without correction fewer than half the hits are called (robust Z ≤ −3); with B-score ≥ 95 % are called and the artifact adds no false hit (same calls as the plate without the artifact) | `test_b_score_removes_a_row_column_artifact_and_keeps_seeded_hits` |
| B-score on the Phase 13 dose–response/combination plate, and on a dilution series down one column | refused, with the reason | `test_b_score_is_blocked_on_dose_response_and_combination_layouts` |
| Randomized library; one compound at 3 doses in scattered wells | allowed | `test_a_randomized_library_is_allowed` |
| Uniformity map from a tilted + domed plate with a debris well | factors within 1 % of the true shading; later plate CV reduced > 60 % | `test_uniformity_map_recovers_a_known_gradient_and_flattens_later_plates` |
| Sharp edge step (edge wells −10 %) | corrected to < 0.5 % with the edge-ring term; a polynomial alone leaves > 2 % | same test |
| Three days with different vehicle levels | R² day > 0.9 raw, < 0.05 after same-day controls | `test_same_day_controls_remove_a_day_effect` |
| Replicate statistics | identical plates r = 1, bias 0; a +5 shift gives bias 5 ± 0.6; a 5-fold IC50 difference is flagged | `test_replicate_concordance_statistics_and_flags` |

Not yet checked: real uniformity plates (edge effects of your plates and instruments);
B-score on a real randomized library screen.

## Server mode (Phase 27)

| Check | Result | Test |
|---|---|---|
| Registry migrations, audit chain and accounts on PostgreSQL 16 | migrate 0001–0009; chain intact; argon2 sign-in | manual run 2026-11-24; queue tests below on PostgreSQL |
| Queue: priority, fair share, GPU routing; events; cancellation; crash → requeue with resume → interrupted after 3 attempts | as specified, on SQLite and PostgreSQL | `tests/test_server_queue.py` |
| **Done-when:** two clients submit at the same time | both complete, separate run folders | `test_two_clients_jobs_complete`, `test_two_clients_at_once` |
| **Done-when:** a worker killed (SIGKILL) mid-job | the pool restarts it; the job is requeued and resumes in the same run folder, keeping finished wells | `test_killed_worker_is_restarted_and_job_resumes` (SQLite and PostgreSQL) |
| **Done-when:** a 384-well plate (308 imaged wells) submitted through a real server (uvicorn, worker pool) by a client that then disconnects | completes; visible to a new desktop client (queue, registry, `RunResults`) and in the web portal (run, report, image tile) | `test_384_well_plate_while_the_client_is_away` (~30 s) |
| API sessions and permissions (wrong password, viewer submits, analyst opens admin, sign-out) | 401 / 403 / 403 / 401 | `test_sessions_and_permissions` |
| Run through the API: worker, registry indexing, run summary, report file, path traversal, image tiles equal to the raw plane, WebSocket progress, audit as the API user | as specified | `test_run_through_the_server` |
| Portal in Chromium (Playwright): sign-in, plates, run state, the canvas viewer draws nuclei, jobs | no page errors | `test_portal_in_a_browser` |
| Connected Jobs dock | submits through the API, sees the job finish | `test_jobs_dock_in_connected_mode` |

Not tested here: Docker Compose and the Windows service (no Docker in the test container; the
compose file, Dockerfile and nginx config are checked for consistency only), GPU workers on a
real GPU, TLS through the proxy.

## Run states, sign-off, locking (Phase 26b)

| Check | Result | Test |
|---|---|---|
| **Done-when:** scripted scenario (layout saved, run, re-annotation, analysis, a denied signature, review, a refused second review, approval, three refused changes of the locked run, re-analysis as a new version) | every action in the audit log with user, role, reason and status; chain intact; the run's audit records in its report | `test_scripted_scenario_is_fully_audited` |
| **Done-when:** a locked run | refused by analyze, annotate and report; files read-only; re-analysis needs a reason and makes `<run>_v2` linked to the original (manifest and registry), the original stays approved | same |
| Report states | DRAFT mark and state until approval; signatures with name, role, time, comment, manifest hash; no DRAFT after approval | same |
| Regulated mode | signing without / with a wrong password refused; approver must differ from the reviewer; signatures record the re-authentication | `test_regulated_mode_requires_password_and_two_people` |
| CLI `run-state`, `sign`, `unlock` (admin only), `analyze --new-version`; unfinished runs are drafts; backup | as specified | `test_unlock_and_cli`, `test_states_of_unfinished_runs`, `test_registry_backup` |
| Sign dialog | reviewed, then approved, then disabled (locked) | `test_sign_run_dialog` |

Not checked: file locking as an ordinary (non-root) user on Windows and Linux shares (the test
container runs as root, which ignores read-only bits; the test checks the mode bits).

## Accounts, roles and the audit trail (Phase 26a)

| Check | Result | Test |
|---|---|---|
| Permission matrix | reviewer for signing and model status, admin for settings/accounts, analyst otherwise | `test_permission_matrix` |
| argon2id passwords, lockout after 5 failures (also against the right password), admin reset, disabled accounts, session timeout | as specified | `test_accounts_passwords_and_lockout`, `test_sessions_time_out` |
| A viewer attempting a change | refused; recorded with status `denied` and the role | `test_services_enforce_roles_and_record_denials` |
| **Done-when:** a reason edited directly in the database; a record deleted | detected (content changed / missing record, broken link) | `test_chain_verifies_and_detects_tampering` |
| A forger who rewrites every hash after an edit | detected by the checkpoint kept outside the database | `test_checkpoints_catch_a_rewritten_chain` |
| Records written before the chain existed | chained by the next event | `test_old_unchained_records_are_chained` |
| Filters, CSV and PDF export, CLI (`audit verify` exit 1 on tampering), audit dialog, sign-in dialog | as specified | `test_query_and_export`, `test_accounts_and_audit_commands`, `test_audit_and_sign_in_dialogs` |

Not checked here: PostgreSQL privileges (append-only role) — Phase 27.

## Model training, validation and registry (Phase 25)

| Check | Result | Test |
|---|---|---|
| Matching and AP on known masks (identical; a square shifted to IoU 0.667; a merge at IoU exactly 0.5; a split; an empty prediction; one-to-one matching below 0.5) | AP 1 / AP 1 up to IoU 0.65 and 0 from 0.70 (mAP 0.4) / 1 merge, 1 match / 1 split / recall 0 / one match | `tests/test_training_metrics.py` |
| Counts summed over fields and per cell line | AP@0.5 = 3/4 overall, 1 and 1/2 per cell line | `test_counts_are_summed_over_fields_and_grouped` |
| Classifier metrics and calibration (known confusion; perfect and over-confident probabilities) | balanced accuracy 0.875; ECE 0 / 0.4; Brier 0 | `test_classifier_metrics_known_answer`, `test_calibration_known_answers` |
| Ground-truth versions, hashes, splits by well/plate/held-out plate | versions immutable; re-split keeps the content hash; no group in two splits | `tests/test_groundtruth.py` |
| **Done-when (synthetic):** a hard "cell line" (touching clusters, uneven brightness), 24 fields split by well; classical segmentation tuned on the train split | held-out (test) AP@0.5 more than 0.15 above the default segmentation | `test_tuning_improves_held_out_ap` |
| **Done-when:** a pipeline pinned to a registered model version, run twice; then another version | identical objects; manifest lists version and SHA-256; the other version is recorded with its own hash | `test_pinned_model_reproduces_masks_and_manifest_records_it` |
| Statuses, approval, demotion, retired and tampered models | refused with clear messages | `test_registry_versions_statuses_and_integrity` |
| Object classifier from a classification set, used by registry reference | balanced accuracy > 0.95; the step loads the registered file | `test_cellpose_and_classifier_accept_registry_references` |
| Training as a queue job (own process) | completes and registers | `test_training_job_in_the_queue` |

| Comparison report: tuned candidate vs base, then vs the production version | report, figures and JSON saved in the model store; candidate AP@0.5 > base + 0.15 on the test split; registry metrics updated | `test_comparison_report_is_saved` |
| Downstream impact on the dose–response reference plate (Drug D on MCF7, 24 wells): an equivalent candidate and a broken one | equivalent: counts within 1 %, IC50 fold change < 1.2 (MSR 3); broken: flagged (count difference) | `test_downstream_impact_against_msr` |
| Drift: a run of the training "cell line" and one of small dim nuclei | no alert / area alert (median < −1 IQR) with a suggestion | `test_drift_and_spot_checks`, `test_drift_maths` |
| Spot checks: queue once per run, flag needs a reason, accept rate per model | 4 / 5 | `test_drift_and_spot_checks` |
| Real Cellpose (`HCS_REAL_CELLPOSE=1`, cpsam_v2 weights): 1 epoch on 4 fields on the CPU | fine-tunes, validates against cpsam_v2, registers (68 s, 2026-11-24) | `test_real_cellpose_fine_tune_registers` |

Not yet checked: fine-tuning **Cellpose** on real annotated fields (needs the 20–40 annotated
fields asked for, and a GPU); the drift baseline comes from the model's own masks on its
training fields, so it is only as representative as those fields; the 1-IQR threshold is a
starting point.

## Cell Painting: profile evaluation (Phase 24c)

| Check | Result | Test |
|---|---|---|
| Synthetic single cells, 2 plates (gain 2 + offset 5), 6 vehicle wells per plate: a strong (+3 SD on 5 features), a weak (+1 SD on 2) and an inert compound | strong mAP 1 and active; weak active (FDR < 0.05); inert and held-out vehicle wells not active (p > 0.05) | `test_seeded_phenotypes_active_vehicles_not` |
| 3 MoAs × 4 compounds (shared feature block + a compound-specific feature) and 2 unannotated queries | all 3 MoAs consistent, mAP > 0.9; cosine/Pearson matrices symmetric with unit diagonal; nearest of a compound are its MoA; kNN leave-one-compound-out accuracy 100 %; the queries get the right MoA | `test_consistency_similarity_and_moa_suggestions` |
| 4 plates, 2 days (gain/offset by day), per-plate feature offsets SD 1 | vehicle wells by plate: mAP 1.0, silhouette ≈ 0.97 before normalization; ≈ 0.22 and −0.1 after; by day 1.0 → ≈ 0.5 (chance) | `test_batch_effect_before_and_after_normalization` |
| Synthetic Cell Painting **images** (8 vehicle wells, 4 wells larger nuclei, 4 fragmented mitochondria) → preset → profiles → evaluation | both phenotypes active, held-out vehicle wells not | `test_painted_plate_end_to_end` |
| MoA files (CSV with `Metadata_` names, XLSX, missing columns); CLI `profile --evaluate`, `evaluate`, `similar` | read / clear error; every file and figure written | `test_read_moa`, `test_profile_and_evaluate_commands` |

Not yet checked: a real Cell Painting plate with reference compounds and MoA annotations
(asked for); with 6 vehicle wells and 3 replicates the test had too little power (perfect mAP,
p = 0.054), hence the guidance in docs/cellpainting.md.

## Cell Painting: profiling pipeline (Phase 24b)

| Check | Result | Test |
|---|---|---|
| Real JUMP CellProfiler CSVs (BR00121431, pycytominer test data: 2 wells × 2 sites × 30 cells, 5790 features): aggregate, mad_robustize normalization (120 cells), feature selection (variance, correlation 0.9, NA, blocklist, outliers), consensus vs pycytominer's own functions called directly | identical values (rtol 1e-12) and missing values; identical kept columns; the log of removed features adds up | `test_reproduces_pycytominer_on_jump_data` |
| Whole chain with a plate map (annotate) on the same data | = annotate → normalize called directly; consensus by the plate-map column; without treatments one profile per well | `test_whole_chain_on_jump_data` |
| Synthetic 2 plates × 12 wells, plate 2 = plate 1 × 2 + 5 cell by cell, compound A +3 SD on 5 features, B −3 SD on 5 | vehicle well medians 0 on each plate (< 1e-12); per-plate profiles of each compound identical on both plates (< 1e-9); consensus A > 5, B < −5 on the seeded features, DMSO within ±1; pooled normalization leaves the plate effect (P1 < −0.5, P2 > 0.5) | `test_vehicle_normalization_per_plate` |
| reference all, spherize (ZCA-cor against the vehicles), none; missing vehicles | each plate's medians 0; whitened vehicle mean 0; unchanged; clear errors (no vehicles, a plate without vehicles) | `test_reference_all_and_spherize` |
| Single-cell QC: 10 cells flagged by object QC, 5 missing 5 of 30 features, 2 missing 1 | 10 + 5 dropped, the 2 kept; well cell counts follow; a constant feature removed by variance, a linear copy by correlation | `test_single_cell_qc_and_feature_selection` |
| `cellpainting profile` on Parquet and on a run folder | all stage files, selection log and options written | `test_profile_command_writes_stages`, `test_cellpainting_commands` |

Not yet checked: a real plate's profiles against its published JUMP profiles (the subset here
is too small for that); seeded phenotypes are scored for activity in Phase 24c (mAP).

## Cell Painting: features, naming, import (Phase 24a)

| Check | Result | Test |
|---|---|---|
| Name mapping (shape, intensity, quartiles, radial rings, Haralick, granularity, correlation, neighbours, children, Zernike, unknown) | CellProfiler names as listed in naming.py | `test_cellprofiler_names` |
| Channel correlation on known images (copy ×2 + 1, negation, independent noise) | 1.0, −1.0, \|r\| < 0.2 | `test_correlation_known_answers` |
| Synthetic Cell Painting plate (3 wells × 2 fields × 25 cells; larger nuclei ×1.3, fragmented mitochondria) through the preset | 150 cells, one row each; > 250 features per compartment; Metadata from the layout; no duplicated child summaries; pycytominer's `infer_cp_features` finds exactly our features; nuclear area ×1.69 (±12 %); mitochondrial granularity differs | `test_preset_single_cell_table` |
| pycytominer's example SQLite (SQ00014613, 2 cells, 1790 features) | imported features = pycytominer's merged single cells exactly (max difference 0, same missing values) | `test_cellprofiler_sqlite_matches_pycytominer` |
| Same data as ExportToSpreadsheet CSVs and as Parquet | same features | `test_cellprofiler_csv_folder_and_parquet` |

Not yet checked: a real Cell Painting plate (asked for, with reference compounds and MoA
annotations); CellProfiler's own texture/granularity values (our definitions differ in detail:
texture averaged over directions, granularity scales in pixels), so features computed here and
in CellProfiler are not interchangeable — profile one source at a time.

## Primary screens: cherry-picking and confirmation (Phase 23c)

| Check | Result | Test |
|---|---|---|
| 30 hits × 3 replicates, 10 µM from 10 mM into 40 µL | 40 nL each (16 droplets); layout concentration 9.99 µM (achieved); no two replicates adjacent; controls untouched | `test_replicates_are_spread_and_volumes_exact` |
| 150 hits × 3 on a 384 template with 320 free wells | 106 compounds on CONF-001, the rest on CONF-002, 3 replicates each | `test_full_plates_continue_on_the_next` |
| 8-point dose–response from 30 µM by 3, 2 replicates | series along a row, highest first, replicates in other rows; 30, 10, 3.3 µM kept (within 10 %), 1.1 µM and below skipped (rounding 12–100 % off or < 1 droplet); whole droplets | `test_dose_response_series_and_droplet_limit` |
| Backfill | every compound well totals the plate's largest volume (120 nL) | `test_backfill_equalises_the_vehicle` |
| Pick-list formats | default Echo columns; custom headers, fixed text, A1 wells, `;`, one file per plate | `test_pick_list_formats` |
| CLI on a synthetic screen | pick list from a library map file, layout registered (v1) with all transfers, one schema-valid `cherrypick.request` in the outbox, summary with the confirmation section | `test_cherry_pick_command_registers_and_sends` |

Found while building: rounding to 2.5 nL droplets moved low concentrations far off (0.37 µM →
0.62 µM, +69 %) → transfers more than 10 % off are skipped and listed (intermediate plates are
needed for low points). Done-when not yet met: the pick list must match a sample Echo file of
the user's Cherry Pick protocol (asked for).

## Primary screens: review views (Phase 23b)

| Check | Result | Test |
|---|---|---|
| Report data of a synthetic screen (4 plates) | summary = tables; thresholds parsed from the rule (`rep_score <= -3`, `count_pct > 60`); replicate pairs for every compound with 2 clean replicates; the replicate hits = the hit list | `test_report_data` |
| Report in Chromium, offline | no script errors; 4 heat maps × 384 wells; every sample well in the order and distribution plots (also after switching score); hover text | `test_report_works_offline_in_a_browser` |
| Image screen: 2 replicate plates, one compound with 8 nuclei per field vs 32–39 elsewhere | the only hit; one gallery strip per replicate; with linked contrast the hit image's mean brightness < half its nearest vehicle wells'; linked from the report | `test_hit_gallery_has_linked_contrast` |
| Screen without images; `screen report` | gallery says "no image"; report rewritten from screen.json | `test_gallery_without_images_says_so`, `test_screen_report_command` |
| Screen tab | hits table = hits.csv; plate map filled; review without a reason refused; exclude with a reason removes the hit (service, overrides.json) | `test_screen_tab_reviews_hits` |

Found while building: an f-string with a backslash (escaping `</` in the embedded JSON) is only
valid on Python 3.12; ruff (target 3.11) caught it — the JSON is escaped before the f-string.

## Primary screens: scoring and hit calling (Phase 23a)

| Check | Result | Test |
|---|---|---|
| **Done-when**: synthetic screen, 4 libraries × 2 replicates (384 wells), 40 seeded hits (−35 %, noise 6 %), a row gradient (±15 %) and an edge offset (−25 %) differing per plate | with B-score 39/40 (97.5 %) hits recovered and no other compound called; robust Z alone: 12/40 and 7 edge-well false hits | `test_seeded_hits_recovered_and_artifact_makes_no_false_hits` |
| 16 cytotoxic compounds (both readouts down), 12 autofluorescent compounds (flagged by image QC) | without the count rule / QC filter they are called; with them none is | `test_cytotoxic_and_artifact_compounds_are_kept_out` |
| Replicate rules all / at least 2 / any | all = 2 ≤ any | `test_replicate_rules` |
| Manual exclude / include with a reason (CLI → service) | hit list rewritten, reason stored; no reason refused | `test_manual_review_needs_a_reason` |
| SSMD of d = (−3, −4, −5); Z′ of known controls | MM −4, UMVUE −4/√π, robust −4/1.4826; Z′ and robust Z′ exact | `test_ssmd_and_z_prime_known_values` |
| Rule language | chained comparisons, `not`, `abs`; attribute access, calls, strings, unknown names refused | `test_rules_language` |
| Dose series along rows | B-score blocked, robust Z used with a note | `test_b_score_refused_on_a_dose_response_plate` |
| Library maps (units in values / header), quadrants, stamp, Echo transfer file with a preamble; registry import and re-import; layouts registered by barcode | placement and concentrations exact (Echo: 10 mM × 80 nL / 40.08 µL); compounds on control wells refused | `test_library_maps_and_assay_layouts`, `test_library_registry_and_registered_layouts` |

Found while building: (1) the cell-count readout defaulted to % inhibition against a positive
control that does not change cell count, so "not cytotoxic" rejected almost every compound (12/40
recovered) → `activity: percent_of_control` for such readouts, documented. (2) The positional
guard rail built a table per compound — 13 s of 15 s on 8 plates → single-concentration
compounds are skipped (same result).

SSMD formulas follow Zhang (2007, 2011) as summarised in the literature; not yet checked against
the papers (NCBI is not reachable from the cloud environment). Not yet checked: a real library
plate map, a real Echo transfer file, the user's hit-calling rules.

## Live-cell kinetics: cell tracking (Phase 22d)

| Check | Result | Test |
|---|---|---|
| Point data: 10 → ~50 cells over 48 h (cycle 20 h ± 10 %), persistent random walk 10 µm/h, hourly frames | ≥ 97 % of links (seen 98.5 %), ≥ 90 % of divisions (seen 92 %); speed within 5 % (9.7 µm/h); cell cycle within 10 % (20.0 h) | `test_tracks_and_divisions_of_known_lineages` |
| Same at 20 µm/h | motion prediction keeps 86 % of links against 74 % without (2-hour frames at 10 µm/h: 77 %) | `test_prediction_helps_fast_cells` |
| A nucleus missed at one time point; an object appearing 200 µm away | gap closed (one track), the far object a new track; speed 5 µm/h exact; no gap closing with max_gap 0 | `test_gap_closing_and_gate` |
| Segmented nuclei of moving, dividing cells (8 → ~25 cells, 24 hourly frames, cycle 14 h, 8 µm/h) through the pipeline | ≥ 95 % of nuclei matched to a true cell; ≥ 90 % of links and ≥ 75 % of divisions; speed within 25 %; cycle within 15 % | `test_tracking_from_images` |

Found while building: (1) daughters can drift > 20 µm apart within one hourly interval, so
"beside a continuing track" missed 60 % of divisions → the daughter is looked for within reach of
the mother's last position. (2) Nearest-position linking lost 26 % of links at 20 µm/h →
assignment on the predicted position. (3) Cycle times were one interval short (first to last
frame) → birth and division at the interval midpoints. (4) Two touching nuclei merged by
segmentation for one frame and separating again read as divisions with 1–5 h "cycles" → a new
object near a track lost in the last max_gap frames is that track coming back, and a cell born
less than min_cycle_h ago does not divide.

Not yet checked: real time-lapse nuclei (crowded monolayers, mitotic rounding and its intensity
change, apoptotic fragments), comparison with TrackMate / ilastik tracking (Phase 16 d).

## Live-cell kinetics: views and QC (Phase 22c)

| Check | Result | Test |
|---|---|---|
| Clean dose–response time course (33 wells, 9 time points) | all four checks pass, no well flagged | `test_clean_plate_passes_every_check` |
| Planted: two acquisitions skipped (18 → 36 h), one well missing 12 h, one well out of focus at 6 h, the plate's focus × 0.6 from 36 h, row A (edge) growing 30 % less by 48 h, a vehicle well not growing | every check warns; gap (18, 36); edge effect log2 0.7 within 0.03; the four wells flagged; interior wells not flagged as edge | `test_planted_problems_are_found` |
| No condition on both edge and interior; vehicle wells without data | edge and control checks "not assessed" | `test_edges_not_assessed_without_a_condition_on_both` |
| Registry with 5 earlier runs (vehicle doubling 19.5–21 h, one excluded) | history read per run and cell line (sample wells ignored); 20.4 h passes, 30 h warns, no growing vehicle fails | `test_control_growth_against_history` |
| Kinetics report in Chromium, offline | no script errors; 33 sparklines; the flagged well outlined; heat map rows = vehicle + 9 concentrations; clicking a well adds its curve; linked from `index.html` | `test_report_page_works_offline_in_a_browser`, `test_report_page_and_link`, `test_doubling_time_from_images` |

Found while building: wells with no objects at a time point (empty after a cytotoxic dose) had
no count at all in the time course and would read as missing time points → objects counted as
0 where a field was imaged. The focus score comes from the `qc` step (not `measure_objects`):
without it the focus check is not assessed.

Not yet checked: the thresholds (gap 1.5 ×, focus 0.5 / 0.8, edge ±25 %, history ± 3 robust SD)
on real time-lapse plates; edge evaporation on a real plate (needs a layout with the same
condition on the edge and inside).

## Live-cell kinetics: kinetic analysis (Phase 22b)

| Check | Result | Test |
|---|---|---|
| Poisson counts (N₀ 2000, 13 time points over 72 h): exponential (Td 20 h), lag 12 h + Td 18 h, logistic (Td 16 h, K 30 000) | right model chosen by AICc; doubling time within 5 % (seen ≤ 2 % exponential, 6 % before the noise was realistic for logistic); lag within 2.5 h; capacity within 10 % | `test_growth_models_recover_doubling_time` |
| Flat, dying, two-point wells | `constant` / `shrinking` / too few points, no doubling time; death rate within 10 % | `test_flat_and_dying_wells_have_no_doubling_time` |
| **Done-when**: segmented nuclei of a time-lapse plate (9 time points, 48 h), doubling 18, 24, 36 h | within 5 % (seen 18.06, 23.99, 36.17 h) through the `kinetics` step | `test_doubling_time_from_images` |
| Dose–response plate (9 concentrations × 3, 6 vehicle wells; drug lowers the growth rate, Emax 1.5, EC50 100 nM): IC50(t) = EC50·o/(1 − o), o = ln 2/(μ₀ Emax t) | IC50 at 24, 48, 72 h within 10 % (seen ≤ 5 %), falling with time; GR50 constant over time = 38 nM within 15 %; GRinf −0.29 within 0.05; vehicle GR ≈ 1 | `test_ic50_and_gr_over_time_known_answers` |
| Drug acting 36 h after treatment, GR over 24 h intervals | GR ≈ 1 in [0, 24], 2^(μ(c)/μ₀) − 1 in [48, 72] (within 0.06); no GR50 early; GR from treatment still diluted at 48 h | `test_gr_over_intervals_shows_a_late_effect` |
| Three separate acquisitions (0, 24, 48 h) through the CLI | doubling 24 h within 5 % (AIC: three points) | `test_kinetics_cli_on_separate_acquisitions` |

GR formula: the endpoint form GR = 2^(log2(x/x₀)/log2(x_ctrl/x₀)) − 1 matches the reference code
(GRmetrics `GRcalculate.R`, datarail gr_metrics and sorgerlab gr50_tools READMEs). The GRmetrics
vignette says time-course calculations are not implemented there. Hafner et al.'s own
time-dependent definition (Nat Methods 2016, online methods) could not be read: NCBI, Europe PMC
and grcalculator.org are not reachable from the cloud environment. GR over an interval with the
interval's start as time zero equals 2^(k(c)/k_ctrl) − 1 exactly, so only the choice of intervals
(consecutive / ≥ interval_h, reported at the midpoint) is unchecked against the paper.

Found while building: GR50 from 3 vehicle wells with one noisy count each before treatment came
out 20 % low (the control growth was over-estimated); exact without noise → tests use 6 vehicle
wells, the docs recommend ≥ 4. Flat dose responses took 3–6 s per 4PL fit (the optimiser wanders
on a flat line), against 0.02 s for real curves → flat time points are detected (range of
concentration means ≤ 3 × replicate SD) and not fitted.

Not yet checked: a real kinetic plate (IC50(t) and GR on an Incucyte / ImageXpress / Nikon
time-lapse plate with its treatment time) — the Phase 22 done-when needs one.

## Live-cell kinetics: time-course data (Phase 22a)

| Check | Result | Test |
|---|---|---|
| Synthetic time-lapse plate: 3 wells growing from 12 cells (doubling 20 h, 30 h, none), 5 uneven time points over 24 h, stage drift 0.4 / −0.25 µm per hour (15 × 9 px at 24 h) | nucleus counts per time point exact; drift per time point within 0.3 px (seen ≤ 0.2 px); times from the plate's timestamps; hours since treatment from the layout's treatment time; confluence rises with the count | `test_time_lapse_run_counts_drift_and_time_course` |
| Textured texture shifted by 1.3 / −0.7 … 4.1 / −2.6 px | within 0.1 px (windowed; unwindowed 0.25 px short: image edges pull towards 0); 30 µm stage jump reported, not applied, later drift still corrected | `test_drift_shifts_recover_subpixel_drift_and_skip_jumps`, `test_register_timepoints_reports_stage_jumps` |
| Brightfield-like cells (r 16 px, texture SD 60, background noise SD 8) at 0–48 % confluence, calibrated on the pooled fields | within 12 % of the truth (≥ 1 pp): seen +5 % (isolated cells) to +11 % (48 %: 4 px gaps between cells count as covered); empty well 0 %; noise-only plate reported and ≈ 0 % | `test_confluence_accuracy_from_texture`, `test_confluence_on_an_empty_plate_is_not_noise` |
| Times from ImageXpress (`acquisition-time-local`), ND2 frame events, OME-Zarr; the Metadata interval when absent | exact | `test_imagexpress_time_points_from_image_timestamps`, `test_nd2_time_points_from_frame_events`, `test_ome_zarr_times_and_the_metadata_interval` |
| Treatment time from an argument, the layout, a `plate.dispensed` event; separate acquisitions joined | precedence as documented; acquisitions ordered by time, 6 h apart; other plates refused | `test_treatment_time_sources`, `test_separate_acquisitions_are_joined_on_their_times` |

Found while building: texture confluence (local SD of a difference of Gaussians, as for
spheroids) read 2–6 × the true confluence: the cell outlines respond far more strongly than the
texture inside cells, and the response spreads a window's width outside → texture of the image
minus its median filter (outlines removed) and an erosion computed from the noise and texture
levels. The first time-lapse prototype lost cells drifting out of the field and filled the edge
with repeated pixels → fields cropped to the area every time point sees.

Not yet checked: real time-lapse plates (Incucyte, ImageXpress, Nikon), real phase-contrast
halos, focus drift over days, the treatment time from a real dispenser event.

## Confocal 3D spheroids (Phase 21c)

| Check | Result | Test |
|---|---|---|
| Synthetic confocal spheroid (r 90 µm, 2 µm pixels, 4 µm Z steps, 60 planes, σ blur 2 µm xy / 4 µm z) with exponential depth attenuation λ = 60, 120, 300 µm | λ within 3 % (seen ≤ 2 %; before background subtraction 20–35 % too long: the offset flattens the decay); per-plane correction flat within 2 % | `test_attenuation_length_recovered` |
| Stack without attenuation; very steep attenuation | unchanged + warning; gain capped at max_gain + warning | `test_attenuation_step_flat_stack_and_gain_cap` |
| Hoechst + EthD-1 spheroids (r 70, 90 µm; dead core radius ratio 0.5–0.6): complete; with attenuation λ = 80 µm; cut by the top of the stack | 3D volume within 2 % (seen −0.3 %, with attenuation the same after correction; without the correction 44 % lost); sphericity 0.97; core volume fraction within 5 pp (seen ≤ 1.7 pp); dead fraction within 3 pp (seen ≤ 0.1 pp); Z coverage 0.71 vs 0.71 and `incomplete_z` | `test_confocal_3d_volume_core_viability_and_coverage` |
| 3 nuclei inside a 3D spheroid | nuclei_count 3, per mm³; brightfield stack refused in 3D | `test_nuclei_counted_per_spheroid_and_3d_checks` |

Found while building: a stain threshold at background + 5 SD counted the blurred tails of the
stained region (3D dead fraction 0.175 instead of 0.125; 2D +2 pp) → half height between
background and the stain's plateau, with the noise floor kept.

Not yet checked: real confocal spheroid stacks (attenuation that is not exponential, refractive
index mismatch, cleared samples), Cellpose 3D nuclei in dense spheroids (needs the Cellpose
weights; not run in CI).

## Spheroids: viability, core, invasion, organoids (Phase 21b)

| Check | Result | Test |
|---|---|---|
| Synthetic brightfield + calcein + EthD-1 spheroids (r 120–200 µm) with core radius ratio 0, 0.3, 0.5, 0.7 (core fraction 0–0.49); two with 2–4 invading strands (10–14 µm wide, 40–140 µm long) | core fraction within 2 pp (seen ≤ 0.2 pp from brightfield and from the dead stain), within 5 pp with strands (their area counts in the spheroid); dead fraction within 3 pp (seen ≤ 2 pp: blurred dead edge counts as dead); viable rim 100 ± 4 µm; protrusion count exact; maximum reach within 6 µm (seen 1); invasion area within 10 % (seen −2 %); no core, no dead area in the untreated spheroid; dead-stain radial profile rises to the centre | `test_core_viability_and_invasion_from_images` |
| Organoid wells: 0, 6 or 20 organoids per field (r 20–45 µm), 2 fields, seeding density 100 | counts exact; median diameter within 3 % (seen 0.2 %); forming efficiency and density exact; `no_organoids`; volume vs vehicle | `test_organoid_mode_counts_sizes_and_efficiency` |
| Disc + one strand; disc with a dark core; flat disc | 1 protrusion, reach 30 px, area 7 × 30 px; shells 0–4 with the centre deepest; core area within 5 %; no core without contrast | `test_invasion_core_and_shells_known_shapes` |

Found while building: small objects (organoids, r ≤ 45 µm) came out up to 30 % too large because
their texture region is mostly background, so the "interior" level was background; the
interior is now the dark class (Otsu) of the region — size errors ≤ 1.3 % from 30 to 400 µm.

Not yet checked: real live/dead spheroids (calcein spill into the core, EthD-1 in a thin rim of
debris), cystic organoids (bright lumen), Matrigel invasion images (out-of-focus strands).

## Spheroids (Phase 21a)

| Check | Result | Test |
|---|---|---|
| Synthetic brightfield wells (2 µm pixels; bright background darkening radially to 40 % at the corners like a meniscus; textured spheroid with a darker core; shot noise): radius 70–200 µm; blurred (σ 8 px); a second spheroid; debris; attached blob; off-centre; cut by the border; an empty well | sphere volume within 1 % (seen ≤ 0.3 %) for clean spheroids and within 5 % blurred (seen −1.4 %); each well gets exactly its expected flags; the empty well: no objects, `no_spheroid`; circularity > 0.85, compactness 1.00, darkness 0.60, edge blur < 3 µm (blurred: > 10 µm) | `test_volume_and_qc_flags_from_brightfield` |
| Fluorescent and dark discs | area within 2 % | `test_fluorescence_and_dark_methods` |
| Ellipse 240 × 120 µm, object cut by the border | ellipsoid volume within 2 %; touching_edge | `test_spheroid_table_known_shapes` |
| 384-well plate (2.5 µm pixels), 16 compounds × 10 concentrations × 2 replicates, IC50 10^−7.5 … 10^−6 M on volume, 2 workers | runs in batch (~35 s here), all 384 volumes within 2 % (seen 1.0 %), 384 crops in the gallery, Z′ > 0.9; IC50s on % of control within 0.05 log (seen 0.005) | `test_384_plate_batch_and_dose_response_on_volume` |

Found while building: the local SD of the raw image is higher on a meniscus gradient than inside
a spheroid (→ difference of Gaussians first); reflected padding at the image border creates a
kink on steep shading (→ odd reflection); two touching spheroids shared one coarse region and
the smaller was lost (→ every refined part kept); a field without a spheroid had its noise split
by Otsu (→ noise floor); an empty well wrote no table and vanished from the summary (→ 0-row
table, reported as no_spheroid).

Not yet checked: real brightfield ULA plates (phase rings, well walls inside the field, very
dense or very loose spheroids, 4× vs 10×); fluorescence spheroids with dim cores.

## Proliferation markers and mitotic errors (Phase 20e)

| Check | Result | Test |
|---|---|---|
| Two nuclei with a micronucleus 3 px away, a segmented micronucleus, a bleb touching a nucleus, debris 20 px away, a 1-px speck | the two micronuclei found and assigned to the right nucleus (the segmented one taken out of the nuclei); bleb, debris and speck rejected | `test_find_micronuclei_geometry` |
| Two log-normal populations (80 / 800, σ 0.35) at 15, 50 and 85 % positive | mixture threshold at √(80·800) ± 5 % at every share; Otsu's moves more | `test_mixture_threshold_does_not_depend_on_the_positive_share` |
| Synthetic images: 4 wells × 4 fields × 36 round cells (cell stain), 1 or 2 nuclei, Ki67 per cell, micronuclei (r 1.5–2.5 px, 3 px from the nucleus), through the preset with cell segmentation | nuclei and cells exact; % Ki67-positive within 1.5 pp (seen: exact); micronuclei per 100 nuclei and % multinucleated within 1 pp (seen: exact) | `test_markers_micronuclei_and_multinucleation_from_images` |
| Object tables: vehicle, nocodazole-like positive control, sample, secondary-only well; Ki67, EdU, pH3 | secondary-only thresholds within 2 pp of truth (seen ≤ 1.2 pp: the positive tail below the threshold); vehicle thresholds (mixture; robust SD for pH3) within 1 pp; Z′ > 0.9 | `test_thresholds_from_secondary_only_and_vehicle` |
| Vehicle without Ki67-positive cells | Ashman's D warning | `test_unimodal_vehicle_warns` |

Found while building: Otsu's threshold on a vehicle with 70 % Ki67-positive nuclei sat at 158
(geometric mean of the populations: 253) and counted the upper tail of the negatives — 3.7 pp too
many positives in treated wells with 15 % positives; replaced by the two-population fit.

Not yet checked: real Ki67/EdU plates with secondary-only wells; pH3 at 20×; micronuclei on real
images (out-of-focus nuclei, apoptotic fragments — run the apoptosis step to tell them apart);
multinucleation with Cellpose cells.

## Nuclear translocation (Phase 20d)

| Check | Result | Test |
|---|---|---|
| Two nuclei 14 px apart, ring and cell-restricted ring | ring never on a nucleus, never nearer the other nucleus; wide rings meet half-way; cell mode = cell minus nucleus | `test_cytoplasm_ring_stays_out_of_nuclei_and_neighbours` |
| Synthetic images: 4 wells × 3 fields × 49 cells (nucleus r 8 px, cell r 15 px), per-cell log2 N/C from N(−1, 0.25) or N(+1, 0.25), background 100 + noise, through the preset (ring) and with `expand_cells` + cell-minus-nucleus | per cell: median error < 3 % and > 90 % of cells within 5 % (seen with the ring: median 1.4 %, 95th percentile 4.4 %); well median N/C within 5 %; % responding within 1.5 pp; Z′ > 0.5 | `test_nc_ratio_per_cell_from_images` |
| Agonist series (9 points, EC50 100 nM, Hill 1) on object tables, through `hcs-studio assay run` and the pharmacology module (% inhibition) | % responding within 1 pp of truth per well; log EC50 −7.0 ± 0.1 (seen: 1.00 × 10⁻⁷ M) on % responding and on log2 N/C | `test_agonist_ec50_through_pharmacology` |
| Inhibitor mode (all wells stimulated; positive controls unstimulated; IC50 30 nM) | threshold from the positive controls; IC50 within 0.1 log | `test_inhibitor_mode_calibrates_on_positive_controls` |
| Threshold methods; cells as cytoplasm; background warnings | robust SD, percentile, Otsu and fixed thresholds all give 50 ± 1.5 % on a 50 % well; tophat/no background/polynomial warnings | `test_threshold_methods_and_errors`, `test_cells_as_cytoplasm`, `test_warnings_about_background` |

Found while building: the triangle threshold (used by the other assays) put the nuclear mask edge
3 px into the cytoplasm (nuclei 1.9× too large) and the ring then measured nucleus — resting N/C
0.75 instead of 0.5; Otsu puts the edge at half height. A polynomial background surface fitted
under ~50 % cell coverage rose into the cytoplasm and inflated N/C ratios ~7×; the preset uses a
constant background.

Not yet checked: real p65 (TNF-α vs vehicle) or YAP (density/serum) plates; blurred nuclear
edges at 20× (the gap); Cellpose cells as cytoplasm; very flat or very confluent cells.

## Apoptosis and cytotoxicity (Phase 20c)

| Check | Result | Test |
|---|---|---|
| Object tables with known phenotypes (viable, early, condensed, late, necrotic; 3 fragments per condensed/late nucleus): vehicle, staurosporine, cytotoxic, cytostatic, necrotic compound, two 0 h wells | every category within 0.5 pp of truth (seen: exact), fragments counted apart, viable counts exact; GR cytotoxic < 0, cytostatic 0–0.8; Z′ > 0.5 | `test_categories_viable_counts_and_gr_match_truth` |
| Annexin V on cells, thresholds by Otsu over vehicle + positive control | categories within 1 pp | `test_annexin_on_cells_and_otsu_on_controls` |
| Vehicle with 6 % marker-positive cells | robust-SD threshold 6 %; the 99th percentile under-calls (< 4 %) — why it is not the default | `test_vehicle_percentile_is_less_robust_than_robust_sd` |
| Morphology only (30 % condensed); classifier column instead of the rules | 30 ± 1 %; classifier classes used as given | `test_morphology_only_classifier_and_fixed_time_zero` |
| Synthetic images: 4 wells × 6 fields × 49 nuclei (DNA, caspase, PI), through the preset pipeline | nuclei within 3 %; every category within 2.5 pp (seen: vehicle exact, staurosporine within 1.7 pp) | `test_apoptosis_from_images` |

Found while building: the 99th percentile of the vehicle fell inside the positives (vehicle
wells have a few % apoptotic cells) → robust SD; time-zero wells counted as vehicle → excluded;
the mean DNA intensity of condensed nuclei was diluted by edge pixels → 95th percentile.

Not yet checked: real staurosporine/vehicle plates with caspase-3/7 and a dead-cell dye; annexin
V on real cell segmentations; very confluent wells (touching condensed nuclei).

## DNA-damage foci (Phase 20b)

| Check | Result | Test |
|---|---|---|
| Synthetic plate: 432 nuclei (radius 10 px) with 0–12 Gaussian foci (σ 1 px, ≥ 5 px apart; vehicle Poisson 0.5, treated Poisson 8) and 15 % pan-nuclear nuclei in the treated wells, through the preset pipeline | every focal nucleus within ±1 of its true count (seen: all 397 exact); all pan-nuclear nuclei recognised, no false pan-nuclear; vehicle 0.45 vs treated 6.3 foci per nucleus; threshold calibrated on the vehicle wells | `test_foci_counts_per_nucleus_from_images` |
| Two foci 8 planes apart above each other | 3D detection counts 2, a max projection 1 | `test_3d_detection_counts_foci_a_max_projection_merges` |
| Foci by cell-cycle phase (foci only above 2.6N) | G1 < 0.6, G2/M ≈ 6 foci per nucleus; % positive in G2/M as expected from Poisson(6) ≥ 5; pan-nuclear 1 %; projection warning | `test_foci_step_breaks_down_by_cell_cycle_and_warns_on_projections` |

Found while building: a fixed 3 µm² maximum focus area dropped real foci whose thresholded
response covered ~3 µm²; the default now follows the focus diameter.

Not yet checked: real γH2AX / 53BP1 / RAD51 plates (irradiated or etoposide-treated vs vehicle);
touching or clustered foci (counts of merged foci); foci at 20× vs 40×/60×.

## Cell cycle assay (Phase 20a)

| Check | Result | Test |
|---|---|---|
| Fit of known phase fractions (5000 nuclei; vehicle 2/55/25/16/2 % and arrested 4/20/14/54/8 % sub-G1/G1/S/G2M/>4N, CV 4 %) | every phase within 2 points (seen: within 0.7); CV 4.0 ± 0.5 %, G2/G1 2.00 | `test_dean_jett_fox_fit_recovers_known_phase_fractions` |
| G1 peak on a G2/M-arrested plate (65 % G2/M) | the 2N peak, not the taller 4N one | `test_g1_peak_is_found_also_on_a_g2_arrested_population` |
| Calibration on vehicle wells, EdU and pH3 | phases within 2 points (fit) and 1 point (EdU gating); EdU % and mitotic index within 0.5; arrest index 1.0 (vehicle) and 3.4 (54/16); a well stained 5 % brighter keeps its phases (G1 peak 2.1N) | `test_cell_cycle_step_calibrates_on_vehicle_wells_and_gates_edu_and_ph3` |
| From images: synthetic plate (nuclei whose summed intensity ∝ DNA content, 1152 per well) through the preset pipeline | fit within 2 points (seen 1.3), EdU gating within 1 (seen 0.4), CV 4.0 % | `test_preset_pipeline_measures_dna_content_from_images` |
| Max projection / no layout | warns; calibrates on all wells with a note | `test_warnings_for_max_projections_and_missing_layouts` |

Found while building: with Otsu's threshold the segmentation trimmed dim G1 nuclei on G2/M-rich
fields (G1 CV 13 % instead of 4 %, 9 % false sub-G1); the preset uses a triangle threshold.

Not yet checked: a real control plate (vehicle + positive control) against your current
analysis; real DAPI/Hoechst widefield vs sum-projected confocal data.

## Automatic image QC (Phase 18a)

Labeled synthetic plate (`synthetic.artifacts.qc_plate`): 256 × 256 px fields with 70–140
nuclei, background 100, noise SD 5; artifacts drawn with known positions: blur (Gaussian
σ 2–3 px), saturation (half the nuclei at the 16-bit maximum), bubble (dark disc of 15–22 %
of the field width with a bright rim), debris (bright irregular blob of 300–600 px), empty
field, gradient (illumination 1.0 → 0.3 across the field), out-of-well (dark region beyond
an arc covering 25–40 % of the field), scratch (cell-free band 22–30 % of the field wide) and
fiber (thin bright line). Default rules, nucleus diameter 4.55 µm.

| Check | Result | Test |
|---|---|---|
| Each artifact class: sensitivity (≥ 95 % required) and false positives on clean fields (≤ 5 %) | CI plate (20 fields per class, 102 clean): 100 % for every class, 0 % false positives per class (1 clean field carried the informational background flag, not excluded). Second plate (seed 7, 40 per class, 204 clean): 100 % for all classes except scratch 97.5 %; 0 % false positives | `test_each_artifact_class_is_detected_with_few_false_positives` |
| Exclusions | blur, saturation, bubble and out-of-well fields excluded, with reasons and severity; gradient, debris, fiber, scratch and empty fields flagged only | `test_decisions_and_files` |
| PLLS and Laplacian focus | blurring a field lowers both; constant image → NaN | `test_plls_follows_the_cellprofiler_definition_and_falls_with_blur` |
| Saturation level | 12-bit metadata → 4095; a 12-bit camera in a 16-bit container recognised by its maximum | `test_saturation_level_uses_metadata_or_a_camera_maximum` |
| Autofocus | best plane following a tilt across the plate is not flagged; two wells 4 planes off are | `test_plate_relative_conditions_and_autofocus_trend` |
| Well and plate decisions | too few good fields, failed controls, Z′ below the limit | `test_well_and_plate_decisions` |
| Speed | about 0.12 s per 256 × 256 two-channel field on one CPU thread; 564 fields in 34 s with 4 threads; 3.5 s per 2048 × 2048 two-channel field on one thread (after replacing a large dilation by a distance transform: was 17.8 s) | — |

The PLLS follows CellProfiler's definition (`centrosome.radial_power_spectrum.rps`, source read
2026-11-21): the image divided by its median absolute deviation from the mean, FFT power
summed in rings of radius 2 … min(H, W)/8, least-squares slope of log power on log radius.
The object ellipse axes used for debris and fibers equal scikit-image `regionprops` values
(checked to 1e-12).

### Step 18b: QC in runs, object QC, compound artifacts, propagation

| Check | Result | Test |
|---|---|---|
| QC before segmentation | in a run with QC rules, `qc_finished` precedes the first `well_started`; blur and bubble fields excluded from counts and object statistics; the report lists them with reasons | `test_qc_runs_before_segmentation_and_excludes_fields` |
| 384-well plate in the job queue (done-when) | QC of 384 fields (80 × 80 px) finishes in a real queue job before segmentation starts; the two blurred wells' fields are excluded; QC took 15 s of a 41 s run (2 workers) | `test_qc_of_a_384_well_plate_runs_in_the_queue_before_segmentation` |
| Fail fast | a plate that fails QC with `stop_run` stops before processing (status `qc_failed`) | `test_a_plate_that_fails_qc_can_stop_the_run` |
| Excluded wells and IC50 fits (done-when) | three Drug D wells at 10 and 3.3 µM with corrupted readouts: with image QC their EC50 equals the fit with the wells excluded by hand (to 1e-6) and differs from the corrupted fit by > 0.1 log; the wells are hollow, listed in `excluded_wells.csv` and the HTML report with "image QC: …" | `test_wells_failing_image_qc_leave_ic50_fits_and_are_listed` |
| Control levels | a QC-failed vehicle well is left out of the vehicle median (with a note) | `test_failed_controls_do_not_set_control_levels` |
| Object QC | of 400 nuclei, a merged object (5× area, solidity 0.7), a fragment (1/10 area) and a 20× bright object are flagged (≤ 5 flagged in all); statistics without them | `test_object_qc_flags_merged_cells_and_fragments` |
| Compound artifacts | autofluorescence rising with dose in channel 2 flags the high doses of that compound only; precipitation (brightfield particles at ≥ 3 µM) flags that compound's top doses; a clean compound and the controls are not flagged | `test_compound_autofluorescence_and_precipitation_follow_the_dose` |
| Registry | run QC summary stored; QC-failed wells flagged `qc_failed` and left out of cross-run queries | `test_registry_keeps_the_run_qc_and_skips_failed_wells` |

### Step 18c: reviews, QC tab, learned QC, report

| Check | Result | Test |
|---|---|---|
| Field review | including a blurred field (reason required) keeps the rules' decision as `rule_excluded`, rebuilds the run tables (the well's excluded fields drop by one) and is recorded (`qc.review` event, `overrides.json`) | `test_reviews_change_exclusions_everywhere_and_are_recorded` |
| Well review | failing a well with a reason marks it failed in `well_qc.csv` and the per-well summary; the report lists the review | same test |
| Learned QC | plate A: the 12 blurred fields confirmed by review, 60 clean; classifier cross-validated balanced accuracy > 0.9. Plate B (other seed, rules without the blur rule): ≥ 90 % of blurred fields flagged and excluded by learned QC, ≤ 5 % of clean fields; a modified model file is refused | `test_learned_qc_catches_an_artifact_the_rules_miss` |
| QC tab | gallery sorted by severity, failed wells hatched, reviews need a reason and update the tab | `test_qc_tab_shows_flags_and_records_reviews` |

Not yet checked: real bad and good fields (thresholds were set on synthetic artifacts;
20–50 bad fields with reasons and ~50 good fields per instrument are needed to tune them);
speed on full-size fields (2048² and larger, 4 channels); bubbles and out-of-well regions in
brightfield or phase contrast.

## Assay monitoring (Phase 17c)

| Check | Result | Test |
|---|---|---|
| Rules on a hand-made series | `1_3s`, `2of3_2s`, `7x`, `6t` fire exactly at the expected points | `test_rules_parse_and_fire_where_expected` |
| 20 stable Z′ runs (0.75 ± 0.03), then a slow decline (−0.012 per run) | no flag in the baseline; drift flagged within 8 runs | `test_control_chart_flags_a_seeded_drift_and_not_the_baseline` |
| Registry: 26 scanned runs whose control noise grows after run 20 | Z′ and vehicle-CV charts flag only runs ≥ 20; another instrument's event is not drawn | `test_z_prime_chart_of_registered_runs_flags_a_drift_and_shows_events` |
| MSR of four EC50 values (1, 2, 1, 0.5 × 10⁻⁷ M) | s = 0.24578, MSR = 10^(2√2 s) = 4.96, equal to the hand calculation; identical runs give 1 | `test_msr_matches_a_hand_calculation` |
| Test–retest of 20 compounds, run 1 2× more potent | MR 0.5, MSR and limits as hand-computed; acceptable; a 4-fold shift is not | same test |
| Bridging, same plates on two instruments | r = 1, ratio and CI as hand-computed; a 5-fold shift is not equivalent | `test_bridging_ratios_and_equivalence` |
| Bridging through the registry (3 plates, 6 fits each) | 18 ratio pairs, mean ratio 0.8–1.25, equivalent at MSR 2; not at 1.01 | `test_bridging_report_of_plates_imaged_on_two_instruments` |

Not yet checked: the MSR formulas against the NCATS Assay Guidance Manual text (implemented
from Eastwood et al. 2006); charts of real assays (choose a baseline of stable runs and the
rules with the people who run the assay); bridging on real plates imaged on two instruments.

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
