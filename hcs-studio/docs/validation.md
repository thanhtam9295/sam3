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
