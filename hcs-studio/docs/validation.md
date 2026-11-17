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
