# Reference data for the pharmacology regression tests

Put reference files here; `tests/test_pharmacology.py` picks them up automatically (the
tests are skipped while the folders are empty).

## `prism/` — GraphPad Prism 4PL fits

For each curve, two files with the same name:

- `<name>.csv`: the Prism XY table. First column X (log10 of the concentration in M, or
  the concentration itself), then one column per replicate (Y1, Y2, …). Empty cells allowed.
- `<name>.json`: Prism's fit ("log(inhibitor) vs. response — variable slope"):
  `{"log_ec50": -6.52, "hill": -1.31, "x_is_log": true}` (or `"ec50"` in the X unit).

Pass: EC50 within 5 %, Hill slope within 0.1.

## `synergyfinder/` — reference combination matrices

- `<name>.csv`: rows = drug A concentrations (first column, ascending, 0 first), header =
  drug B concentrations (0 first), values = % inhibition.
- `<name>.json`: SynergyFinder's summary scores and the tolerance, e.g.
  `{"bliss": 4.2, "hsa": 7.9, "loewe": 1.1, "zip": 3.4, "tolerance": 1.0,
  "synergyfinder_version": "3.x", "options": "baseline correction off"}`.
