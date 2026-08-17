# Example 3: three-model NEB benchmark

This example runs one fixed NEB problem for three registered MACE models and
keeps their outputs in one results root. The DFT reference folder contains
the eight converged image structures and `neb.dat`; it is used only for
evaluation, never as an MLIP initial path.

Run all three model calculations:

```bash
mlip-neb --inputs demo/neb/example3_benchmark/input
```

The command fans out once per model and writes:

```text
demo/neb/example3_benchmark/outputs/
  small-omat-0/raw/
  mace-omat-0-medium/raw/
  mace-mpa-0-medium/raw/
```

After all three paths exist, generate the combined benchmark report:

```bash
mlip-neb --inputs demo/neb/example3_benchmark/input --report-benchmark
```

This creates per-model and family-level energy-profile and path-fidelity
plots in `input/plot/`, plus `report.md` and `report.json`. The report compares
barrier error, energy-profile RMSE, and geometric path displacement. It is a
comparison of these three models on this one migration pathway, not a general
model ranking.
