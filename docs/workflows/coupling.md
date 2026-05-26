# Phonon Coupling Workflow

Run:

```bash
mlip-coupling --inputs inputs/coupling/case --outputs results/coupling/case
```

The input directory contains physical files:

- `CONTCAR_GS`
- `CONTCAR_ES`
- DFT `band.yaml`, preferably named `band_dft.yaml`
- one or more MLIP `band.yaml` files in subdirectories, or explicit `--band-ml` paths

Config/flags contain intent and parameters:

- `threshold`
- `freq_cluster_tol`
- `freq_window`
- `alpha`
- `weight_kind`
- `gamma_only`
- `q_tol`
- `lattice_tol`

Config-only execution reads the local `config.yml` in the input folder:

```bash
mlip-coupling
```
