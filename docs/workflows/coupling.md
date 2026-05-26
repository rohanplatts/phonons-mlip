# Phonon Coupling Workflow

`mlip-coup` compares DFT and MLIP phonon-coupling spectra from one input
directory. It reads a `config.yml`, the two endpoint structures, one DFT
`band.yaml`, and one or more MLIP `band.yaml` files, then writes a report and
the derived plots.

Run it from an input directory:

```bash
mlip-coup --inputs inputs/coupling/input
```

## Input Directory

Required files:

```text
inputs/coupling/input/
  config.yml
  CONTCAR_GS
  CONTCAR_ES
  band_dft.yaml
  ml/
    model_a/band.yaml
    model_b/band.yaml
```

The `ml/` tree is optional if `band_ml_paths` is listed in `config.yml`.
If `band_ml_paths` is empty, `mlip-coup` looks for MLIP `band.yaml` files
under the input directory itself, then under `results/` inside the input
directory, and finally under `--outputs` if you provide one.

## `config.yml`

The workflow uses one top-level section: `phonon_coupling`.

```yaml
phonon_coupling:
  # Ground-state structure used for the DFT reference.
  contcar_gs: CONTCAR_GS
  # Excited-state structure used for the DFT reference.
  contcar_es: CONTCAR_ES
  # DFT phonon-coupling band file.
  band_dft_path: band_dft.yaml
  # One or more MLIP band files. Leave empty to auto-discover them.
  band_ml_paths:
    - ml/model_a/band.yaml
    - ml/model_b/band.yaml
  # q-point matching tolerance.
  q_tol: 1.0e-4
  # Lattice-matching tolerance between the two endpoint structures.
  lattice_tol: 1.0e-5
  # Cumulative weight threshold used to choose the important modes.
  threshold: 0.9
  # Frequency clustering tolerance for grouping nearby modes.
  freq_cluster_tol: 0.5
  # Frequency window used when collecting nearby modes around a cluster.
  freq_window: 0.5
  # Remove mass-weighted centre-of-mass motion before analysis.
  remove_mass_weighted_com: true
  # If true, only Gamma-like q-points are kept.
  gamma_only: true
  # Weighting exponent used in the DFT score.
  alpha: 1.3
  # Weighting scheme: p, S, or lambda.
  weight_kind: S
```

## Configuration Reference

| Path | Type | Default | Meaning |
| --- | --- | --- | --- |
| `phonon_coupling.contcar_gs` | path | `CONTCAR_GS` | Ground-state endpoint structure. |
| `phonon_coupling.contcar_es` | path | `CONTCAR_ES` | Excited-state endpoint structure. |
| `phonon_coupling.band_dft_path` | path | `band_dft.yaml` | DFT reference band file. |
| `phonon_coupling.band_ml_paths` | list[path] | auto-discovered | MLIP band files to compare against DFT. |
| `phonon_coupling.q_tol` | float | `1e-4` | Maximum q-point mismatch allowed when aligning bands. |
| `phonon_coupling.lattice_tol` | float | `1e-5` | Maximum lattice mismatch allowed between endpoints. |
| `phonon_coupling.threshold` | float | `0.9` | Cumulative weight threshold used when selecting modes. |
| `phonon_coupling.freq_cluster_tol` | float | `0.5` | Frequency gap threshold for clustering nearby modes. |
| `phonon_coupling.freq_window` | float | `0.5` | Window around a selected cluster used when collecting summary modes. |
| `phonon_coupling.remove_mass_weighted_com` | bool | `true` | Remove the mass-weighted centre-of-mass shift before analysis. |
| `phonon_coupling.gamma_only` | bool | `true` | Keep only Gamma-like q-points when comparing spectra. |
| `phonon_coupling.alpha` | float | `1.3` | Weighting exponent used in the report score. |
| `phonon_coupling.weight_kind` | string | `S` | Weighting mode: `p`, `S`, or `lambda`. |

## Command Variants

| Command | Use when |
| --- | --- |
| `mlip-coup --inputs <input-dir>` | The input directory already contains `config.yml`. |
| `mlip-coup --inputs <input-dir> --outputs <dir>` | You want the report files written somewhere else. |
| `mlip-coup --config <path/to/config.yml>` | The config file lives outside the input directory. |
| `mlip-coup --band_ml <path>` | You want to override auto-discovery and provide MLIP band files explicitly. |
| `mlip-coup --help` | You want the parser-level CLI reference. |

## Outputs

The workflow writes a text report and the derived plots to the report directory.
By default that is `input-dir/results/`.

Expected artifacts include:

- the rendered report text
- per-q summaries
- coupling-cluster statistics
- the summary plots referenced by the report

## Common Mistakes

- The DFT band file is `band_dft.yaml` in the config, not `band.yaml` by default.
- If you do not list MLIP band files in `band_ml_paths`, the workflow must be
  able to discover them from the input directory or `results/`.
- `--outputs` changes where the report is written, but it does not change the
  input directory that `config.yml` is read from.
