# ShakeNBreak Quickstart

`mlip-snb` uses an MLIP to rank ShakeNBreak candidates, cluster the relaxed
structures, and optionally export the selected structures for DFT validation.

Use it when you have either:

- a bulk `POSCAR` and a defect `POSCAR`, or
- an existing ShakeNBreak output directory that already contains candidate
  structures

## Input Directory

The workflow runs from an input directory that contains a `config.yml`.

### Generation mode

```text
inputs/snb/input/
  config.yml
  bulk/POSCAR
  defect/POSCAR
  vasp_inputs/
```

### Import mode

```text
inputs/snb/input/
  config.yml
  snb/
    ...
  vasp_inputs/
```

In generation mode, `bulk/POSCAR` and `defect/POSCAR` are the starting
structures. In import mode, the `snb/` directory already contains the
ShakeNBreak candidate structures and `mlip-snb` skips candidate generation.

## `config.yml`

The workflow uses one top-level `snb:` section with two nested blocks:
`defaults` and `settings`.

```yaml
snb:
  defaults:
    # Model name registered in SUPPORTED_MODELS.yml.
    model_name: mace-mpa-0-medium
    # Root directory for the results tree.
    results_root: resultsSNB
    # Root directory that contains the model checkpoints.
    models_root: assets/models

    # Bulk structure used to generate candidate distortions.
    bulk: bulk/POSCAR
    # Defect structure used to generate candidate distortions.
    defect: defect/POSCAR
    # Existing ShakeNBreak directory to import instead of generating candidates.
    snb_dir: null
    # Element-to-oxidation-state map used for distortion generation.
    oxidation_states:
      Cs: 1
      Pb: 2
      I: -1

    # VASP input folder copied into exported validation directories.
    vasp_inputs_dir: vasp_inputs
    # Export DFT validation folders after the MLIP selection step.
    prepare_dft: true
    # Copy POTCAR instead of symlinking it.
    copy_potcar: false
    # Use the dispersion-corrected calculator path when available.
    include_vdw: true
    # Overwrite the output directory instead of resuming.
    overwrite: false
    # Calculator device.
    device: cuda
    # Calculator dtype.
    dtype: float32

  settings:
    # Maximum force allowed during the MLIP relaxation.
    fmax: 0.03
    # Maximum number of relaxation steps.
    max_steps: 600
    # Energy window used to keep candidates for DFT export.
    energy_window_eV: 0.50
    # Maximum number of cluster representatives exported per model.
    max_clusters_per_model: 10
    # Length tolerance used when clustering structures.
    matcher_ltol: 0.2
    # Species-position tolerance used when clustering structures.
    matcher_stol: 0.3
    # Angle tolerance used when clustering structures.
    matcher_angle_tol: 5
```

## Configuration Reference

| Path | Type | Default | Meaning |
| --- | --- | --- | --- |
| `snb.defaults.model_name` | string | `mace-mpa-0-medium` | Model name registered in `SUPPORTED_MODELS.yml`. |
| `snb.defaults.results_root` | path | `resultsSNB` | Root directory for the output tree. |
| `snb.defaults.models_root` | path | `assets/models` | Root directory that contains the model checkpoints. |
| `snb.defaults.bulk` | path | optional | Bulk `POSCAR` used for candidate generation. |
| `snb.defaults.defect` | path | optional | Defect `POSCAR` used for candidate generation. |
| `snb.defaults.snb_dir` | path | optional | Existing ShakeNBreak directory to import instead of generating candidates. |
| `snb.defaults.oxidation_states` | map | optional | Element-to-oxidation-state map required for candidate generation. |
| `snb.defaults.vasp_inputs_dir` | path | optional | Folder copied into exported DFT validation directories. |
| `snb.defaults.prepare_dft` | bool | `false` | Export DFT validation folders after the MLIP selection step. |
| `snb.defaults.copy_potcar` | bool | `false` | Copy POTCAR instead of symlinking it. |
| `snb.defaults.include_vdw` | bool | `true` | Use the dispersion-corrected calculator path when supported. |
| `snb.defaults.overwrite` | bool | `false` | Overwrite the output tree instead of resuming. |
| `snb.defaults.device` | string | `cuda` | Calculator device. |
| `snb.defaults.dtype` | string | `float32` | Calculator dtype. |
| `snb.settings.fmax` | float | `0.03` | Maximum force allowed during the MLIP relaxation. |
| `snb.settings.max_steps` | int | `600` | Maximum number of MLIP relaxation steps. |
| `snb.settings.energy_window_eV` | float | `0.50` | Energy window used to keep candidates for DFT export. |
| `snb.settings.max_clusters_per_model` | int | `10` | Maximum number of cluster representatives exported per model. |
| `snb.settings.matcher_ltol` | float | `0.2` | Length tolerance used when clustering structures. |
| `snb.settings.matcher_stol` | float | `0.3` | Species-position tolerance used when clustering structures. |
| `snb.settings.matcher_angle_tol` | float | `5` | Angle tolerance used when clustering structures. |

## Command Variants

| Command | Use when |
| --- | --- |
| `mlip-snb --inputs <input-dir>` | The input directory already contains `config.yml`. |
| `mlip-snb --inputs <input-dir> --prepare-dft` | You want DFT validation folders exported after the MLIP selection step. |
| `mlip-snb --inputs <input-dir> check-dft` | You have finished DFT and want to check the returned structures. |
| `mlip-snb --inputs <input-dir> compare-dft` | You want the MLIP-vs-DFT comparison report. |
| `mlip-snb --inputs <input-dir> report` | You want the final report output. |
| `mlip-snb --config <path/to/config.yml>` | The config file is not inside the input directory. |

## Outputs

The workflow writes a results tree under `snb.defaults.results_root`.
The exact subdirectory name is derived from the input directory label unless you
override it with the workflow’s label option.

Expected artifacts include:

- MLIP-relaxed candidate clusters
- selected DFT validation folders when `prepare_dft: true`
- comparison CSV files
- analysis reports

## Common Mistakes

- `bulk/POSCAR` and `defect/POSCAR` are only needed in generation mode.
- If you import an existing ShakeNBreak directory, set `snb.defaults.snb_dir`
  and leave the generation-only paths empty.
- `prepare_dft: true` creates the validation folders, but it does not run DFT.
- The final report is a post-processing step; it does not replace the DFT
  relaxations themselves.
