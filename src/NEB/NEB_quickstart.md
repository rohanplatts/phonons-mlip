# NEB Quickstart

`mlip-neb` builds a minimum-energy path between two endpoint structures and can
optionally export VASP-ready follow-up folders.

Use it when you have:

- an initial endpoint `POSCAR`
- a final endpoint `POSCAR`
- optionally, a `neb.dat` file that defines the raw DFT path and image count

Run it from an input directory:

```bash
mlip-neb --inputs path/to/neb-input
```

## Input Directory

```text
path/to/neb-input/
  config.yml
  POSCAR_i
  POSCAR_f
  neb.dat          # optional
  vasp_inputs/     # optional
```

## `config.yml`

The workflow uses a `defaults:` section and a `workflows.neb.defaults:` section.

```yaml
defaults:
  # Model name registered in SUPPORTED_MODELS.yml.
  model_name: mace-mpa-0-medium
  # Root directory for NEB outputs.
  outputs_root: resultsNEB
  # Root directory that contains the model checkpoints.
  models_root: assets/models
  # Initial endpoint structure.
  poscar_i: POSCAR_i
  # Final endpoint structure.
  poscar_f: POSCAR_f
  # Optional raw DFT NEB path file.
  dft_neb_dat: neb.dat
  # Optional fallback directory for paths and endpoints.
  structures_dir: assets/structures/NEB
  # Optional VASP input folder copied into exported image folders.
  vasp_inputs_dir: vasp_inputs
  # Calculator device and dtype.
  device: cuda
  dtype: float32
  # Keep the endpoints fixed unless you explicitly want relaxation.
  relax_endpoints: false
  # Remap final to initial species ordering when needed.
  remap_f_i: true
  # Use the dispersion-corrected calculator path when available.
  include_vdw: true
  # Overwrite existing outputs instead of resuming.
  overwrite: false

workflows:
  neb:
    defaults:
      # Fallback image count when neb.dat does not define one.
      n_images_fallback: 9
      # First rough relaxation stage.
      maxstep_mlip_guess: 0.05
      fmax_mlip_guess: 0.03
      steps_mlip_guess: 3000
      k_spring_mlip: 0.6
      # Refinement stage.
      k_spring: 0.6
      maxstep_mlip_d3: 0.03
      fmax_mlip_d3: 0.03
      steps_mlip_d3: 1400
      # Climbing-image stage.
      maxstep_ci: 0.03
      fmax_ci: 0.03
      steps_ci: 1000
```

## Configuration Reference

| Path | Type | Default | Meaning |
| --- | --- | --- | --- |
| `defaults.model_name` | string | `ivac0_neb_ft` | Model name registered in `SUPPORTED_MODELS.yml`. |
| `defaults.outputs_root` | path | `resultsNEB` | Root directory for NEB outputs. |
| `defaults.models_root` | path | `assets/models` | Root directory for MLIP checkpoints. |
| `defaults.poscar_i` | path | `POSCAR_i` | Initial endpoint structure file. |
| `defaults.poscar_f` | path | `POSCAR_f` | Final endpoint structure file. |
| `defaults.dft_neb_dat` | path | optional | Raw DFT path file used to infer the number of images. |
| `defaults.structures_dir` | path | `assets/structures/NEB` | Fallback directory for endpoints and `neb.dat`. |
| `defaults.vasp_inputs_dir` | path | optional | Folder copied into exported VASP image directories. |
| `defaults.device` | string | `cuda` | Calculator device. |
| `defaults.dtype` | string | `float32` | Calculator dtype. |
| `defaults.relax_endpoints` | bool | `false` | Relax the endpoints before building the NEB path. |
| `defaults.remap_f_i` | bool | `true` | Remap final to initial species ordering before interpolation. |
| `defaults.include_vdw` | bool | `true` | Use the dispersion-corrected calculator path when supported. |
| `defaults.overwrite` | bool | `false` | Overwrite the output tree instead of resuming. |
| `workflows.neb.defaults.n_images_fallback` | int | `9` | Fallback image count when `neb.dat` does not specify one. |
| `workflows.neb.defaults.maxstep_mlip_guess` | float | `0.05` | Max atomic step for the first rough relaxation. |
| `workflows.neb.defaults.fmax_mlip_guess` | float | `0.03` | Force threshold for the first rough relaxation. |
| `workflows.neb.defaults.steps_mlip_guess` | int | `3000` | Step limit for the first rough relaxation. |
| `workflows.neb.defaults.k_spring_mlip` | float | `0.6` | Spring constant for the first rough relaxation. |
| `workflows.neb.defaults.k_spring` | float | `0.6` | Spring constant for the refinement stages. |
| `workflows.neb.defaults.maxstep_mlip_d3` | float | `0.03` | Max step for the D3 refinement stage. |
| `workflows.neb.defaults.fmax_mlip_d3` | float | `0.03` | Force threshold for the D3 refinement stage. |
| `workflows.neb.defaults.steps_mlip_d3` | int | `1400` | Step limit for the D3 refinement stage. |
| `workflows.neb.defaults.maxstep_ci` | float | `0.03` | Max step for the climbing-image stage. |
| `workflows.neb.defaults.fmax_ci` | float | `0.03` | Force threshold for the climbing-image stage. |
| `workflows.neb.defaults.steps_ci` | int | `1000` | Step limit for the climbing-image stage. |

## Command Variants

| Command | Use when |
| --- | --- |
| `mlip-neb --inputs <input-dir>` | The input directory already contains `config.yml`. |
| `mlip-neb --inputs <input-dir> --no-relax-endpoints` | You want to preserve the supplied endpoints exactly. |
| `mlip-neb --inputs <input-dir> --compare` | You want to run the comparison/reporting path instead of a fresh NEB run. |
| `mlip-neb --inputs <input-dir> --report-benchmark` | You want the baseline-vs-fine-tuned benchmark report for a two-model benchmark config. |
| `mlip-neb --config <path/to/config.yml>` | The config file is not inside the input directory. |

## Outputs

The workflow writes:

- optimizer logs
- trajectories
- `neb_raw.npz`
- summary text
- VASP-ready image folders when requested

The raw NEB tree is written under `defaults.outputs_root`, grouped by model
name.
