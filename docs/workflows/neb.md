# NEB Workflow

`mlip-neb` runs a minimum-energy path calculation between two endpoint
structures and can optionally export VASP-ready single-point folders.

Run it from an input directory:

```bash
mlip-neb --inputs inputs/neb/input
```

Required input directory layout:

```text
inputs/neb/input/
  config.yml
  POSCAR_i
  POSCAR_f
  neb.dat            # optional
  INCAR              # optional
  KPOINTS            # optional
  POTCAR             # optional
```

`POSCAR_i` and `POSCAR_f` are required. `neb.dat` is optional but useful
because it can specify the image count and the raw DFT path.

Example `config.yml`:

```yaml
defaults:
  # Model name registered in SUPPORTED_MODELS.yml.
  model_name: mace-mpa-0-medium
  # Root directory for results from this input directory.
  outputs_root: results/neb
  # Where the model checkpoints live.
  models_root: assets/models
  # Initial endpoint POSCAR, relative to the input directory.
  poscar_i: POSCAR_i
  # Final endpoint POSCAR, relative to the input directory.
  poscar_f: POSCAR_f
  # Optional DFT reference path for the raw NEB input.
  dft_neb_dat: neb.dat
  # Optional fallback directory that contains POSCAR_i/POSCAR_f/neb.dat.
  structures_dir: assets/structures/NEB
  # Optional DFT VASP input folder for exported endpoint images.
  vasp_inputs_dir: vasp_inputs
  # Calculator device and dtype.
  device: cuda
  dtype: float32
  # Whether to relax the endpoints before the NEB.
  relax_endpoints: false
  # Whether to remap atom ordering between endpoints.
  remap_f_i: true
  # Whether to use dispersion when the model supports it.
  include_vdw: true
  # Overwrite existing outputs instead of resuming.
  overwrite: false

workflows:
  neb:
    defaults:
      # Fallback number of images when the DFT path does not define one.
      n_images_fallback: 7
      # First-stage rough relaxation settings.
      maxstep_mlip_guess: 0.05
      fmax_mlip_guess: 0.05
      steps_mlip_guess: 200
      k_spring_mlip: 0.1
      # Second-stage relaxation settings.
      k_spring: 0.1
      maxstep_mlip_d3: 0.05
      fmax_mlip_d3: 0.03
      steps_mlip_d3: 500
      # Climbing-image refinement settings.
      maxstep_ci: 0.05
      fmax_ci: 0.03
      steps_ci: 200
```

Configuration reference:

| Path | Type | Default | Meaning |
| --- | --- | --- | --- |
| `defaults.model_name` | string | `ivac0_neb_ft` | Model name registered in `SUPPORTED_MODELS.yml`. |
| `defaults.outputs_root` | path | `resultsNEB` | Root directory for outputs from this input directory. |
| `defaults.models_root` | path | `assets/models` | Root directory for MLIP checkpoints. |
| `defaults.poscar_i` | path | `POSCAR_i` | Initial endpoint structure file. |
| `defaults.poscar_f` | path | `POSCAR_f` | Final endpoint structure file. |
| `defaults.dft_neb_dat` | path | optional | DFT NEB path file used to infer the number of images. |
| `defaults.structures_dir` | path | `assets/structures/NEB` | Fallback folder for endpoints and `neb.dat`. |
| `defaults.vasp_inputs_dir` | path | optional | Folder copied into exported VASP image directories. |
| `defaults.device` | string | `cuda` | Calculator device. |
| `defaults.dtype` | string | `float32` | Calculator dtype. |
| `defaults.relax_endpoints` | bool | `true` | Relax the endpoints before the NEB path is built. |
| `defaults.remap_f_i` | bool | `false` | Remap final to initial species ordering before interpolation. |
| `defaults.include_vdw` | bool | `true` | Use the dispersion-corrected calculator path when available. |
| `defaults.overwrite` | bool | `false` | Overwrite the output directory instead of resuming. |
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

Command variants:

| Command | Use when |
| --- | --- |
| `mlip-neb --inputs <input-dir>` | The input directory already contains `config.yml`. |
| `mlip-neb --inputs <input-dir> --no-relax-endpoints` | You want to preserve the endpoint structures exactly as supplied. |
| `mlip-neb --inputs <input-dir> --compare` | You want the NEB comparison/reporting path instead of a fresh NEB run. |
| `mlip-neb --inputs <input-dir> --report-benchmark` | You want the baseline-vs-fine-tuned benchmark report for a two-model benchmark config. |
| `mlip-neb --config <path/to/config.yml>` | The config file is not inside the input directory. |

Outputs:

- optimizer logs
- trajectories
- `neb_raw.npz`
- summary text
- VASP-ready image folders

The raw NEB tree is written under `defaults.outputs_root`, grouped by model
name.
