# ShakeNBreak Workflow

`mlip-snb` generates or consumes ShakeNBreak candidates, relaxes them with an
MLIP, clusters the relaxed structures, and optionally prepares DFT follow-up
folders.

Run it from an input directory:

```bash
mlip-snb --inputs inputs/snb/input
```

Input directory layout, generation mode:

```text
inputs/snb/input/
  config.yml
  bulk/POSCAR
  defect/POSCAR
  vasp_inputs/
```

Input directory layout, import mode:

```text
inputs/snb/input/
  config.yml
  snb/
    ...
```

The `bulk/POSCAR` + `defect/POSCAR` layout is for generating candidates from a
bulk/defect pair. The `snb/` layout is for reusing an existing ShakeNBreak
candidate tree.

Example `config.yml`:

```yaml
snb:
  defaults:
    # Model name registered in SUPPORTED_MODELS.yml.
    model_name: mace-mpa-0-medium
    # Root directory for this input directory.
    results_root: resultsSNB
    # Where the model files live.
    models_root: assets/models
    # Use either bulk/defect inputs or an existing snb_dir.
    bulk: bulk/POSCAR
    defect: defect/POSCAR
    snb_dir: null
    # Oxidation states are required when generating candidates from POSCARs.
    oxidation_states:
      Cs: 1
      Pb: 2
      I: -1
    # Optional DFT input folder.
    vasp_inputs_dir: vasp_inputs
    # Prepare DFT folders after selection.
    prepare_dft: true
    # Copy POTCAR instead of symlinking it.
    copy_potcar: false
    # Use the dispersion-corrected calculator path when supported.
    include_vdw: true
    # Overwrite previous outputs.
    overwrite: false
    # Calculator device and dtype.
    device: cuda
    dtype: float32

  settings:
    # MLIP relaxation force threshold.
    fmax: 0.03
    # MLIP relaxation step limit.
    max_steps: 600
    # Selection window above the cluster ground state.
    energy_window_eV: 0.50
    # Maximum selected clusters per model.
    max_clusters_per_model: 10
    # StructureMatcher tolerances.
    matcher_ltol: 0.2
    matcher_stol: 0.3
    matcher_angle_tol: 5.0
```

Configuration reference:

| Path | Type | Default | Meaning |
| --- | --- | --- | --- |
| `snb.defaults.model_name` | string | `mace-mpa-0-medium` | Model name registered in `SUPPORTED_MODELS.yml`. |
| `snb.defaults.results_root` | path | `resultsSNB` | Root directory for this input directory. |
| `snb.defaults.models_root` | path | `assets/models` | Root directory for MLIP checkpoints. |
| `snb.defaults.bulk` | path | optional | Bulk POSCAR for candidate generation. |
| `snb.defaults.defect` | path | optional | Defect POSCAR for candidate generation. |
| `snb.defaults.snb_dir` | path | optional | Existing ShakeNBreak output directory to import. |
| `snb.defaults.oxidation_states` | map | optional | Element-to-charge map required when generating candidates from POSCARs. |
| `snb.defaults.vasp_inputs_dir` | path | optional | Folder copied into exported DFT validation folders. |
| `snb.defaults.prepare_dft` | bool | `false` | Export VASP validation folders after selection. |
| `snb.defaults.copy_potcar` | bool | `false` | Copy POTCAR instead of symlinking it. |
| `snb.defaults.include_vdw` | bool | `true` | Use the dispersion-corrected calculator path when supported. |
| `snb.defaults.overwrite` | bool | `false` | Overwrite the output directory instead of resuming. |
| `snb.defaults.device` | string | `cuda` | Calculator device. |
| `snb.defaults.dtype` | string | `float32` | Calculator dtype. |
| `snb.settings.fmax` | float | `0.03` | Force threshold for MLIP relaxations. |
| `snb.settings.max_steps` | int | `600` | Max optimizer steps for MLIP relaxations. |
| `snb.settings.energy_window_eV` | float | `0.50` | Energy window used during selection. |
| `snb.settings.max_clusters_per_model` | int | `10` | Maximum clusters to export per model. |
| `snb.settings.matcher_ltol` | float | `0.2` | StructureMatcher length tolerance. |
| `snb.settings.matcher_stol` | float | `0.3` | StructureMatcher site tolerance. |
| `snb.settings.matcher_angle_tol` | float | `5.0` | StructureMatcher angle tolerance. |

Command variants:

| Command | Use when |
| --- | --- |
| `mlip-snb --inputs <input-dir>` | The input directory already contains `config.yml`. |
| `mlip-snb --inputs <input-dir> run` | You want the full generation/import + relax + cluster + select pipeline explicitly. |
| `mlip-snb --inputs <input-dir> check-dft` | You already have DFT validation folders and want to parse them. |
| `mlip-snb --inputs <input-dir> compare-dft` | You want the MLIP-vs-DFT comparison tables. |
| `mlip-snb --inputs <input-dir> report` | You want the human-readable report. |
| `mlip-snb --config <path/to/config.yml>` | The config file is not inside the input directory. |

Outputs:

- generated or imported candidate structures
- relaxation summaries
- cluster assignments
- DFT selection manifests
- optional VASP validation folders
- comparison and report files

The results tree is written under `snb.defaults.results_root` and grouped by the
input label inferred from the directory name or the optional label override.
