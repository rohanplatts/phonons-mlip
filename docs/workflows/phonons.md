# Phonon Workflow

`mlip-phonons` builds force constants, band structures, and DOS curves from an
input directory plus a local `config.yml`.

Run it from an input directory:

```bash
mlip-phonons --inputs inputs/phonons/input
```

Required input directory layout:

```text
inputs/phonons/input/
  config.yml
  POSCAR
  primitive.poscar
```

`POSCAR` is the main cell. `primitive.poscar` is optional, but if you provide
it the workflow can use it directly instead of inferring a primitive cell.

Example `config.yml`:

```yaml
models:
  mace-mpa-0-medium:
    # Environment name from SUPPORTED_MODELS.yml.
    environment: mace_env
    # Model file relative to assets/models/mace/.
    model_path: assets/models/mace/mace-mpa-0-medium.model
    # Optional default structure label for this model.
    material: hbn
  # Add one entry per supported model you want this workflow to know about.

executive:
  # Turn plots on or off.
  plots: true
  # Turn Plumipy-ready exports on or off.
  plumipy: true
  # Where to write the results tree for this run.
  results_root: results/phonons
  # Raw files live here under results_root.
  raw_subdir: raw
  # Plot files live here under results_root.
  plot_subdir: plot
  # Optional filename overrides for the generated files.
  output_names:
    relax_traj: "{base}_relax.traj"
    relaxed_poscar: "{base}_relaxed.poscar"
    phonons_obj: "{base}_phonons.yaml"
    force_constants: "{base}_force_constants.yaml"
    phonon_dos_npz: "{base}_phonon_dos.npz"
    phonon_band_yaml: "{base}_phonon_band.yaml"
    phonon_band_plot: "{base}_phonon_band_plot.png"
    phonon_dispersion_dos_plot: "{base}_phonon_dispersion_dos.png"
    phonon_dos_plot: "{base}_phonon_dos.png"
    band_plumipy: "{base}_band.yaml"
    contcar_gs_plumipy: "{base}_CONTCAR_GS"
    outcar_gs_plumipy: "{base}_OUTCAR_GS"

structures:
  pure:
    # name of pure material:
    pure_material_example: 
      # Unit cell or relaxed structure used to generate phonons.
      unitcell_path: POSCAR
      # Optional primitive cell; omit if the workflow can infer it.
      primitive_cell_path: primitive.poscar
      # True when the supplied structure is already relaxed.
      is_file_relaxed: true
      # Supercell used for displacements.
      supercell_matrix: [3, 3, 3]
      # Displacement amplitude in angstrom.
      delta: 0.01
      # Whether to generate the band structure path.
      want_band_structure: true
      # DOS mesh.
      kpts: [12, 12, 12]
      # Number of points per band segment.
      npts: 400
      # Gaussian broadening in eV.
      width_ev: 0.001
  defects:
    # name of defect material:
    vacancy_example:
      unitcell_path: POSCAR
      primitive_cell_path: primitive.poscar
      is_file_relaxed: false
      supercell_matrix: [2, 2, 2]
      delta: 0.01
      want_band_structure: false
      kpts: [8, 8, 8]
      npts: 300
      width_ev: 0.001
```

Configuration reference:

| Path | Type | Default | Meaning |
| --- | --- | --- | --- |
| `models.<name>.environment` | string | required | Conda environment for that model. |
| `models.<name>.model_path` | path | required | Checkpoint path relative to `assets/models/<family>/`. |
| `models.<name>.material` | string | optional | Default structure label for that model. |
| `executive.plots` | bool | `false` | Generate plots in addition to raw data. |
| `executive.plumipy` | bool | `false` | Generate Plumipy-ready export files. |
| `executive.results_root` | path | `results` | Root directory for all phonon outputs. |
| `executive.raw_subdir` | string | `raw` | Name of the raw-output folder. |
| `executive.plot_subdir` | string | `plot` | Name of the plot folder. |
| `executive.output_names.*` | string | built-in template | Override filenames for generated artifacts. |
| `structures.pure.<name>.unitcell_path` | path | required | Main structure file. |
| `structures.pure.<name>.primitive_cell_path` | path | optional | Primitive cell file. |
| `structures.pure.<name>.is_file_relaxed` | bool | `false` | Set when the supplied structure is already relaxed. |
| `structures.pure.<name>.supercell_matrix` | 3 ints or 3x3 | required | Supercell used for phonon displacements. |
| `structures.pure.<name>.delta` | float | `0.01` | Displacement amplitude in angstrom. |
| `structures.pure.<name>.want_band_structure` | bool | `true` | Compute the band path. |
| `structures.pure.<name>.kpts` | 3 ints | `12, 12, 12` | DOS k-point mesh. |
| `structures.pure.<name>.npts` | int | `400` | Number of points per band segment. |
| `structures.pure.<name>.width_ev` | float | `0.0` | Gaussian broadening in eV. |
| `structures.defects.<name>.*` | same as above | same | Same fields, but for defect structures. |

Command variants:

| Command | Use when |
| --- | --- |
| `mlip-phonons --inputs <input-dir>` | The input directory already contains `config.yml`. |
| `mlip-phonons --config <path/to/config.yml>` | The config file lives elsewhere. |
| `mlip-phonons --inputs <input-dir> --outputs <dir>` | You want to override the results root without editing the config. |
| `mlip-phonons --help` | You want the parser-level CLI reference. |

Outputs:

- raw phonon data under `executive.results_root`
- plots under `executive.plot_subdir`
- Plumipy-ready export files under `raw/Plumipy_Files` when `executive.plumipy` is true

The key rule is:

- model selection lives under `models`
- output naming lives under `executive`
- structure selection lives under `structures.pure` or `structures.defects`
