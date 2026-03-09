# `mlip_phonons`

`mlip-phonons` is the MLIP phonon workflow in this repo. It takes a structure from `config.yml`, loads an ASE calculator for a named MLIP, optionally relaxes the structure, runs finite-displacement phonons with Phonopy, computes a total phonon DOS, optionally computes a phonon band structure, and writes plots plus a small set of Plumipy-compatible files.

This package is for MLIP-only phonon generation. It does not do DFT itself, and it does not do model ranking or mode-coupling comparison; those live elsewhere in the repo.

## What It Does

For one `(model, structure)` run, the pipeline in [main.py](main.py) does the following:

1. Resolves the requested structure from `config.yml`.
2. Loads the MLIP calculator from [get_calc.py](get_calc.py).
3. Optionally relaxes the structure if `is_file_relaxed: false`.
4. Builds Phonopy displaced supercells and evaluates MLIP forces on each one.
5. Produces force constants and saves a Phonopy YAML object.
6. If enabled, computes a phonon band structure on an ASE/AFLOW-style high-symmetry path.
7. Computes the total phonon DOS on the configured mesh.
8. Optionally writes DOS / band / band+DOS figures.
9. Writes a Plumipy-ready `CONTCAR_GS`, synthetic `OUTCAR_GS`, and Gamma-only `band.yaml`.

## CLI

The package installs the `mlip-phonons` command via [pyproject.toml](../../pyproject.toml).

```bash
mlip-phonons <model_name>
mlip-phonons <model_name> --structure <structure_key>
mlip-phonons --config path/to/config.yml
```

Arguments:

- `model_name`: optional model key from `config.yml -> models`.
- `--structure`: optional override for the structure key under `structures.pure` or `structures.defects`.
- `--config`: path to `config.yml` (default `<repo>/config.yml`). 

If `--structure` is omitted, the code uses `mlip_phonons.defaults.structure` first, then `models.<model>.material` as a fallback. That `material` field is effectively a legacy name for the model's default structure key.

## Required Inputs

At minimum, a run needs:

- A valid `config.yml`.
- A `model_name` that exists both in `config.yml -> models` and in the calculator registry in [get_calc.py](get_calc.py).
- Model weights/checkpoints under `assets/models/` or another directory reachable through the configured `assets_root`.
- A structure entry under `structures.pure` or `structures.defects`.

The structure entry is read through [config_classes.py](config_classes.py) and should provide:

- `unitcell_path`: required.
- `primitive_cell_path`: optional, mainly for pure materials.
- `is_file_relaxed`: whether to skip the relaxation step.
- `supercell_matrix`: Phonopy supercell definition.
- `delta`: finite-displacement amplitude in angstrom.
- `want_band_structure`: whether to compute and save `band.yaml`.
- `kpts`: DOS mesh.
- `npts`: points per high-symmetry path segment.

## Pure vs Defect Workflows

The pipeline treats pure materials and defects differently:

- `structures.pure`: can use a separate primitive cell. For pure systems, the code tries to build a standard primitive-cell phonon workflow when the provided files are consistent.
- `structures.defects`: assumes the supplied structure is already the full cell to analyse. Defects are run with identity primitive/supercell handling inside the phonon step.

Band-path detection uses the unrelaxed structure chosen for the Phonopy workflow, not the relaxed one. The path is generated with ASE's band-path mechanics in [phonons.py](phonons.py).

## Outputs

Results are written under:

```text
results/<model_name>/<structure_name>/
```

More precisely, the output root is:

```text
<config.yml.executive.results_root>/<model_name>/<structure_name>/
```

with raw files in `raw/` and plots in `plot/`.

Files actually written by the current code:

```text
results/<model>/<structure>/
├── raw/
│   ├── <base>_relax.traj      # only if relaxation was needed
│   ├── <base>_relaxed.poscar   # only if relaxation was needed
│   ├── <base>_phonons.yaml
│   ├── <base>_phonon_band.yaml   # only if want_band_structure: true
│   └── Plumipy_Files/
│       ├── <base>_CONTCAR_GS
│       ├── <base>_OUTCAR_GS
│       └── <base>_band.yaml     # Gamma-only Plumipy export
└── plot/
    ├── <base>_phonon_dos.png        # only if executive.plots: true
    ├── <base>_phonon_band_plot.png    # only if plots + band structure
    └── <base>_phonon_dispersion_dos.png  # only if plots + band structure
```

Notes on the raw artifacts:

- `<base>_phonons.yaml` is the saved Phonopy object and includes force constants because the save call enables `force_constants=True`.
- `<base>_phonon_band.yaml` is the main Phonopy band-structure output used elsewhere in the repo.
- The Plumipy export is separate from the main band output and always writes a single-Gamma `band.yaml`.

Some output name templates exist in `config.yml` but are not currently written by this pipeline, including a separate force-constants file and a raw DOS `.npz`.

## Main Modules

- [main.py](main.py): CLI entrypoint and end-to-end pipeline orchestration.
- [phonons.py](phonons.py): Phonopy construction, finite-displacement force collection, AFLOW-style band path generation, and DOS/band calculations.
- [plot.py](plot.py): Matplotlib wrappers for DOS, band, and combined band+DOS figures.
- [relax.py](relax.py): ASE-based structural relaxation helper.
- [get_calc.py](get_calc.py): Registry that maps model names to backend-specific ASE calculators.
- [config_classes.py](config_classes.py): Parsing of `config.yml` into model, structure, and output-plan objects.

## Supported Calculator Families

The calculator loader currently contains builders for several MLIP families, including:

- MACE
- MatterSim
- ORB
- PET / Metatomic
- MatGL / CHGNet / M3GNet / TensorNetDGL variants

Support is not automatic; the model name must be explicitly registered in [get_calc.py](get_calc.py). (any models listed in config.yml.models are automatically supported)

## Current issues

- [main.py](main.py) currently hard-requires CUDA and sets Torch's default device to `cuda`.
