# Phonon Workflow

Run:

```bash
mlip-phonons mace-mpa-0-medium --inputs inputs/phonons/hbn --outputs results/phonons
```

The input directory contains physical files:

- POSCAR/CONTCAR structure files
- optional primitive-cell POSCAR/CONTCAR files

Config/flags contain intent and parameters:

- model name
- structure key
- `device` and `dtype`
- whether the input is already relaxed
- supercell matrix
- displacement amplitude
- DOS mesh
- whether to compute band structure and plots

Config-only execution reads the local `config.yml` in the input folder:

```bash
mlip-phonons
```

Outputs are written under `<outputs>/<model_name>/<structure_key>/`, with `raw/`, `plot/`, and Plumipy-ready files when generated.
