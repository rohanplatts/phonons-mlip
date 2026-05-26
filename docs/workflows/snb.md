# ShakeNBreak Workflow

Run:

```bash
mlip-snb mace-mpa-0-medium --inputs inputs/snb/case --outputs results/snb/case
```

The input directory contains physical files in one of two forms:

- existing ShakeNBreak candidate folders containing POSCAR files, or
- `bulk/POSCAR` and `defect/POSCAR` for generating candidates

Config/flags contain intent and parameters:

- model name
- `device` and `dtype`
- `include_vdw`
- relaxation `fmax` and `max_steps`
- clustering tolerances
- energy window and cluster limits
- whether to prepare VASP DFT folders
- oxidation states when generating SnB candidates

Config-only execution reads the local `config.yml` in the input folder:

```bash
mlip-snb
```

Outputs are written under `<outputs>/<case_name>/`, including staged candidates, relaxation summaries, cluster data, DFT selection manifests, optional VASP folders, and reports.
