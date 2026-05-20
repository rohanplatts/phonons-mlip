# MLIP-SnB Usage

`mlip-snb` accelerates a ShakeNBreak workflow with any MLIP supported by
`mlip_phonons.get_calc_object()`.

The intended loop is:

1. Import or generate ShakeNBreak distorted `POSCAR` candidates.
2. Relax every candidate with an MLIP.
3. Cluster the relaxed structures into distinct minima.
4. Select low-energy cluster representatives.
5. Export selected representatives as VASP-ready DFT refinement folders.
6. Parse completed DFT refinements.
7. Compare MLIP-selected clusters against DFT-relaxed clusters.

## Setup

From the repository root, install the package in the environment that contains
the MLIP dependencies:

```bash
conda activate mace_env
pip install -e .
```

Check the CLI is available:

```bash
mlip-snb --help
```

If the console script is not installed yet, the module form is equivalent:

```bash
PYTHONPATH=src python -m defect_landscape.snb.cli --help
```

## Fast Path: Existing ShakeNBreak POSCARs

If you already have a folder containing SnB distorted `POSCAR` files:

```bash
mlip-snb mace-mpa-0-medium \
  --snb-dir path/to/snb_outputs \
  --case-name v_I_q0 \
  --results-root resultsSNB \
  --models-root assets/models \
  --include-vdw \
  --prepare-dft \
  --vasp-inputs-dir path/to/vasp_template
```

This shortcut is the same as:

```bash
mlip-snb run mace-mpa-0-medium \
  --snb-dir path/to/snb_outputs \
  --case-name v_I_q0 \
  --prepare-dft
```

The `--snb-dir` can be nested. Every file named `POSCAR` under that folder is
treated as one candidate.

## Staged Workflow

Use staged commands when running on a cluster, resuming interrupted work, or
debugging one step at a time.

```bash
mlip-snb run mace-mpa-0-medium \
  --snb-dir path/to/snb_outputs \
  --case-name v_I_q0
```

This imports candidates, relaxes them, clusters the relaxed structures, selects
DFT representatives, and writes a report.

After a candidate manifest exists, for example from `generate` or a previous
`run`, the individual stages are:

```bash
mlip-snb relax mace-mpa-0-medium --case-name v_I_q0
mlip-snb cluster mace-mpa-0-medium --case-name v_I_q0
mlip-snb select mace-mpa-0-medium --case-name v_I_q0
mlip-snb prepare-dft --case-name v_I_q0 --vasp-inputs-dir path/to/vasp_template
```

For existing SnB folders, the practical import command is `run --snb-dir`.
Completed relaxation folders are skipped unless `--overwrite` is supplied, so
rerunning the same command is the normal resume path.

## Generate ShakeNBreak Inputs

If ShakeNBreak is installed and you want to generate candidates from bulk and
defect structures:

```bash
mlip-snb generate \
  --bulk bulk/POSCAR \
  --defect defect/POSCAR \
  --case-name v_I_q0 \
  --oxidation-states Cs=1 Pb=2 I=-1
```

Then run the MLIP stages:

```bash
mlip-snb run mace-mpa-0-medium --case-name v_I_q0
```

If ShakeNBreak is not installed, generate the SnB inputs elsewhere and use
`--snb-dir`.

The same generation can be fully config-driven. Put `bulk`, `defect`,
`oxidation_states`, `vasp_inputs_dir`, and `prepare_dft: true` under
`snb.defaults` in `config.yml`, then run:

```bash
mlip-snb
```

## Selection Policy

By default, `select` keeps one representative per MLIP cluster within `0.50 eV`
of the MLIP ground cluster, capped at `10` clusters per model.

Useful options:

```bash
mlip-snb select mace-mpa-0-medium \
  --case-name v_I_q0 \
  --energy-window 0.30 \
  --max-clusters 6
```

```bash
mlip-snb select mace-mpa-0-medium \
  --case-name v_I_q0 \
  --top-k-clusters 5
```

For multiple models, `--union-across-models` deduplicates selected structures
with `StructureMatcher`.

## DFT Refinement Folders

Prepare VASP folders from selected MLIP cluster representatives:

```bash
mlip-snb prepare-dft \
  --case-name v_I_q0 \
  --vasp-inputs-dir path/to/vasp_template
```

Template files copied or linked if present:

- `INCAR`
- `KPOINTS`
- `POTCAR`
- `submit.sh`

`POTCAR` is symlinked by default to avoid duplication. Use `--copy-potcar` only
when a real copy is required.

After this step, submit the folders under:

```text
resultsSNB/<case_name>/dft_validate/
```

## Check Completed DFT Jobs

After copying completed VASP outputs back:

```bash
mlip-snb check-dft --case-name v_I_q0
```

This parses `vasprun.xml`, `OUTCAR`, and `OSZICAR` where available and writes:

```text
resultsSNB/<case_name>/analysis/dft_status.csv
resultsSNB/<case_name>/analysis/dft_jobs_to_restart.csv
```

## Compare MLIP Against DFT

Once DFT refinements have completed:

```bash
mlip-snb compare-dft --case-name v_I_q0
mlip-snb report --case-name v_I_q0
```

The comparison reports:

- whether the MLIP-selected ground cluster is the DFT ground cluster
- whether the top 3 MLIP clusters contain the DFT ground cluster
- DFT energy penalty of using the MLIP-selected ground cluster
- relative-energy MAE across validated representatives
- direct MLIP-to-DFT geometry RMS/max distances
- `StructureMatcher` fit between MLIP and DFT structures

## Output Layout

```text
resultsSNB/<case_name>/
  snb_inputs/
  mlip_relaxed/<model>/<candidate_id>/
    CONTCAR
    trajectory.traj
    result.json
  selected_for_dft/
  dft_validate/
  analysis/
    candidate_manifest.csv
    relaxation_results.csv
    clusters_<model>.csv
    cluster_members_<model>.csv
    selected_for_dft_manifest.csv
    dft_validation_manifest.csv
    dft_status.csv
    dft_jobs_to_restart.csv
    mlip_vs_dft.csv
    geometry_comparisons.csv
    ranking_metrics.json
    report.md
```

## Common Flags

```text
--config config.yml
--results-root resultsSNB
--models-root assets/models
--device cuda
--dtype float32
--include-vdw / --no-include-vdw
--overwrite / --no-overwrite
--prepare-dft / --no-prepare-dft
```

Defaults live in the `snb:` section of `config.yml`.

## Cluster Matching Tolerances

MLIP and DFT clusters use:

```text
ltol = 0.2
stol = 0.3
angle_tol = 5
primitive_cell = False
scale = False
attempt_supercell = False
```

These are the default `pymatgen.analysis.structure_matcher.StructureMatcher`
settings used by this workflow. The geometry CSV also includes direct
species-resolved Hungarian minimum-image distances, so the report contains both
cluster identity and physical displacement errors.
