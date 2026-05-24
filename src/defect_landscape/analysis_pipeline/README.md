# Defect Landscape Analysis Pipeline

This pipeline answers Carla's question directly:

> Do MLIP-relaxed ShakeNBreak candidates identify the same ground-state structure cluster as DFT?

The workflow uses existing DFT-relaxed SnB variants as the answer key. It does not create VASP folders or submit DFT jobs.

## Primary 20 Run

```bash
cd /home/rnpla/projects/mlip_phonons/src/defect_landscape/analysis_pipeline
./run_primary20.sh
```

Outputs go to:

```text
/home/rnpla/projects/mlip_phonons/src/defect_landscape/runs/primary_cs_pbi2br_q0_snb/
```

Use a different run folder name with:

```bash
./run_primary20.sh my_analysis_name
```

The script refuses to overwrite an existing analysis folder unless you set:

```bash
RESET=1 ./run_primary20.sh primary_cs_pbi2br_q0_snb
```

To stage inputs only before a Bunya transfer:

```bash
PREPARE_ONLY=1 ./run_primary20.sh primary_cs_pbi2br_q0_snb
```

## q+1 Mixed-Halide SnB Run

The q+1 script stages the same 20 CsPbI2Br gamma vacancy/site/order cases for the positive charge state and compares three MACE models:

```text
base_mace
finetuned_mace
finetuned_mace_positive
```

The positive model resolves to:

```text
/home/rnpla/projects/mlip_phonons/assets/models/mace/mace-mpa-0-medium-ft-cspbi3-positive.model
```

Stage the q+1 inputs only:

```bash
PREPARE_ONLY=1 ./run_qplus_snb.sh primary_cs_pbi2br_q+1_snb
```

After staged inputs are transferred back or if they already exist locally, run all relaxations/comparisons without restaging:

```bash
SKIP_PREPARE=1 ./run_qplus_snb.sh primary_cs_pbi2br_q+1_snb
```

The currently staged q+1 data contain 3 SnB variants per case (`Dimer`, `Rattled`, `Unperturbed`), so the full run performs 20 cases x 3 SnB variants x 3 models = 180 MLIP relaxations.

## Manual Single-Case Usage

Prepare one case:

```bash
python 00_prepare_case.py \
  --analysis-name primary_cs_pbi2br_q0_snb \
  --case-label VBr_q0_end_Br4c_test1 \
  --input-poscar /path/to/input/POSCAR \
  --dft-references-dir /path/to/dft_references
```

Relax with one model:

```bash
python 01_relax_mlip.py --analysis-name primary_cs_pbi2br_q0_snb --model base_mace
python 01_relax_mlip.py --analysis-name primary_cs_pbi2br_q0_snb --model finetuned_mace
```

Check progress:

```bash
python 01_relax_mlip.py --analysis-name primary_cs_pbi2br_q0_snb --model base_mace --status
```

Compare MLIP clusters to DFT clusters:

```bash
python 02_compare_to_existing_dft.py --analysis-name primary_cs_pbi2br_q0_snb
python 03_write_carla_report.py --analysis-name primary_cs_pbi2br_q0_snb
python 04_plot_energy_geometry_summary.py --analysis-name primary_cs_pbi2br_q0_snb
```

## Important Outputs

```text
runs/<analysis_name>/analysis/candidate_manifest.csv
runs/<analysis_name>/analysis/dft_clusters.csv
runs/<analysis_name>/analysis/mlip_clusters.csv
runs/<analysis_name>/analysis/mlip_dft_cluster_distances.csv
runs/<analysis_name>/analysis/variant_energy_rankings.csv
runs/<analysis_name>/analysis/case_model_summary.csv
runs/<analysis_name>/analysis/case_consensus_summary.csv
runs/<analysis_name>/analysis/ground_cluster_energy_geometry_summary.csv
runs/<analysis_name>/analysis/tested_structure_summary.csv
runs/<analysis_name>/analysis/unresolved_cases.csv
runs/<analysis_name>/analysis/carla_ground_cluster_answer.md
runs/<analysis_name>/analysis/combined_mlip_vs_dft.png
runs/<analysis_name>/analysis/combined_mlip_vs_dft_low_energy_zoom.png
runs/<analysis_name>/analysis/mlip_dft_geometry_summary.png
runs/<analysis_name>/analysis/per_case_ground_cluster_geometry.png
```

`mlip_clusters.csv` and `mlip_dft_cluster_distances.csv` contain the geometry checks: StructureMatcher fit, RMS distance, max distance, and per-species distance JSON.

## CsPbI3 NEB Endpoint Preservation

This is a separate test from SnB ground-cluster selection. It starts MLIP relaxations from existing DFT NEB endpoints and checks whether the relaxed MLIP structure remains closest to the same endpoint.

Stage the endpoint inputs only:

```bash
PREPARE_ONLY=1 ./run_cspbi3_neb_endpoints.sh cspbi3_neb_endpoint_preservation
```

Run the full local workflow:

```bash
./run_cspbi3_neb_endpoints.sh cspbi3_neb_endpoint_preservation
```

Continue from already staged inputs:

```bash
SKIP_PREPARE=1 ./run_cspbi3_neb_endpoints.sh cspbi3_neb_endpoint_preservation
```

The model schedule is charge-specific:

```text
q0:  base_mace, finetuned_mace
q+1: base_mace, finetuned_mace, finetuned_mace_positive
q-1: base_mace, finetuned_mace, finetuned_mace_negative
```

Outputs:

```text
runs/<analysis_name>/analysis/endpoint_case_metadata.csv
runs/<analysis_name>/analysis/endpoint_geometry_comparisons.csv
runs/<analysis_name>/analysis/endpoint_model_summary.csv
runs/<analysis_name>/analysis/endpoint_case_summary.csv
runs/<analysis_name>/analysis/endpoint_cases_needing_attention.csv
runs/<analysis_name>/analysis/neb_endpoint_preservation_report.md
```

The strict preservation condition is:

```text
StructureMatcher(MLIP relaxed endpoint, expected DFT endpoint) is True
and the nearest DFT endpoint in that NEB case is the expected endpoint label.
```

## Bunya Policy

This pipeline should not clog Bunya with DFT. If you run MLIP relaxations on Bunya, transfer only:

```text
runs/<analysis_name>/snb_variant_inputs/
runs/<analysis_name>/analysis/candidate_manifest.csv
analysis_pipeline/
```

Transfer back:

```text
runs/<analysis_name>/mlip_relaxed/
```

Then run the comparison scripts locally against the staged DFT references. Do not transfer or duplicate `POTCAR`.
