# mlip-phonons

MLIP workflows for phonons, NEB/MEP, ShakeNBreak, and fine-tuning demos.

## Workflows

- `mlip-phonons --inputs <config-dir>` for phonons, DOS, and band structures
- `mlip-neb --inputs <config-dir>` for NEB / MEP runs
- `mlip-snb --inputs <config-dir>` for ShakeNBreak defect workflows
- `mlip-coup --inputs <config-dir>` for phonon-coupling analysis

## Fine Tuning

Three demo families are included:

- [MACE](demo/fine_tuning/mace/README.md)
- [ORB](demo/fine_tuning/orb/README.md)
- [PET-MAD](demo/fine_tuning/petmad/README.md)

Each family follows the same structure:

`0_raw_inputs -> 1_curated_data -> 2_train -> 3_results -> 4_benchmark`

The benchmark step compares the baseline model and the fine-tuned model on the same raw NEB input.

## Benchmark Rule

If `defaults.model_name` is a list, the ordinary workflow command fans out once per model and runs each case in the model-specific environment from `SUPPORTED_MODELS.yml`.

For the fine-tuning demos, step 4 uses:

- `mlip-neb --inputs demo/fine_tuning/<family>/4_benchmark --report-benchmark`

## Setup

- Put model files under `assets/models/<family>/`
- Register model environments in `SUPPORTED_MODELS.yml`
- See [demo/fine_tuning/README.md](demo/fine_tuning/README.md) for the demo queue
- See [src/NEB/NEB_quickstart.md](src/NEB/NEB_quickstart.md) for NEB details
