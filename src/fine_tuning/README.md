# Fine Tuning

`mlip-ft` curates NEB data for three family-specific fine-tuning workflows:
MACE, ORB, and PET-MAD.

Use the same pattern for all families:

```bash
mlip-ft <family> --curate-neb --inputs demo/fine_tuning/<family>/0_raw_inputs/siv_rules.yml
./demo/fine_tuning/<family>/2_train/run_fine_tuning.sh
mlip-neb --inputs demo/fine_tuning/<family>/4_benchmark --report-benchmark
```

Family names:

- `mace`
- `orb`
- `petmad`

Each family demo has the same numbered step layout:

- `0_raw_inputs/` contains the raw NEB bundle and curation rules
- `1_curated_data/` receives the curated training split
- `2_train/` contains the family-specific training launcher and config
- `3_results/` stores the trained model and export artifacts
- `4_benchmark/` compares the baseline model against the fine-tuned model on
  the raw NEB endpoints

Important differences:

- MACE and PET-MAD keep curated `extxyz` outputs for training.
- ORB uses the same curation step, then converts the curated `extxyz` files
  into ASE sqlite DBs before training.
- The benchmark step always uses `mlip-neb` with a list-valued
  `defaults.model_name`.

For the family-specific command details and the exact files each launcher reads,
see:

- [MACE demo](../../demo/fine_tuning/mace/README.md)
- [ORB demo](../../demo/fine_tuning/orb/README.md)
- [PET-MAD demo](../../demo/fine_tuning/petmad/README.md)
