# Fine-Tuning Demo

The demo tree is organized by family. Each workflow now uses the same
five-step layout:

- `0_raw_inputs/` keeps the raw NEB bundle and the `siv_rules.yml` file that
  drives curation
- `1_curated_data/` receives the curated output
- `2_train/` holds the family-specific training launcher
- `3_results/` collects the training artifacts
- `4_benchmark/` compares the baseline model against the fine-tuned model on
  the raw NEB problem with `mlip-neb`

Available workflows:

- [MACE](/home/rnpla/projects/mlip_phonons/demo/fine_tuning/mace/README.md): MACE LoRA
- [ORB](/home/rnpla/projects/mlip_phonons/demo/fine_tuning/orb/README.md): repo-local ORB LoRA/replay
- [PET-MAD](/home/rnpla/projects/mlip_phonons/demo/fine_tuning/petmad/README.md): native metatrain PET LoRA

Each workflow README describes the full path from step 0 through step 4 and
shows the exact `mlip-neb --report-benchmark` command that generates the final
comparison plots and README section.
