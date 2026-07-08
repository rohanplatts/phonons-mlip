# ORB Fine-Tuning Scripts

This folder contains the ORB training launchers that consume the curated ASE
sqlite DB splits written by `mlip-ft orb --curate-neb --inputs ...`.

ORB expects calculator-style `energy` and `forces` fields in the DBs. The
repo's `mlip-ft` path handles the extxyz-to-DB conversion automatically, so
the training launchers here only deal with ORB training.

## Entry Points

- `train_orb.py`: actual ORB trainer
- `replay_fine_tuning_laptop.sh`: target-only or replay fine-tuning
- `lora_fine_tuning_laptop.sh`: LoRA fine-tuning on the same DB inputs
- `replay_fine_tuning.sh`: bunya replay submission wrapper
- `replay_lora_fine_tuning.sh`: bunya LoRA submission wrapper
- `install_orb_env.sh`: creates or updates the conda environment from
  `orb_ft_env.yml`

## Setup

```bash
./install_orb_env.sh
```

The installer creates or updates the hardcoded `mace_env2` environment. If you
want a different environment name, edit `install_orb_env.sh` directly.

## Replay Or Target-Only Fine-Tuning

```bash
./replay_fine_tuning_laptop.sh
```

The launcher uses explicit values at the top of the file. By default it runs
target-only fine-tuning with `orb-v3-conservative-inf-omat`, which is not an
integrated D3 model. That matches the current target convention where the D3
correction has already been subtracted. If you want replay, edit `REPLAY_DB`
in the launcher before running it.

## LoRA Fine-Tuning

```bash
./lora_fine_tuning_laptop.sh
```

LoRA can also be combined with replay by setting `REPLAY_DB`.
Edit the `REPLAY_DB` and LoRA regex variables in the launcher file if you want
to enable replay or change which modules receive adapters.

## Cluster Jobs

```bash
sbatch replay_fine_tuning.sh
sbatch replay_lora_fine_tuning.sh
```

These assume the same module/conda pattern as the MACE scripts and activate
`mace_env2`.
