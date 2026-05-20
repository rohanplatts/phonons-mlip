# ORB Fine-Tuning Scripts

This folder contains a self-contained ORB fine-tuning path for the curated
`REF_energy` / `REF_forces` extxyz splits in:

```bash
assets/training_data/curated_data/neutral_model
```

ORB's fine-tuning loader expects ASE sqlite databases with calculator-style
`energy` and `forces` fields. The scripts here convert your extxyz files into
that format without changing the original data.

## Setup

```bash
cd /home/rnpla/projects/mlip_phonons/src/fine_tuning/bash_fine_tuning_scripts/orb
./install_orb_env.sh
```

The default environment name is `orb_env`. Override it with:

```bash
ORB_CONDA_ENV=my_orb_env ./install_orb_env.sh
```

## Prepare Databases

```bash
PREFIX=all_neb_data_300_samples ./prepare_orb_dbs.sh
```

This writes:

```bash
assets/training_data/curated_data/orb_ase_db/<prefix>_train.db
assets/training_data/curated_data/orb_ase_db/<prefix>_val.db
assets/training_data/curated_data/orb_ase_db/<prefix>_test.db
```

## Full Fine-Tuning

```bash
PREFIX=all_neb_data_300_samples ./replay_fine_tuning_laptop.sh
```

By default this uses `orb-v3-conservative-inf-omat`, which is not an integrated
D3 model. That matches your current target convention where the D3 correction
has already been subtracted.

## Replay Fine-Tuning

ORB does not currently expose a MACE-style `--pt_train_file mp` shortcut. Replay
is therefore implemented by mixing your target DB with a replay DB that you
provide.

```bash
PREFIX=all_neb_data_300_samples \
REPLAY_EXTXYZ=/path/to/replay.extxyz \
REPLAY_RATIO=1.0 \
./replay_fine_tuning_laptop.sh
```

or, if the replay data is already converted:

```bash
PREFIX=all_neb_data_300_samples \
REPLAY_DB=/path/to/replay.db \
REPLAY_RATIO=1.0 \
./replay_fine_tuning_laptop.sh
```

`REPLAY_RATIO=1.0` means an expected 50/50 mix of target and replay samples.

## LoRA Fine-Tuning

```bash
PREFIX=all_neb_data_300_samples ./lora_fine_tuning_laptop.sh
```

Useful overrides:

```bash
LORA_RANK=8 LORA_ALPHA=16 LR=3e-4 ./lora_fine_tuning_laptop.sh
```

LoRA can also be combined with replay by setting `REPLAY_DB` or `REPLAY_EXTXYZ`.

## Cluster Jobs

```bash
sbatch replay_fine_tuning.sh
sbatch replay_lora_fine_tuning.sh
```

These assume the same module/conda pattern as the MACE scripts and activate
`ORB_CONDA_ENV` or `orb_env`.

## Quick Smoke Test

```bash
./quick_test_laptop.sh
```

This runs one epoch, two training steps, and one validation batch on
`all_neb_data_50_samples`.

## Main Environment Variables

`PREFIX`: dataset prefix, for example `all_neb_data_300_samples`.

`BASE_MODEL`: ORB model name, default `orb-v3-conservative-inf-omat`.

`MAX_EPOCHS`, `NUM_STEPS`, `BATCH_SIZE`, `LR`: training controls.

`REPLAY_DB` or `REPLAY_EXTXYZ`: optional replay data source.

`STRESS_LOSS_WEIGHT`: defaults to `0.0` because your current curated files do
not contain stress labels.
