# PET-MAD Fine-Tuning Scripts

This folder contains the PET-MAD family-specific fine-tuning path. It keeps
the repo layout family-first and avoids adding another conversion layer:

- `neb_data_set_synth` remains the source of truth for NEB curation.
- PET-MAD consumes the curated `*_train.extxyz`, `*_val.extxyz`, and
  `*_test.extxyz` files directly.
- Fine-tuning is driven by a native `metatrain` PET options file stored next
  to the family scripts.

## Entry Points

- `mtt train`: the actual PET-MAD trainer
- `petmad_options.yml`: default full fine-tuning config
- `train_petmad_laptop.sh`: runs `mtt train` with the family-local config
- `train_petmad_bunya.sh`: cluster submission wrapper for the same path
- `install_petmad_env.sh`: creates or updates the conda environment from
  `env/mace_env2.yml`

Supported PET-MAD fine-tuning here is full fine-tuning of the PET checkpoint
on the curated extxyz splits. There is no replay or LoRA path in this repo for
PET-MAD yet.

## Setup

Create or activate the PET-MAD training environment:

```bash
./install_petmad_env.sh
```

The environment definition comes from the shared `env/mace_env2.yml` file and
installs `metatrain[pet]`, `upet`, and the Python dependencies needed by the
launcher. The installer creates or updates the hardcoded `mace_env2`
environment.

## Training

The main training launcher is:

```bash
./train_petmad_laptop.sh
```

The launcher reads the family-local config directly:

```bash
src/fine_tuning/fine_tuning_scripts/petmad/petmad_options.yml
```

The launcher keeps the metatrain checkpoint/output directories local to the
family script folder. The PET checkpoint path, data paths, and demo-size
training knobs are encoded in `petmad_options.yml`, so that file is the place
to change the run shape.

On first run, the launcher downloads the PET-MAD finetuning checkpoint
(`pet-mad-v1.1.0.ckpt`) into the launcher directory and reuses it later.

The config expects a `selected_data/` folder next to this script directory.
The demo launcher creates a symlink to the step-1 curated data before
training starts.

## Cluster Run

```bash
sbatch ./train_petmad_bunya.sh
```

The bunya script uses the same config shape as the laptop launcher and keeps
the family-specific PET-MAD setup self-contained.

## Notes

- The bundled `assets/models/petmad/upet/*.pt` files are the exported PET
  foundation weights kept for reference; the finetuning launcher uses the
  downloadable `pet-mad-v1.1.0.ckpt` checkpoint instead.
- The fine-tuned model is exported as a `.pt` file and should be loaded later
  with the appropriate PET variant, for example `variants={"energy": "neb_ft"}`.
- To run a smoke-sized job, edit `petmad_options.yml` directly or change the
  `CONFIG_FILE` assignment in the launcher if you want to point at another copy.
