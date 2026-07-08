# PET-MAD Fine-Tuning Scripts

This folder contains the PET-MAD family-specific fine-tuning path. It keeps
the repo layout family-first and avoids adding another conversion layer:

- `neb_data_set_synth` remains the source of truth for NEB curation.
- PET-MAD consumes the curated `*_train.extxyz`, `*_val.extxyz`, and
  `*_test.extxyz` files directly.
- Fine-tuning is driven by native `metatrain` PET options files stored next
  to the family scripts.

## Entry Points

- `mtt train`: the actual PET-MAD trainer
- `petmad_options.yml`: default PET LoRA fine-tuning config
- `petmad_full_options.yml`: alternate full fine-tuning config
- `train_petmad_laptop.sh`: runs `mtt train` with the LoRA config
- `train_petmad_bunya.sh`: cluster submission wrapper for the LoRA path
- `train_petmad_full_laptop.sh`: alternate full fine-tuning launcher
- `train_petmad_full_bunya.sh`: cluster submission wrapper for full fine-tuning
- `install_petmad_env.sh`: creates or updates the conda environment from
  `env/mace_env.yml`

The primary PET-MAD path is native metatrain PET LoRA. The repo does not
inject adapters itself; `mtt train` owns the PET model modification.

## Setup

Create or activate the PET-MAD training environment:

```bash
./install_petmad_env.sh
```

The environment definition comes from the repo-level `env/mace_env.yml` file.
It installs MACE, ORB, `metatrain[pet]`, `upet`, and the shared dependencies
used by the fine-tuning demos. The installer creates or updates the hardcoded
`mace_env` environment.

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
family script folder. The PET checkpoint path, data paths, LoRA rank/alpha,
target modules, and demo-size training knobs are encoded in
`petmad_options.yml`, so that file is the place to change the run shape.

On first run, the launcher downloads the PET-MAD finetuning checkpoint
(`pet-mad-v1.1.0.ckpt`) into the launcher directory and reuses it later.

The config expects a `selected_data/` folder next to this script directory.
The demo launcher creates a symlink to the step-1 curated data before
training starts.

## Alternate Full Fine-Tuning

Full fine-tuning is still available for comparison:

```bash
./train_petmad_full_laptop.sh
```

That launcher reads:

```bash
src/fine_tuning/fine_tuning_scripts/petmad/petmad_full_options.yml
```

It writes `results/petmad/model_full.pt` and uses a separate extensions
directory so it does not overwrite the default LoRA output.

## Cluster Run

```bash
sbatch ./train_petmad_bunya.sh
sbatch ./train_petmad_full_bunya.sh
```

The first job is the default LoRA path. The second job is the alternate full
fine-tuning path.

## Notes

- The bundled `assets/models/petmad/upet/*.pt` files are the exported PET
  foundation weights kept for reference; the finetuning launcher uses the
  downloadable `pet-mad-v1.1.0.ckpt` checkpoint instead.
- The fine-tuned model is exported as a `.pt` file and is loaded later with
  the `energy/neb_ft` variant.
- To run a smoke-sized job, edit `petmad_options.yml` directly or change the
  `CONFIG_FILE` assignment in the launcher if you want to point at another copy.
