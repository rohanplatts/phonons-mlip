# Supported Models

The supported model list lives in [SUPPORTED_MODELS.yml](../SUPPORTED_MODELS.yml).

Use this file as the shared registry. Do not copy the full model list into per-run `config.yml` files.

If a model is supported here, place the actual checkpoint(s) in `assets/models/<family>/` so `get_calc.py` can find them.

## Sources

The commands below are the direct sources I used to fetch the checkpoints. If an upstream URL changes, regenerate the local copy from the official source instead of editing the runtime registry.

### MACE

Official source:
- https://github.com/ACEsuit/mace-foundations/releases
- https://github.com/ACEsuit/mace/releases

Fetch example:

```bash
python - <<'PY'
from mace.calculators import mace_mp

mace_mp(model="medium-mpa-0")
mace_mp(model="medium-omat-0")
PY
```

Store the resulting `.model` files in `assets/models/mace/`.

### MatterSim

Official source:
- https://github.com/microsoft/mattersim
- https://github.com/microsoft/mattersim/tree/main/pretrained_models

Fetch example:

```bash
wget -O assets/models/mattersim/mattersim-v1.0.0-1M.pth \
  https://raw.githubusercontent.com/microsoft/mattersim/main/pretrained_models/MatterSim-v1.0.0-1M.pth
wget -O assets/models/mattersim/mattersim-v1.0.0-5M.pth \
  https://raw.githubusercontent.com/microsoft/mattersim/main/pretrained_models/MatterSim-v1.0.0-5M.pth
```

Store the resulting `.pth` files in `assets/models/mattersim/`.

### ORB

Official source:
- https://github.com/orbital-materials/orb-models
- https://huggingface.co/Orbital-Materials/OrbMol

Fetch example:

```bash
python - <<'PY'
from orb_models.forcefield import pretrained

pretrained.orb_v3_conservative_inf_omat(device="cpu")
pretrained.orb_v3_direct_inf_omat(device="cpu")
pretrained.orb_d3_sm_v2(device="cpu")
PY
```

Copy the downloaded `.ckpt` files into `assets/models/orb/`.

### PET-MAD / UPET

Official source:
- https://github.com/lab-cosmo/upet
- https://huggingface.co/lab-cosmo/upet/tree/main/models

Fetch example:

```bash
huggingface-cli download lab-cosmo/upet models/pet-mad-s-v1.1.0.ckpt \
  --local-dir assets/models/petmad/upet
huggingface-cli download lab-cosmo/upet models/pet-omad-s-v1.0.0.ckpt \
  --local-dir assets/models/petmad/upet
```

The repo uses the exported `.pt` form at runtime, so keep the checkpoint in `assets/models/petmad/upet/` and export or convert it to the `.pt` file name expected by `get_calc.py`.

### MatGL / CHGNet / M3GNet / TensorNet / QET

Official source:
- https://github.com/materialyzeai/matgl
- https://huggingface.co/materialyze

Fetch example:

```bash
python - <<'PY'
import matgl

matgl.load_model("materialyze/CHGNet-PES-MatPES-PBE-2025.2.10")
matgl.load_model("materialyze/CHGNet-PES-MatPES-r2SCAN-2025.2.10")
matgl.load_model("materialyze/M3GNet-PES-MatPES-PBE-2025.2")
matgl.load_model("materialyze/M3GNet-PES-MatPES-r2SCAN-2025.2")
PY
```

The same `matgl.load_model("<repo-id>")` pattern applies to the other MatGL-backed entries in `SUPPORTED_MODELS.yml`.

