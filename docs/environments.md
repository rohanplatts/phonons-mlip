
Last update: 31/01/2026.

Models are not always mutually compatible. ASE is the standard for atomic work, so most model families expose an ASE calculator interface. That is why the repository keeps separate environments for different families.

The shared model list lives in [SUPPORTED_MODELS.yml](../SUPPORTED_MODELS.yml). The workflows use that registry to map each model to the right environment.

For the current set of included model families, the environments are as follows:

## MACE, orb-materials, PET-mad model families:
From repo root:
```
conda deactivate
conda env create -f env/mace_env.yml
conda activate mace_env
pip install -e . 
```

## MATTERSIM model family:
from repo root: 
```
conda deactivate
conda env create -f env/mattersim_env.yml
conda activate mattersim_env
pip install -e .
```

## MATGL model families:
> I.e. M3GNet, TENSORNET, CHGNet
from repo root:
```
conda deactivate
conda env create -f env/matgl_env.yml
conda activate matgl_env
pip install -e .
```

Thats probably all you need to read from here :). 

> NOTE FOR MATGL ENVIRONMENT INSTALLATION: at that time DGL (a matgl backend that governed model tensor operation) was not supported and thus had to be manually installed since the matgl models had not been shifted from dgl to pytorch (PyG) yet. This was january 2026, and may have since changed. **SO IF YOU ARE GETTING SOME KIND OF BACKEND ERROR FOR MATGL SPECIFICALLY** read their updates [here](https://matgl.ai/#major-update-v200-nov-12-2025). You should be able to check for which backend (PyG or DGL) your model uses and install that. If the installation is messy and not working, i found these tips helpful:
1. install the desired version of torch first 
2. install the required backend for matgl first 
3. install matgl
