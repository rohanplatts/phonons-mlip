
Last update: 31/01/2026. 

Models are not always mutually compatible. However, ASE is the standard for atomic work, consequently most models have a described method (in their documentation) by which one can access the model as an ASE calculator object. Thus, models most often are supported for ASE. Regardless, the fact the model families are not necessarily mutually compatible motivates the requirement for separate environments dedicated to compatible model families. If you are planning however to do large model sweeps, this should not inconvenience you, since the supplied bash scripts do model sweeps based upon the `config.yml` file, which list the models and points to the correct environment for them. 

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
conda deactivate mattersim_env
pip install -e .
```

Thats probably all you need to read from here :). 

> NOTE FOR MATGL ENVIRONMENT INSTALLATION: at that time DGL (a matgl backend that governed model tensor operation) was not supported and thus had to be manually installed since the matgl models had not been shifted from dgl to pytorch (PyG) yet. This was january 2026, and may have since changed. **SO IF YOU ARE GETTING SOME KIND OF BACKEND ERROR FOR MATGL SPECIFICALLY** read their updates [here](https://matgl.ai/#major-update-v200-nov-12-2025). You should be able to check for which backend (PyG or DGL) your model uses and install that. If the installation is messy and not working, i found these tips helpful:

1. install the desired version of torch first 
2. install the required backend for matgl first 
3. install matgl

NOTES FOR DEV: 
of 31/01/2026: Checked mace_environment.yml, matgl_env and it loaded all calc objects

For mace:
```    
conda create --name mace_env python=3.10 pip 
pip install mace-torch 
pip install phonopy
```

for mattersim:

```
conda create --name mattersim_env python=3.10 pip -y 
pip install mattersim 
```

for orb-models: 

```
conda create --name orb_env python=3.10 pip -y
pip install orb-models
pip install "pynanoflann@git+https://github.com/dwastberg/pynanoflann#egg=af434039ae14bedcbb838a7808924d6689274168"
```

for pet-mad:

```
conda create --name petmad_env python=3.10 pip -y
pip install upet 
```



