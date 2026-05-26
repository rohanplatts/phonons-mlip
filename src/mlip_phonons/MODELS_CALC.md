
Generally, if you wanted to add a model, then you will first have to resolve a way to obtain the ASE calculator object from that model file, and implement that workflow into `src/common/get_calc.py`. You will find though, that if this model is of the model families accomodataed by this script, it is very likely all you will have to do is:

* copy the model file into the correct family subfolder in `assets/models`, 
* then add a new dictionary item for that specific model in `src/common/get_calc.model_build`
* the key for this dictionary item is simply the name of the model, as for the value, you will be able to identify that easily when you are appending this.

Next, it is reccomended to edit config.yml and add the supported model to your list. Note that there is a structure key to be added under the model. This structure key is important when it comes to model sweeps of Phonon workflows, as in those cases, the config.yml file is read and each model is fed its own structure. If it does not appear to you that this would be useful (perhaps you are doing only NEB workflows) you can set this key to None.

Edit `config.yml`:
- `models`: model names (must match the keys supported by `src/common/get_calc.py`) and which structure/material they run on.
- `structures`: where structures live (`assets/structures/...`) plus phonon settings (supercell, displacement `delta`, DOS mesh, etc.).

See `config.yml` for the expected structure of inputs.

Model weights/checkpoints are expected under `assets/models/` 
