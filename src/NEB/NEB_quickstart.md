## NEB Quickstart (poscar_i/poscar_f + MLIP)

Suppose you want the MEP from NEB between two endpoints with the MLIP 'mace-mpa-0-medium'.

1. Ensure you have mace_env set up (see above)
2. Obtain the 'mace-mpa-0-medium' model file (I HAVE ALREADY INCLUDED THIS FILE FOR CONVENIENCE...)
3. Put the model file in `assets/models/<model_family>/<model_file>` (I HAVE ALREADY DONE THIS...)
4. Prepare endpoints: `POSCAR_i` and `POSCAR_f` (or pass paths via flags). (AS AN EXAMPLE, I HAVE PUT SAMPLE POSCAR_i and POSCAR_f in `src/NEB` )
5. Check if the model file is supported by reading config.yml (IT IS), and if it isnt, add its calculator object to src/common/get_calc.py
6. Run the case directory:

```bash
mlip-neb --inputs path/to/neb-case
```

Useful overrides:
- `--results-root`: where outputs go (default `resultsNEB/`).
- `--models-root`: where model weights live (default `assets/models/`).
- `--dft-neb-dat`: optional DFT reference `neb.dat` for comparisons.
- `--n-images`: number of images (otherwise inferred or defaulted).

Now if you want this to be even more efficient, you can prepare your `mlip-neb` command in config.yml by editting NEB defaults. Say i had the path to POSCAR_i, and path to POSCAR_f, say i wanted the results to be located in some obscure folder, that i wanted van-der-waals term correction on, that i wanted the endpoints to be relaxed by the MLIP, and that i want within-species remapping of the initial and final poscars, and that i wanted the final supplied MLIP MEP to be vasp loadable. then in config.yml, i would change NEB to: 

```text
neb:
  defaults:
    model_name: mace-mpa-0-medium # the model
    results_root: /some/obscure/folder/resultsNEB # where you want the results
    models_root: assets/models # where to look for the model 
    structures_dir: None # this is only really for a backup place to look for poscar_i or poscar_f, or neb.dat when trying to compare with DFT 

    poscar_i: /path/to/POSCAR_i
    poscar_f: /path/to/POSCAR_f

    dft_neb_dat: null # the neb.dat path when wanting to compare model performance with DFT predictions.
    vasp_inputs_dir: /path/to/vasp_inputs 
    relax_endpoints: true 
    remap_f_i: true
    include_vdw: true
    overwrite: false
    device: cuda
    dtype: float32
  settings:
    n_images_fallback: 9
    maxstep_mlip_guess: 0.05 
    fmax_mlip_guess: 0.03
    steps_mlip_guess: 3000
    k_spring_mlip: 0.6
    k_spring: 0.6
    maxstep_mlip_d3: 0.03
    fmax_mlip_d3: 0.03
    steps_mlip_d3: 1400
    maxstep_ci: 0.03
    fmax_ci: 0.03
    steps_ci: 1000
```

Then, all you will have to do is type into command line:
```
mlip-neb --inputs path/to/neb-case
```
And it shall run according to your settings.

For more NEB details (including comparison and VASP export), see `src/NEB/README.md`.
