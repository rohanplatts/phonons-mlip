## SnB Quickstart (bulk/defect POSCAR + MLIP + VASP export)

Suppose you want to use ShakeNBreak to find candidate ground-state defect
structures, but you do not want DFT to relax every distorted structure. The main
purpose of `mlip-snb` is to use an MLIP, for example `'mace-mpa-0-medium'`, to
quickly reduce the ShakeNBreak structure set to a small number of MLIP-relaxed
cluster representatives, and then export those representatives as VASP-ready
DFT validation folders.

1. Ensure you have `mace_env` set up.
2. Obtain the `'mace-mpa-0-medium'` model file. (I HAVE ALREADY INCLUDED THIS FILE FOR CONVENIENCE...)
3. Put the model file in `assets/models/<model_family>/<model_file>`. (I HAVE ALREADY DONE THIS...)
4. Prepare a case directory with `config.yml`, the bulk `POSCAR`, the defect `POSCAR`, and the VASP input folder.
5. Check if the model file is supported by reading `config.yml`. (IT IS), and if it isnt, add its calculator object to `src/common/get_calc.py`.
6. Run:

```bash
mlip-snb --inputs path/to/snb-case \
  --prepare-dft
```

This will generate ShakeNBreak distortions, relax them with the MLIP, cluster
the relaxed structures, select the low-energy cluster representatives, and write
VASP-ready folders here:

```text
resultsSNB/v_I_q0/dft_validate/
```

The case folder is inferred from the input directory name. If you want to force
a different label, add `--case-name my_defect_case`.

Submit those VASP folders. The final DFT-refined ground-state structure is only
known after these selected DFT relaxations have finished.

Useful overrides:
- `--results-root`: where outputs go (default `resultsSNB/`).
- `--models-root`: where model weights live (default `assets/models/`).
- `--case-name`: optional output label if you do not want to use the inferred folder name.
- `--vasp-inputs-dir`: folder containing `INCAR`, `KPOINTS`, `POTCAR`, and optionally `submit.sh`.
- `--energy-window`: how far above the MLIP ground cluster to keep candidates for DFT (default `0.50 eV`).
- `--max-clusters`: maximum number of MLIP clusters to export for DFT (default `10`).
- `--copy-potcar`: copy `POTCAR` instead of symlinking it.

After the selected VASP relaxations finish, copy the completed `CONTCAR`,
`OUTCAR`, `OSZICAR`, and/or `vasprun.xml` files back into the same
`dft_validate/` folders, then run:

```bash
mlip-snb --inputs path/to/snb-case check-dft
mlip-snb --inputs path/to/snb-case compare-dft
mlip-snb --inputs path/to/snb-case report
```

If the case directory contains more than one results folder, add `--case-name v_I_q0`
to those commands.

The structure you want is identified by `dft_ground_selection_id` in:

```text
path/to/snb-case/resultsSNB/v_I_q0/analysis/mlip_vs_dft.csv
```

and the corresponding final DFT-relaxed ground-state structure is:

```text
path/to/snb-case/resultsSNB/v_I_q0/dft_validate/<dft_ground_selection_id>/CONTCAR
```

Now if you want this to be even more efficient, you can prepare your `mlip-snb`
command in `config.yml` by editing SnB defaults. Say i had the path to the bulk
POSCAR, path to the defect POSCAR, say i wanted the results to be located in
some obscure folder, that i wanted van-der-waals term correction on, that i had
a folder containing the VASP inputs, and that i wanted only a small number of
MLIP-selected structures to be exported for DFT. then in `config.yml`, i would
change SnB to:

```text
snb:
  defaults:
    model_name: mace-mpa-0-medium # the model
    results_root: /some/obscure/folder/resultsSNB # where you want the results
    models_root: assets/models # where to look for the model

    bulk: /path/to/bulk/POSCAR
    defect: /path/to/v_I_q0/POSCAR
    snb_dir: null # use this instead of bulk/defect if SnB structures already exist
    oxidation_states:
      Cs: 1
      Pb: 2
      I: -1

    vasp_inputs_dir: /path/to/vasp_inputs
    prepare_dft: true
    copy_potcar: false
    include_vdw: true
    overwrite: false
    device: cuda
    dtype: float32
  settings:
    fmax: 0.03
    max_steps: 600
    energy_window_eV: 0.50
    max_clusters_per_model: 10
    matcher_ltol: 0.2
    matcher_stol: 0.3
    matcher_angle_tol: 5
```

Then, all you will have to do is type into command line:

```bash
mlip-snb --inputs path/to/snb-case
```

This works because the model name, bulk POSCAR, defect POSCAR, oxidation states,
VASP input folder, and `prepare_dft: true` are all defined in `config.yml`.

For more SnB details (including DFT checking, comparison, and report outputs),
see `src/defect_landscape/snb/README.md`.
