# NEB Workflow

Run:

```bash
mlip-neb mace-mpa-0-medium --inputs inputs/neb/case --outputs results/neb/case
```

The input directory contains physical files:

- `POSCAR_i`
- `POSCAR_f`
- optional `neb.dat`
- optional `INCAR`, `KPOINTS`, and `POTCAR` copied into exported VASP folders

Config/flags contain intent and parameters:

- model name
- `device` and `dtype`
- `n_images`
- `include_vdw`
- endpoint relaxation/remapping
- FIRE `fmax`, `steps`, `maxstep`, and spring settings
- overwrite behavior

Config-only execution reads the local `config.yml` in the input folder:

```bash
mlip-neb
```

Useful examples:

```bash
mlip-neb
mlip-neb mace-mpa-0-medium --inputs assets/structures/NEB --outputs results/neb
mlip-neb mace-mpa-0-medium --inputs assets/structures/NEB --outputs results/neb --n-images 9 --no-include-vdw
mlip-neb mace-mpa-0-medium --inputs assets/structures/NEB --outputs results/neb --compare
```

Outputs are written under `<outputs>/<model_name>/raw`, including optimizer logs, trajectories, `neb_raw.npz`, `summary.txt`, and VASP-ready image folders.
