# Example 2: VASP-native NEB input

This input directory is intentionally VASP-shaped. `INCAR` contains
`IMAGES = 5` and the optional MLIP directives; `00/POSCAR` and `06/POSCAR`
are the two fixed endpoints. Intermediate folders are not required because
MLIP-Workflows builds the five intermediate coordinates with IDPP.

Run it directly:

```bash
mlip-neb --vasp demo/neb/example2/input
```

The frontend reads `MLIP_MODEL` from `INCAR`, selects the registered model
environment if needed, and writes the generated NEB outputs under
`demo/neb/example2/input/resultsNEB/raw/`. The VASP-ready climbing-image path
is `vasp_ci/`. If a known submission script such as `vasp_bunya.sh` is added
to this input directory, it is copied into the exported path and submitted
only after a successful VASP-frontend MLIP run.
