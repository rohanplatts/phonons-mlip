# Example 1: explicit MLIP-Workflows input

This is the ordinary research interface: `config.yml` explicitly selects the
model, endpoints, output location, and NEB settings. The included `neb.dat`
is a DFT energy reference and fixes the total image count at seven.

Run the MLIP NEB calculation from the repository root:

```bash
mlip-neb --inputs demo/neb/example1/input
```

Outputs are written to `demo/neb/example1/outputs/raw/`, including
`neb_raw.npz`, optimiser trajectories, and VASP-ready `vasp_mlip_d3/` and
`vasp_ci/` paths.

After the calculation, compare its energy profile with the included DFT
reference:

```bash
mlip-neb --inputs demo/neb/example1/input --compare
```

The comparison writes plots and a compact report below
`demo/neb/example1/outputs/raw/plot/`.
