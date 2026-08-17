# NEB demos

These representative CsPbI3 defect-migration inputs demonstrate the three
supported ways to work with NEB results.

- `example1/` is the standard explicit MLIP-Workflows interface: endpoint
  structures and a `config.yml`.
- `example2/` is the VASP-native interface: a normal VASP NEB-style input
  directory with only `00/POSCAR`, the final endpoint, and `IMAGES` required
  to construct the MLIP path with IDPP.
- `example3_benchmark/` is a three-model benchmark setup. It fans out the
  same NEB calculation across registered models, then creates a combined
  energy-profile and path-fidelity report against the included DFT reference
  path.

Each example has its own README with the command to run. The config-driven
and benchmark examples write into their `outputs/` directories; the VASP
frontend deliberately writes VASP-ready outputs inside its input directory.
Model checkpoints and their Conda environments must be available as described
in `SUPPORTED_MODELS.yml`.
