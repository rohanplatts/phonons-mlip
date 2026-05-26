# Workflow Guides

These pages describe the four user-facing workflow commands:

- [Phonons](phonons.md)
- [NEB / MEP](neb.md)
- [ShakeNBreak](snb.md)
- [Phonon coupling](coupling.md)

Every workflow follows the same basic rule:

- put a `config.yml` in an input directory
- run the workflow with `--inputs <config-dir>`
- keep model selection in `config.yml` under `defaults.model_name`

