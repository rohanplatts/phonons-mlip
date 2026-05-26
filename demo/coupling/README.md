# Coupling Demos

Each example is a self-contained input directory for `mlip-coup`.

Run them with:

```bash
mlip-coup --inputs demo/coupling/example1
mlip-coup --inputs demo/coupling/example2
```

Each input directory should contain:

- `config.yml`
- `CONTCAR_GS`
- `CONTCAR_ES`
- one DFT `band.yaml`
- one or more MLIP `band.yaml` files, either listed in `config.yml` or
  discoverable under the input directory
