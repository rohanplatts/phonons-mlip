# Phonons Demos

Two phonon workflow input directories are included:

- `example1` for hBN
- `example2` for diamond

Each example is self-contained. Run it with:

```bash
mlip-phonons --inputs demo/phonons/example1/input
mlip-phonons --inputs demo/phonons/example2/input
```

Each input directory contains a `config.yml` and the structure files the
phonon workflow reads.

The hBN example also enables Plumipy-ready exports in its `executive` config.
