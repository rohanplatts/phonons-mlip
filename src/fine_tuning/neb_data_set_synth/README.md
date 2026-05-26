# NEB Data Selection

`neb_data_set_synth` turns NEB OUTCAR trees into curated training data.

The workflow is driven by a `siv_rules.yml` file that defines:

- output directory and filename prefix
- total sample count
- optional D3 subtraction
- NEB sources and image lists
- the selection rule
- train/val/test split percentages

The curated output is written as `extxyz` files plus bookkeeping files.

Run it through the package CLI:

```bash
mlip-ft mace --curate-neb --inputs demo/fine_tuning/mace/0_raw_inputs/siv_rules.yml
```

Use `mace` when the final artifact should stay as `extxyz`.

For ORB, the same command line is used with `orb` as the family key. That
invokes the same curation pipeline and then converts the curated `extxyz`
splits into ASE sqlite databases in place:

```bash
mlip-ft orb --curate-neb --inputs demo/fine_tuning/orb/0_raw_inputs/siv_rules.yml
```

The family key is what tells `mlip-ft` which final artifact form to write.
`petmad` follows the same `extxyz` final form as `mace`:

```bash
mlip-ft petmad --curate-neb --inputs demo/fine_tuning/petmad/0_raw_inputs/siv_rules.yml
```
