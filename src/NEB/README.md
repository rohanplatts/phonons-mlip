# `NEB`

Obtain the minimum energy path (MEP) via the nudged elastic band method for an arbitrary MLIP model file. Contains the MEP in a vasp loadable format. Accessible via command line by `mlip-neb` (from `src/NEB/run_neb_raw_v2.py`). Defaults are read from `config.yml` under `neb.defaults` and `neb.settings`.

## Setup

1. Install the repo in editable mode:
   - `pip install -e .`
2. Ensure `config.yml` has your NEB defaults in `neb.defaults` (model, endpoints, results root).
3. Ensure you have:
   - Model weights under `assets/models/` (or set `neb.defaults.models_root` to the model weight directory).
   - Endpoint structures (`POSCAR_i`, `POSCAR_f`) (or set `neb.defaults.poscar_i`/`poscar_f` in config.yml).
4. Optional: provide a DFT `neb.dat` (or set `neb.defaults.dft_neb_dat`  in config.yml) for reference comparisons.

## Running

```bash
mlip-neb

OR 

mlip-neb <model_name> --poscar-i path/to/POSCAR_i --poscar-f path/to/POSCAR_f
```

Useful flags:
- `--config`: path to `config.yml` (default `./config.yml`).
- `--n-images`: number of images (>= 3). If not provided, it tries to infer from `--dft-neb-dat` and falls back to 9.
- `--results-root`: where to write results (default `resultsNEB/` in repo).
- `--models-root`: override model weights location (default `assets/models/` relative to repo root).
- `--dft-neb-dat`: path to reference `neb.dat` for comparisons.
- `--vasp-inputs-dir`: directory containing `INCAR`, `KPOINTS`, `POTCAR` to copy into exported VASP folders.
- `--compare`: run `NEB_compare_all.py` and exit (no NEB run).
- `--relax-endpoints` / `--no-relax-endpoints`: relax initial/final endpoints before NEB.
- `--remap-f-i` / `--no-remap-f-i`: apply within‑species remapping to the final endpoint before interpolation.
- `--device`, `--dtype`: control MLIP device/dtype.
- `--overwrite` / `--no-overwrite`: reuse the same results root instead of suffixing.

## Outputs

For a model `<model_name>`, the run writes:

If the base results root already exists and `--overwrite` is not set, a suffixed directory is created (e.g., `resultsNEB_1`, `resultsNEB_2`, ...).

- `resultsNEB/<model_name>/raw/`
  - `neb_mlip.traj`, `neb_mlip.log` (coarse MLIP path, the .log file records optimizer convergence)
  - `neb_mlip_d3.traj`, `neb_mlip_d3.log` (refined MLIP+D3 path)
  - `neb_ci.traj`, `neb_ci.log` (climbing image path)
  - `neb_raw.npz` (arrays + metadata)
  - `summary.txt` (human‑readable summary)
  - `vasp_mlip_d3/00..0(N+1)/POSCAR` (pre‑climbing-image VASP‑ready path)
  - `vasp_ci/00..0(N+1)/POSCAR` (climbing‑image VASP‑ready path)

## VASP Export Requirements

The VASP‑ready folders contain the numbered image directories and POSCARs. To run VASP NEB, you also need `INCAR`, `KPOINTS`, and `POTCAR` in the parent folder:

```
vasp_neb/
├── INCAR            
├── KPOINTS
├── POTCAR
├── 00/POSCAR
├── 01/POSCAR
└── 0(N+1)/POSCAR
```

If you pass `--vasp-inputs-dir`, these three files are copied into `vasp_mlip_d3/` and `vasp_ci/`. If you do not pass that vasp input directory, then:
TO BE IMPLEMENTED: the minimum vasp loadable POTCAR and INCAR are created, with KPOINTS skipped by setting `KGAMMA = .TRUE.`

## Finding the MEP via MANY models

If you would like to compare the MEPs for multiple MLIPs, then you will not want to have to run mlip-neb multiple times. That would be annoying. For this reason I have a BASH script main_neb.bs that reads the list of models you want to evaluate from config.yml `models` list, reads the environment that is defined under the specific model, activates that environment, and then runs `mlip-neb <model_name>`. Your default values will be pulled from config.yml and used as the settings for that MEP calculation. 

This will give you all of the MEP paths as calculated from NEB on the potential energy surface of all of the MLIPs. You can then compare the outputs of the models:

## Comparing & Ranking

Use `NEB_compare_all.py` to compare MLIP paths against DFT (defaults come from `config.yml` under `neb.defaults` and can be overridden by flags):

```bash
python src/NEB/NEB_compare_all.py --results-root resultsNEB --dft-neb-dat path/to/neb.dat
```

You can also run the comparison through the CLI:

```bash
mlip-neb --compare --results-root resultsNEB --dft-neb-dat path/to/neb.dat
```
If you are comparing MLIP results obtained from mlip-neb, (ensure to use --overwrite flag when comparing different models so that results are pooled into the same resultsNEB directory) this should not be a problem, all that you will need to do is supply the directory of resultsNEB. Otheriwse, It expects the results root to be the directory containing a folder tree like: 
```
<model_name>/raw/
<model_name>/raw/neb_raw.npz
<model_name>/raw/neb_mlip.log || this is not necessary, but used for timing metrics
```

Outputs:
- `resultsNEB/<model>/plot/mep_compare.png`
- `resultsNEB/<model>/plot/report.md` + `report.html`
- `resultsNEB/rankings/rankings.txt`

## Prepare VASP Single-Point Inputs

To compare against DFT single-point energies on the MLIP path, first prepare a VASP-ready folder
that you can copy to an HPC:

```bash
python src/NEB/NEB_prepare_vasp_singlepoints.py \
  --results-root resultsNEB \
  --model <model_name> \
  --path vasp_ci \
  --vasp-inputs-dir path/to/vasp_inputs
```

This copies `INCAR/KPOINTS/POTCAR` into each image folder and writes a helper
`run_vasp_singlepoints.sh` in the selected path. You can copy the prepared
`raw/vasp_ci/` (or `raw/vasp_mlip_d3/`) directory to your HPC and run the script there.

After DFT runs complete, you can compare with:

```bash
mlip-neb --compare --results-root resultsNEB --dft-neb-dat path/to/neb.dat
```

Metrics computed by `NEB_compare_all.py`:
- `mlip_barrier_eV`, `mlip_deltaE_eV`: max barrier and final delta‑E from MLIP CI‑NEB path.
- `dft_barrier_eV`, `dft_deltaE_eV`: same metrics from the DFT `neb.dat`.
- Timing stats from optimizer logs (if present): `Mlip dt`, `Mlip_d3 dt`, `mlip_d3 climb dt`, and `Total NEB time (s)`.
- Optional force agreement vs DFT (requires DFT VASP NEB images with `OUTCAR`s to be in the same directory as `neb.dat`): `force_RMSE_eV_per_A`, `max_force_err_eV_per_A`.
- Optional path quality: `max_F_perp_eV_per_A` computed on the final MLIP path.

Optional metrics:
- Force errors vs DFT (`force_RMSE_eV_per_A`, `max_force_err_eV_per_A`) if the DFT NEB directory includes image `OUTCAR`s.
- Path quality (`max_F_perp_eV_per_A`) from `raw/vasp_ci/`.

## Notes

- `src/NEB/run_neb_raw_OLD.py` is legacy; `run_neb_raw_v2.py` is the maintained entrypoint.
- `src/NEB/checking_neb.py` is a small utility to remap endpoints and print basic displacement metrics (useful to check if the indices of an initial and final configuration are correctly matched)
