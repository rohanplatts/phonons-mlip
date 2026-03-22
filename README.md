# mlip-workflows

This package is for a) executing particular workflows on various MLIPs, and b) benchmarking the MLIPs against DFT with respect to these workflows. 

**Currently supported workflows include**:
* MEP via NEB. (including vasp-exportable initial MLIP MEPs). Benchmarking capability for these MEPs is included. 
* Phonon, DOS, and band structures Calculations 
* Coupling mode analysis (of the calculated phonons), and DFT comparison / ranking
* Photoilluminescence/vibronic calculations Via [Plumipy](https://github.com/verdi-group/plumipy)

**Currently supported models include**:
* Mace, Orbital-materials, PET-mad, CHGNet, M3GNet, TensorNet and Mattersim model families


## Installation
Clone this repository:
```
git clone https://github.com/rohanplatts/phonons-mlip.git
cd phonons-mlip
```

Then identify the models you want to use by reading [ENVIRONMENTS.md](env/ENVIRONMENTS.md), and follow the environment install instructions. 


> Note: If you intend to use D3 (vdw) term correction, you will need the d3 backend from the `dftd3-python` + `simple-dftd3` packages via **conda-forge**. This enables you to utilise OpenMP's parallelisation. If you want to do that, see [README.md](src/NEB/README.md).

For a HPC install guide (slurm orirented) see [HPC_install.md](env\HPC_install.md)


## Configuration
Next step is to configure this such that it works for you. First you need to obtain the model files you are interested in, see [MODELS.md](assets/models/MODELS.md). 
As a default, we have included `mace-mpa-0-medium` for ease of testing. Models should be saved in the `ssets/models/<model_family>/<model_file>` directory. You can see `mace-mpa-0-medium.model` in `assets/models/mace/mace-mpa-0-medium.model`. Any models that you would like to test should be copied into `assets/models/<model_family>/<your_model_weight_file>`. 

For a NEB Quickstart, see [NEB_quickstart.md](src/NEB/NEB_quickstart.md)

## Run Phonon Workflow

The main entry point is the console script (defaults read from `config.yml` under `mlip_phonons.defaults`):

If you would like to obtain the phonons using a particular model, on a particular structure, then you may run either of the following:

```bash
mlip-phonons
mlip-phonons <model_name>
mlip-phonons <model_name> --structure <structure_key>
```
The less variables you flag, the more variables that are pulled from [config.yml](config.yml).

Structure_key is read from config.yml, under `structures`. See config.yml for setting this up for a given structure you have. 
For more details on phonon calculations see `src/mlip_phonons/README.md`

## Outputs

Phonon outputs are written under:

```text
results/<model_name>/<structure_key>/
  raw/
  plot/
```

Additionally, Plumipy-ready (a package for obtaining PL spectra) files are written under:

```text
results/<model_name>/<structure_key>/raw/Plumipy_Files/
```

Note: there are some TODOs in this repo about splitting subprojects (e.g., coupling, plumipy helpers).

## DFT vs ML Coupling/Ranking 

`src/coupling_modes/phonon_coupling.py` compares a DFT `band.yaml` against one or more MLIP `band.yaml` files using:
- a GS→ES displacement vector (from `CONTCAR_GS`/`CONTCAR_ES`) for projection weights,
- mode matching via overlap + Hungarian assignment,
- weighted errors `E_freq`, `E_vec`, and combined `Score = E_freq + alpha * E_vec`,
- frequency-cluster window diagnostics.

Run:

```bash
python src/coupling_modes/phonon_coupling.py --alpha 0.5 --weight_kind S
```

Reports are printed and also saved to `resultsPhonCoupling/phonon_coupling_report_<i>.txt`.

## PL Plot Comparison 

`src/plumipy_run/exploratory_script.py` contains helpers to call `plumipy.calculate_spectrum` and generate comparison plots (partial Huang–Rhys, spectral function, PL, IPR). Requires plumipy package installed.

## What Benchmarking Is Actually Done

The repository does two main kinds of implemented MLIP-vs-DFT benchmarking, plus one plotting-oriented comparison workflow:

1. **NEB / MEP benchmarking**
   `src/NEB/NEB_compare_all.py` (accessible via command line as `mlip-neb --compare`) compares each MLIP CI-NEB path against a DFT `neb.dat`.

   It reports:
   - `mlip_barrier_eV` and `mlip_deltaE_eV`
   - `dft_barrier_eV` and `dft_deltaE_eV`
   - `barrier_abs_err_eV` and `deltaE_abs_err_eV`
   - optional force errors on DFT geometries: `force_RMSE_eV_per_A`, `max_force_err_eV_per_A`
   - optional path quality on the final MLIP path: `max_F_perp_eV_per_A`
   - timing metrics parsed from ASE optimizer logs

   The final NEB ranking is sorted by lower `barrier_abs_err_eV`, then lower `deltaE_abs_err_eV`. Outputs are written under `resultsNEB/<model>/plot/` plus a combined `resultsNEB/rankings/rankings.txt`.

2. **Phonon-coupling / mode-matching benchmarking**
   `src/coupling_modes/phonon_coupling.py` compares MLIP `band.yaml` files against a DFT `band.yaml`, using the GS->ES displacement from `CONTCAR_GS` and `CONTCAR_ES` to decide which phonon modes matter most.

   It reports:
   - `E_freq`: weighted frequency error after DFT-to-ML mode matching
   - `E_vec`: weighted eigenvector mismatch
   - `Score = E_freq + alpha * E_vec`
   - `E_freq_rel`
   - `X_mean` for coupling-subspace agreement
   - cluster-window metrics for near-degenerate eigenspaces

   a more comprehensive description on what is computed WRT phonon-coupling is included in the src/phonon_coupling folder. 

   The final phonon-coupling ranking is sorted by lower `Score_mean`. Reports are written to `resultsPhonCoupling/phonon_coupling_report_<i>.txt`.

3. **PL / vibronic comparison**
   `src/plumipy_run/exploratory_script.py` uses the MLIP phonon outputs to generate comparison plots such as Huang-Rhys contributions, spectral functions, PL spectra, and IPR-style views. This is useful for side-by-side benchmarking, but it is currently a plotting workflow rather than a single scalar ranking.

So, in short: `mlip-phonons` produces the phonon data, `NEB_compare_all.py` benchmarks MEPs against DFT, and `phonon_coupling.py` benchmarks the displacement-relevant phonon modes against DFT.
