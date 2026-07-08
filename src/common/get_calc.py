from __future__ import annotations
import __main__
from pathlib import Path
from typing import Callable, Any, Dict




"""
Help(load_model):
load_model(path: 'str | Path', **kwargs)
    Convenience method to load a model from a directory or name.

    Args:
        path (str|path): Path to saved model or name of pre-trained model. The search order is path, followed by
            download from PRETRAINED_MODELS_BASE_URL (with caching).
        **kwargs: Additional kwargs passed to RemoteFile class. E.g., a useful one might be force_download if you
            want to update the model.

    Returns:
        Returns: model_object if include_json is false. (model_object, dict) if include_json is True.
"""
#get_calc_object("CHGNet-MatPES-PBE-2025.2.10-2.7M-PES")





def _resolve_model_path(models_root: Path, *segments: str) -> Path:
    """Try backend-specific path first, then fall back to a flat model dir."""
    candidate = models_root.joinpath(*segments)
    if candidate.exists():
        return candidate
    fallback = models_root / segments[-1]
    if fallback.exists():
        return fallback
    return candidate


def _ensure_suffix(model_name: str, suffix: str) -> str:
    """Return the on-disk filename for backends that store single-file weights."""
    return model_name if model_name.endswith(suffix) else f"{model_name}{suffix}"

def _mace_builder(model_rel: str) -> Callable[[Path, str, str], Any]:
    def _build(models_root: Path, device: str, dtype: str) -> Any:
        from mace.calculators import MACECalculator
        model_path = _resolve_model_path(models_root, "mace", _ensure_suffix(model_rel, ".model"))
        return MACECalculator(model_path=str(model_path), device=device, default_dtype=dtype)
    return _build

def _mattersim_builder(model_rel: str) -> Callable[[Path, str, str], Any]:
    def _build(models_root: Path, device: str, dtype: str) -> Any:
        from mattersim.forcefield import MatterSimCalculator
        model_path = _resolve_model_path(models_root, "mattersim", _ensure_suffix(model_rel, ".pth"))
        return MatterSimCalculator(from_checkpoint=str(model_path), device=device)
    return _build

def _matgl_builder(model_dir: str) -> Callable[[Path, str, str], Any]:
    def _build(models_root: Path, device: str, dtype: str) -> Any:
        import os
        os.environ["MATGL_BACKEND"] = "DGL"
        import matgl
        matgl.set_backend("DGL")
        from matgl.utils.io import load_model
        from matgl.ext._ase_dgl import PESCalculator

        model_path = _resolve_model_path(models_root, "matgl", "pretrained_models", model_dir)
        pot = load_model(model_path, force_download=False)
        pot.cuda()
        return PESCalculator(pot)
    return _build

def _pet_builder(model_rel: str, energy_variant: str | None = None) -> Callable[[Path, str, str], Any]:
    def _build(models_root: Path, device: str, dtype: str) -> Any:
        from metatomic.torch.ase_calculator import MetatomicCalculator
        model_path = _resolve_model_path(models_root, "petmad", "upet", _ensure_suffix(model_rel, ".pt"))
        variants = None
        if energy_variant is not None:
            variants = {
                "energy": energy_variant,
                "non_conservative_forces": None,
                "non_conservative_stress": None,
            }
        return MetatomicCalculator(
            str(model_path),
            device=device,
            variants=variants,
            non_conservative=True,
        )
    return _build

def _orb_builder(model_rel: str, precision: str) -> Callable[[Path, str, str], Any]:
    def _build(models_root: Path, device: str, dtype: str) -> Any:

        # orb models as i have encountered them have unique functions to load 
        # their respective weights. hence we need a dictionary to call the particular function
        # if one wants to add a new orb model then one will have to get ofc a) get the model weights
        # b) add the model name to config.yml
        # c) add the model name to the below function dictionary, and also to the model_build dictionary
        # d) add its corresponding orb model load function to the below dictionary. and that is that.
        

        """
        # below is an attempt to condense the if statements however i am yet to test 
        # whether orb_d3_sm_v2 accepts the precision argument. 

        orb_model_function_dict = {
            "orb-v3-direct-inf-omat": pretrained.orb_v3_direct_inf_omat,
            "orb-v3-conservative-inf-omat": pretrained.orb_v3_conservative_inf_omat,
            "orb-d3-sm-v2": pretrained.orb_d3_sm_v2,
        }
        orb_function = orb_model_function_dict[model_name]

        orbff = orb_function(
            weights_path = str(model_path),
            device = device, 
            precision = dtype,
        )
        """

        import torch

        from orb_models.forcefield import pretrained
        from orb_models.forcefield.inference.calculator import ORBCalculator

        if dtype == "float32": 
            dtype = "float32-high"

        model_file = _ensure_suffix(model_rel, ".ckpt")
        model_path = _resolve_model_path(models_root, "orb", model_file)
        model_name = str(Path(model_file).stem)

        if model_name == "orb_demo_neb_ft_lora":
            import json
            from argparse import Namespace

            from fine_tuning.fine_tuning_scripts.orb.train_orb import make_orb_model

            checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            config_path = model_path.with_suffix(".config.json")
            if not config_path.exists():
                raise FileNotFoundError(f"Missing ORB finetune config: {config_path}")
            config = json.loads(config_path.read_text())
            args = Namespace(**config)
            orbff, atoms_adapter = make_orb_model(args, torch.device(device))
            orbff.load_state_dict(state_dict, strict=True)
            orbff = orbff.to(device).eval()
            return ORBCalculator(orbff, atoms_adapter, device=device)

        if model_name == "orb-v3-direct-inf-omat":
            orbff, atoms_adapter = pretrained.orb_v3_direct_inf_omat(
                weights_path=str(model_path),
                device=device,
                precision=dtype,
            )
        elif model_name == "orb-v3-conservative-inf-omat":
            orbff, atoms_adapter = pretrained.orb_v3_conservative_inf_omat(
                weights_path=str(model_path),
                device=device,
                precision=dtype,
            )
        elif model_name == "orb-d3-sm-v2":
            orbff, atoms_adapter = pretrained.orb_d3_sm_v2(
                weights_path=str(model_path),
                device=device,
            )
        else:
            raise ValueError(
                f"Model not recognised for {model_name}, check it exists in get_calc_object"
            )

        return ORBCalculator(orbff, atoms_adapter, device=device)
    return _build


model_build: Dict[str, Callable[[Path, str, str], Any]] = {
    "small-omat-0": _mace_builder("mace-omat-0-small"),
    "mace-omat-0-medium": _mace_builder("mace-omat-0-medium"),
    "mace-mpa-0-medium": _mace_builder("mace-mpa-0-medium"),
    "mace-matpes-pbe-omat-ft": _mace_builder("MACE-matpes-pbe-omat-ft"),
    "mace-matpes-r2scan-omat-ft": _mace_builder("MACE-matpes-r2scan-omat-ft"),
    
    "ivac0_neb_ft_1": _mace_builder("ivac0_neb_ft_1"),
    "ivac0_neb_ft_compiled_1": _mace_builder("ivac0_neb_ft_compiled_1"),

    "ivac0_neb_ft_2": _mace_builder("ivac0_neb_ft_1"),
    "ivac0_neb_ft_compiled_2": _mace_builder("ivac0_neb_ft_compiled_1"),

    "ivac0_neb_ft_3": _mace_builder("ivac0_neb_ft_3"),

    "neutral_lora_optuna_v2_trial032": _mace_builder("neutral_lora_optuna_v2_trial032"),

    "mattersim-v1.0.0-1M": _mattersim_builder("mattersim-v1.0.0-1M"),
    "mattersim-v1.0.0-5M": _mattersim_builder("mattersim-v1.0.0-5M"),

    "orb-v3-direct-inf-omat": _orb_builder("orb-v3-direct-inf-omat", precision="float32-high"),
    "orb-d3-sm-v2": _orb_builder("orb-d3-sm-v2", precision="float64"),
    "orb-v3-conservative-inf-omat": _orb_builder("orb-v3-conservative-inf-omat", precision="float64"),
    "orb_demo_neb_ft_lora": _orb_builder("orb_demo_neb_ft_lora", precision="float64"),

    "pet-mad-s-v1.1.0": _pet_builder("pet-mad-s-v1.1.0"),
    "pet-omad-l-v0.1.0": _pet_builder("pet-omad-l-v0.1.0"),
    "pet-omad-s-v1.0.0": _pet_builder("pet-omad-s-v1.0.0"),
    "pet-omat-l-v1.0.0": _pet_builder("pet-omat-l-v1.0.0"),
    "pet-omat-m-v1.0.0": _pet_builder("pet-omat-m-v1.0.0"),
    "pet-omat-xl-v1.0.0": _pet_builder("pet-omat-xl-v1.0.0"),
    "petmad_demo_neb_ft": _pet_builder("petmad_demo_neb_ft", energy_variant="neb_ft"),

    "mace_demo_neb_ft_lora": _mace_builder("mace_demo_neb_ft_lora"),

    "CHGNet-MatPES-r2SCAN-2025.2.10-2.7M-PES": _matgl_builder("CHGNet-MatPES-r2SCAN-2025.2.10-2.7M-PES"),
    "CHGNet-MatPES-PBE-2025.2.10-2.7M-PES": _matgl_builder("CHGNet-MatPES-PBE-2025.2.10-2.7M-PES"),
    "CHGNet-MPtrj-2023.12.1-2.7M-PES": _matgl_builder("CHGNet-MPtrj-2023.12.1-2.7M-PES"),
    "CHGNet-MPtrj-2024.2.13-11M-PES": _matgl_builder("CHGNet-MPtrj-2024.2.13-11M-PES"),
    "M3GNet-ANI-1x-Subset-PES": _matgl_builder("M3GNet-ANI-1x-Subset-PES"),
    "M3GNet-MatPES-PBE-v2025.1-PES": _matgl_builder("M3GNet-MatPES-PBE-v2025.1-PES"),
    "M3GNet-MatPES-r2SCAN-v2025.1-PES": _matgl_builder("M3GNet-MatPES-r2SCAN-v2025.1-PES"),
    "M3GNet-MP-2021.2.8-DIRECT-PES": _matgl_builder("M3GNet-MP-2021.2.8-DIRECT-PES"),
    "M3GNet-MP-2021.2.8-PES": _matgl_builder("M3GNet-MP-2021.2.8-PES"),
    "TensorNetDGL-MatPES-PBE-v2025.1-PES": _matgl_builder("TensorNetDGL-MatPES-PBE-v2025.1-PES"),
    "TensorNetDGL-MatPES-r2SCAN-v2025.1-PES": _matgl_builder("TensorNetDGL-MatPES-r2SCAN-v2025.1-PES"),
}

subdir_model_build: Dict[str, Callable] = {
    "mace": _mace_builder,
    "mattersim": _mattersim_builder, 
    "orb": lambda model_rel: _orb_builder(model_rel, precision="float64"),
    "petmad": _pet_builder, 
    "upet": _pet_builder, 
    "matgl": _matgl_builder,
}


def _find_builder(models_root: Path, model: str):
    """
    if the model is not explicitly known in the model_build dictionary, we attempt to identify 
    the correrct builder based upon the name of subdirectory where the model is stored, as 
    determined by subdir_model_build.

    """
    cand_model_path = next(models_root.glob(f'**/{model}.*'), None)
    print(cand_model_path)
    if not cand_model_path:
        raise FileNotFoundError(f"Model '{model}' does not exist in provided models_root: '{models_root}'. ")
    cand_model_subdir = cand_model_path.parent
    if cand_model_subdir == models_root: 
        raise ValueError(f"Model '{model}' was found to be in the base models_root. Cannot infer how to build the model, too much degeneracy.")
    
    cand_subdir_str = str(cand_model_subdir.name)
    try: 
        builder = subdir_model_build[cand_subdir_str]
        return builder
    except KeyError:
        raise KeyError(
            f"Subdirectory '{cand_model_subdir}' in '{models_root}' is not tied to a builder via the dictionary "
            "'subdir_model_build' in <repo_root>/src/common/get_calc.py."
        )

    


def get_calc_object(
    model: str,
    models_root: str | Path | None = None,
    device: str = "cuda",
    dtype: str = "float32",
    *,
    include_vdw: bool = False,
    vdw_method: str = "PBE",
    vdw_damping: str = "d3bj",
    vdw_abc: bool = False,
    vdw_params_tweaks: dict[str, float] | None = None,
    vdw_realspace_cutoff: dict[str, float] | None = None,
    vdw_cache_api: bool = True,
) -> Any:
    models_root = Path(models_root) if models_root is not None else Path(__file__).resolve().parents[2] / Path("assets") / "models"

    try:
        calc = model_build[model](models_root, device, dtype)
    except KeyError:
        print(f"WARNING: Unknown model '{model}' — It does not have a defined and dedicated builder to it",
              "in the dictionary 'model_build' in 'src/common/get_calc.py'.",
              f"INSTEAD i will attempt to infer model build by finding the first subdirectory location in '{models_root}'",
              "containing the model name.")
        builder = _find_builder(models_root, model)
        calc = builder(model)(models_root, device, dtype)

    has_builtin_dispersion = ("orb-d3" in model.lower())

    if include_vdw and not has_builtin_dispersion:
        from ase.calculators.mixing import SumCalculator
        from dftd3.ase import DFTD3

        d3_kwargs: dict[str, Any] = {
            "method": vdw_method,
            "damping": vdw_damping,
            "cache_api": vdw_cache_api,
        }
        if vdw_params_tweaks is not None:
            d3_kwargs["params_tweaks"] = dict(vdw_params_tweaks)
        if vdw_realspace_cutoff is not None:
            d3_kwargs["realspace_cutoff"] = dict(vdw_realspace_cutoff)

        d3 = DFTD3(**d3_kwargs)
        calc = SumCalculator([calc, d3])

    return calc


mace_env = [
    "small-omat-0",
    "mace-omat-0-medium",
    "mace-mpa-0-medium",
    "mace-matpes-pbe-omat-ft",
    "mace-matpes-r2scan-omat-ft",
    "orb-v3-direct-inf-omat",
    "orb-d3-sm-v2",
    "orb-v3-conservative-inf-omat",
    "pet-mad-s-v1.1.0",
    "pet-omad-l-v0.1.0",
    "pet-omad-s-v1.0.0",
    "pet-omat-l-v1.0.0",
    "pet-omat-m-v1.0.0",
    "pet-omat-xl-v1.0.0",
]

matgl_env = [
    "CHGNet-MatPES-r2SCAN-2025.2.10-2.7M-PES",
    "CHGNet-MatPES-PBE-2025.2.10-2.7M-PES",
    "CHGNet-MPtrj-2023.12.1-2.7M-PES",
    "CHGNet-MPtrj-2024.2.13-11M-PES",
    "M3GNet-ANI-1x-Subset-PES",
    "M3GNet-MatPES-PBE-v2025.1-PES",
    "M3GNet-MatPES-r2SCAN-v2025.1-PES",
    "M3GNet-MP-2021.2.8-DIRECT-PES",
    "M3GNet-MP-2021.2.8-PES",
    "TensorNetDGL-MatPES-PBE-v2025.1-PES",
    "TensorNetDGL-MatPES-r2SCAN-v2025.1-PES",
]

mattersim_env = [
    "mattersim-v1.0.0-1M",
    "mattersim-v1.0.0-5M",
]

        
    
if __name__ == "__main__":
    for model in matgl_env: #mattersim_env: #mace_env: #matgl_env: #mattersim_env: #matgl_env: #mace_env: #+ matgl_env + mattersim_env:
        try:
            calc = get_calc_object(model)
            print(type(calc))
            print(f"Successfully loaded calculator for model: {model}")
            #if hasattr(calc, "element_types"):
            #    print("model element_types:", calc.element_types)
            if hasattr(calc, "model") and hasattr(calc.model, "element_types"):
                print("model element_types:", calc.model.element_types)  
        except Exception as e:
            print(f"Failed to load calculator for model: {model}. Error: {e}")



# Old version. havnt fully tested the implementtion  of the new one 
#( should be fine) but wanted to keep this just in case i stuffed up smth.

"""
def get_calc_object2(model: str, models_root: str | Path | None = None) -> Any:
    #Construct a calculator by model name using local model paths.

    #Args:
    #    model (str): Model name key corresponding to an on-disk checkpoint.
    #    models_root (str | Path | None): folder containing the folders containing the models (default: ./assets/models).

    #Returns:
    #    Any: ASE-compatible calculator instance for the requested model.
    
    models_root = Path(models_root) if models_root is not None else Path("assets") / "models"

    calc: Any | None = None

    if model == "small-omat-0":
        from mace.calculators import MACECalculator
        model_path = models_root / "mace" / "mace-omat-0-small.model"
        calc = MACECalculator(model_path=str(model_path), device="cuda", default_dtype="float32")
        print(f"using {model} calculator")

    elif model == "mace-omat-0-medium":
        from mace.calculators import MACECalculator
        model_path = models_root / "mace" / "mace-omat-0-medium.model"
        calc = MACECalculator(
            model_path=str(model_path),
            device="cuda",
            default_dtype="float32",
        )
        print(f"using {model} calculator")

    elif model == "mace-mpa-0-medium":
        from mace.calculators import MACECalculator
        model_path = models_root / "mace" / "mace-mpa-0-medium.model"
        calc = MACECalculator(
            model_path=str(model_path),
            device="cuda",
            default_dtype="float32",
        )
        print(f"using {model} calculator")

    elif model == "mace-matpes-pbe-omat-ft":
        from mace.calculators import MACECalculator
        model_path = models_root / "mace" / "MACE-matpes-pbe-omat-ft.model"
        calc = MACECalculator(
            model_path=str(model_path),
            device="cuda",
            default_dtype="float32",
        )
        print(f"using {model} calculator")

    elif model == "mace-matpes-r2scan-omat-ft":
        from mace.calculators import MACECalculator
        model_path = models_root / "mace" / "MACE-matpes-r2scan-omat-ft.model"
        calc = MACECalculator(
            model_path=str(model_path),
            device="cuda",
            default_dtype="float32",
        )
        print(f"using {model} calculator")

    elif model == "mattersim-v1.0.0-1M":
        from mattersim.forcefield import MatterSimCalculator
        checkpoint_path = models_root / "mattersim" / "mattersim-v1.0.0-1M.pth"
        calc = MatterSimCalculator(from_checkpoint=str(checkpoint_path), device="cuda")
        print(f"using {model} calculator")

    elif model == "mattersim-v1.0.0-5M":
        from mattersim.forcefield import MatterSimCalculator
        checkpoint_path = models_root / "mattersim" / "mattersim-v1.0.0-5M.pth"
        calc = MatterSimCalculator(from_checkpoint=str(checkpoint_path), device="cuda")
        print(f"using {model} calculator")

    elif model == "orb-v3-direct-inf-omat":  # ORB models are not working.
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        model_path_ = models_root / "orb" / "orb-v3-direct-inf-omat.ckpt"
        device_ = "cuda"
        precision_ = "float32-high" # or "float32-highest" / "float64"

        orbff = pretrained.orb_v3_direct_inf_omat(
            weights_path=str(model_path_),
            device=device_,
            #precision=precision_,
        )
        calc = ORBCalculator(orbff, device=device_)
        print(f"using {model} calculator")

    elif model == "orb-d3-sm-v2":
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        model_path_ = models_root / "orb" / "orb-d3-sm-v2.ckpt"
        device_ = "cuda"
        precision_ = "float64" # or "float32-highest" / "float64"

        orbff = pretrained.orb_d3_sm_v2(
            weights_path=str(model_path_),
            device=device_,
            #precision=precision_,
        )
        calc = ORBCalculator(orbff, device=device_)
        print(f"using {model} calculator")

    elif model == "orb-v3-conservative-inf-omat":
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        model_path_ = models_root / "orb" / "orb-v3-conservative-inf-omat.ckpt"
        device_ = "cuda"
        precision_ = "float64" # or "float32-highest" / "float64"

        orbff = pretrained.orb_v3_conservative_inf_omat(
            weights_path=str(model_path_),
            device=device_,
            precision=precision_,
        )
        calc = ORBCalculator(orbff, device=device_)
        print(f"using {model} calculator")

    elif model == "pet-mad-s-v1.1.0":
        
        #petmad requires downloading files from their initialisation and moving them out
        #of the cache directory (do not moove the hugging face files those are symmlinks to the
        #upet cache), or the use of the the upet.save_upet() function.

        #models can then be loaded using metatomic.torch.ase_calculator and the path to the file.
        #the file should be a torch script file (.pt), or a .chpt file, a sanity check is whether the
        #file is in bytes.
        
        checkpoint_path_ = models_root / "petmad" / "upet" / "pet-mad-s-v1.1.0.pt"
        from metatomic.torch.ase_calculator import MetatomicCalculator
        calc = MetatomicCalculator(str(checkpoint_path_), device="cuda", non_conservative=True)
        print(f"using {model} calculator")

    elif model == "pet-omad-l-v0.1.0":
        checkpoint_path_ = models_root / "petmad" / "upet" / "pet-omad-l-v0.1.0.pt"
        from metatomic.torch.ase_calculator import MetatomicCalculator
        calc = MetatomicCalculator(str(checkpoint_path_), device="cuda", non_conservative=True)
        print(f"using {model} calculator")

    elif model == "pet-omad-s-v1.0.0":
        checkpoint_path_ = models_root / "petmad" / "upet" / "pet-omad-s-v1.0.0.pt"
        from metatomic.torch.ase_calculator import MetatomicCalculator
        calc = MetatomicCalculator(str(checkpoint_path_), device="cuda", non_conservative=True)
        print(f"using {model} calculator")

    elif model == "pet-omat-l-v1.0.0":
        checkpoint_path_ = models_root / "petmad" / "upet" / "pet-omat-l-v1.0.0.pt"
        from metatomic.torch.ase_calculator import MetatomicCalculator
        calc = MetatomicCalculator(str(checkpoint_path_), device="cuda", non_conservative=True)
        print(f"using {model} calculator")

    elif model == "pet-omat-m-v1.0.0":
        checkpoint_path_ = models_root / "petmad" / "upet" / "pet-omat-m-v1.0.0.pt"
        from metatomic.torch.ase_calculator import MetatomicCalculator
        calc = MetatomicCalculator(str(checkpoint_path_), device="cuda", non_conservative=True)
        print(f"using {model} calculator")

    elif model == "pet-omat-xl-v1.0.0":
        checkpoint_path_ = models_root / "petmad" / "upet" / "pet-omat-xl-v1.0.0.pt"
        from metatomic.torch.ase_calculator import MetatomicCalculator
        calc = MetatomicCalculator(str(checkpoint_path_), device="cuda", non_conservative=True)
        print(f"using {model} calculator")

    ###########################

    elif model == "CHGNet-MatPES-PBE-2025.2.10-2.7M-PES":
        import os
        os.environ["MATGL_BACKEND"] = "DGL"  # set first

        import matgl
        matgl.set_backend("DGL")

        from matgl.utils.io import load_model
        from matgl.ext._ase_dgl import PESCalculator

        model_dir = models_root / "matgl" / "pretrained_models" / "CHGNet-MatPES-PBE-2025.2.10-2.7M-PES"
        pot = load_model(model_dir, force_download=False)
        print(type(pot))
        pot.cuda()
        calc = PESCalculator(pot)
        print(type(calc))
        print(f"using {model} calculator")

    elif model == "CHGNet-MatPES-r2SCAN-2025.2.10-2.7M-PES":
        import os
        os.environ["MATGL_BACKEND"] = "DGL"  # set first

        import matgl
        matgl.set_backend("DGL")

        from matgl.utils.io import load_model
        from matgl.ext._ase_dgl import PESCalculator

        model_dir = models_root / "matgl" / "pretrained_models" / "CHGNet-MatPES-r2SCAN-2025.2.10-2.7M-PES"
        pot = load_model(model_dir, force_download=False)
        pot.cuda()
        calc = PESCalculator(pot)
        print(f"using {model} calculator")

    elif model == "CHGNet-MPtrj-2023.12.1-2.7M-PES":
        import os
        os.environ["MATGL_BACKEND"] = "DGL"  # set first

        import matgl
        matgl.set_backend("DGL")

        from matgl.utils.io import load_model
        from matgl.ext._ase_dgl import PESCalculator

        model_dir = models_root / "matgl" / "pretrained_models" / "CHGNet-MPtrj-2023.12.1-2.7M-PES"
        pot = load_model(model_dir, force_download=False)
        pot.cuda()
        calc = PESCalculator(pot)
        print(f"using {model} calculator")

    elif model == "CHGNet-MPtrj-2024.2.13-11M-PES":
        import os
        os.environ["MATGL_BACKEND"] = "DGL"  # set first

        import matgl
        matgl.set_backend("DGL")

        from matgl.utils.io import load_model
        from matgl.ext._ase_dgl import PESCalculator

        model_dir = models_root / "matgl" / "pretrained_models" / "CHGNet-MPtrj-2024.2.13-11M-PES"
        pot = load_model(model_dir, force_download=False)
        pot.cuda()
        calc = PESCalculator(pot)
        print(f"using {model} calculator")

    elif model == "M3GNet-ANI-1x-Subset-PES":
        import os
        os.environ["MATGL_BACKEND"] = "DGL"  # set first

        import matgl
        matgl.set_backend("DGL")
        
        from matgl.utils.io import load_model
        from matgl.ext._ase_dgl import PESCalculator


        model_dir = models_root / "matgl" / "pretrained_models" / "M3GNet-ANI-1x-Subset-PES"
        pot = load_model(model_dir, force_download=False)


        pot.cuda()

        calc = PESCalculator(pot)
        print(f"using {model} calculator")

    elif model == "M3GNet-MatPES-PBE-v2025.1-PES":

        import os
        os.environ["MATGL_BACKEND"] = "DGL"  # set first

        import matgl
        matgl.set_backend("DGL")

        from matgl.utils.io import load_model
        from matgl.ext._ase_dgl import PESCalculator

        model_dir = models_root / "matgl" / "pretrained_models" / "M3GNet-ANI-1x-Subset-PES"
        pot = load_model(model_dir, force_download=False)
        pot.cuda()
        calc = PESCalculator(pot)
        print(f"using {model} calculator")

    elif model == "M3GNet-MatPES-r2SCAN-v2025.1-PES":

        import os
        os.environ["MATGL_BACKEND"] = "DGL"  # set first

        import matgl
        matgl.set_backend("DGL")

        from matgl.utils.io import load_model
        from matgl.ext._ase_dgl import PESCalculator

        model_dir = models_root / "matgl" / "pretrained_models" / "M3GNet-MatPES-r2SCAN-v2025.1-PES"
        pot = load_model(model_dir, force_download=False)
        pot.cuda()
        calc = PESCalculator(pot)
        print(f"using {model} calculator")

    elif model == "M3GNet-MP-2021.2.8-DIRECT-PES":

        import os
        os.environ["MATGL_BACKEND"] = "DGL"  # set first

        import matgl
        matgl.set_backend("DGL")

        from matgl.utils.io import load_model
        from matgl.ext._ase_dgl import PESCalculator

        model_dir = models_root / "matgl" / "pretrained_models" / "M3GNet-MP-2021.2.8-DIRECT-PES"
        pot = load_model(model_dir, force_download=False)
        pot.cuda()
        calc = PESCalculator(pot)
        print(f"using {model} calculator")

    elif model == "M3GNet-MP-2021.2.8-PES":

        import os
        os.environ["MATGL_BACKEND"] = "DGL"  # set first

        import matgl
        matgl.set_backend("DGL")

        from matgl.utils.io import load_model
        from matgl.ext._ase_dgl import PESCalculator

        model_dir = models_root / "matgl" / "pretrained_models" / "M3GNet-MP-2021.2.8-PES"
        pot = load_model(model_dir, force_download=False)
        print(type(pot))
        pot.cuda()
        calc = PESCalculator(pot)
        print(f"using {model} calculator")

    elif model == "TensorNetDGL-MatPES-PBE-v2025.1-PES":

        import os
        os.environ["MATGL_BACKEND"] = "DGL"  # set first

        import matgl
        matgl.set_backend("DGL")

        from matgl.utils.io import load_model
        from matgl.ext._ase_dgl import PESCalculator

        model_dir = models_root / "matgl" / "pretrained_models" / "TensorNetDGL-MatPES-PBE-v2025.1-PES"
        pot = load_model(model_dir, force_download=False)
        pot.cuda()
        calc = PESCalculator(pot)
        print(f"using {model} calculator")
    
    elif model == "TensorNetDGL-MatPES-r2SCAN-v2025.1-PES":
        
        import os
        os.environ["MATGL_BACKEND"] = "DGL"  # set first

        import matgl
        matgl.set_backend("DGL")

        from matgl.utils.io import load_model
        from matgl.ext._ase_dgl import PESCalculator

        model_dir = models_root / "matgl" / "pretrained_models" / "TensorNetDGL-MatPES-r2SCAN-v2025.1-PES"
        pot = load_model(model_dir, force_download=False)
        pot.cuda()
        calc = PESCalculator(pot)
        print(f"using {model} calculator")

    if calc:
        return calc
    raise ValueError(f"Unknown model '{model}'")

"""
