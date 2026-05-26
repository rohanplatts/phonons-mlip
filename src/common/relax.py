from pathlib import Path
from typing import Any, Callable

from ase import Atoms
from ase.optimize import BFGS, BFGSLineSearch, FIRE, GPMin, LBFGS, MDMin

try:
    from ase.optimize.precon import PreconLBFGS
except Exception:  # pragma: no cover - optional ASE feature
    PreconLBFGS = None


_OPTIMIZERS: dict[str, Callable[..., Any]] = {
    "bfgs": BFGS,
    "bfgsls": BFGSLineSearch,
    "bfgs-line-search": BFGSLineSearch,
    "fire": FIRE,
    "lbfgs": LBFGS,
    "mdmin": MDMin,
    "gpmin": GPMin,
}
if PreconLBFGS is not None:
    _OPTIMIZERS["preconlbfgs"] = PreconLBFGS
    _OPTIMIZERS["precon-lbfgs"] = PreconLBFGS


def relax(
    structure: Atoms,
    fmax: float = 0.01,
    outdir: Path | None = None,
    filename: str | None = None,
    *,
    type: str = "BFGS",
    steps: int | None = None,
    optimizer_kwargs: dict[str, Any] | None = None,
):
    """Relax an ASE Atoms structure with a selected optimizer."""
    key = str(type).strip().lower()
    relaxer = _OPTIMIZERS.get(key)
    if relaxer is None:
        supported = ", ".join(sorted(_OPTIMIZERS.keys()))
        raise ValueError(f"Unknown relax type: {type!r}. Supported: {supported}")

    trajectory = None
    saved_outdir = None
    if outdir is not None and filename is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        trajectory = outdir / str(filename)
        saved_outdir = outdir

    kwargs = dict(optimizer_kwargs or {})
    if trajectory is not None:
        kwargs.setdefault("trajectory", str(trajectory))

    opt = relaxer(structure, **kwargs)
    if steps is None:
        opt.run(fmax=fmax)
    else:
        opt.run(fmax=fmax, steps=steps)

    message = f"Relaxation complete ({key})."
    if saved_outdir is not None:
        message += f' "{filename}" was saved to {saved_outdir}'
    print(message)

    return structure
