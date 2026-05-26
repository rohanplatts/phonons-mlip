#!/usr/bin/env python3
"""Fine-tune ORB force-field models on ASE sqlite datasets.

This script intentionally stays outside the orb-models package.  It adds two
repo-specific conveniences:
  1. replay mixing from a second ASE DB, if provided;
  2. generic LoRA adapters for torch.nn.Linear layers.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from pathlib import Path
from typing import Iterable
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import (
    BatchSampler,
    ConcatDataset,
    DataLoader,
    RandomSampler,
    SequentialSampler,
    WeightedRandomSampler,
)

try:
    from orb_models.common.dataset import augmentations, property_definitions
    from orb_models.common.dataset.ase_sqlite_dataset import AseSqliteDataset
    from orb_models.common.dataset.loaders import worker_init_fn
    from orb_models.common.training.util import init_device, set_torch_precision
    from orb_models.common.utils import seed_everything
    from orb_models.forcefield import pretrained
except ModuleNotFoundError as exc:
    raise SystemExit(
        "orb_models is not installed. Create/activate the ORB environment first, "
        "for example with ./install_orb_env.sh in this folder."
    ) from exc


LOG = logging.getLogger("orb_finetune")


class LoRALinear(nn.Module):
    """A frozen Linear layer plus trainable low-rank update."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scaling


def count_parameters(model: nn.Module, *, trainable_only: bool) -> int:
    return sum(
        param.numel()
        for param in model.parameters()
        if (param.requires_grad or not trainable_only)
    )


def find_parent_module(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def install_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    target_regex: str,
    exclude_regex: str,
) -> list[str]:
    target = re.compile(target_regex)
    exclude = re.compile(exclude_regex) if exclude_regex else None
    replaced: list[str] = []

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not target.search(name):
            continue
        if exclude and exclude.search(name):
            continue
        parent, child_name = find_parent_module(model, name)
        setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
        replaced.append(name)

    if not replaced:
        raise ValueError(
            f"No torch.nn.Linear modules matched --lora-target-regex={target_regex!r}"
        )
    return replaced


def set_requires_grad_matching(model: nn.Module, patterns: Iterable[str], value: bool) -> int:
    regexes = [re.compile(pattern) for pattern in patterns if pattern]
    changed = 0
    for name, param in model.named_parameters():
        if any(regex.search(name) for regex in regexes):
            param.requires_grad = value
            changed += param.numel()
    return changed


def make_orb_model(args: argparse.Namespace, device: torch.device):
    is_conservative = "conservative" in args.base_model
    loss_weights: dict[str, float] = {
        "energy": args.energy_loss_weight,
        "confidence": args.confidence_loss_weight,
    }
    if is_conservative:
        loss_weights["grad_forces"] = args.forces_loss_weight
        loss_weights["grad_stress"] = args.stress_loss_weight
        if args.equigrad_loss_weight is not None:
            loss_weights["rotational_grad"] = args.equigrad_loss_weight
    else:
        loss_weights["forces"] = args.forces_loss_weight
        loss_weights["stress"] = args.stress_loss_weight

    loader = getattr(pretrained, args.base_model, None)
    if loader is None and hasattr(pretrained, "ORB_PRETRAINED_MODELS"):
        loader = pretrained.ORB_PRETRAINED_MODELS.get(args.base_model)
    if loader is None:
        raise ValueError(f"Unknown ORB base model: {args.base_model}")

    kwargs = {
        "device": device,
        "precision": args.precision,
        "compile": False,
        "train": True,
        "train_reference_energies": args.trainable_reference_energies,
        "loss_weights": loss_weights,
    }
    if args.weights_path:
        kwargs["weights_path"] = args.weights_path

    model, atoms_adapter = loader(**kwargs)
    if args.stress_loss_weight == 0.0 and hasattr(model, "disable_stress"):
        model.disable_stress()
        LOG.info("Stress training disabled because --stress-loss-weight=0.0")

    if args.lora:
        for param in model.parameters():
            param.requires_grad = False
        replaced = install_lora(
            model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_regex=args.lora_target_regex,
            exclude_regex=args.lora_exclude_regex,
        )
        LOG.info("Installed LoRA adapters on %d Linear modules", len(replaced))
        if args.trainable_reference_energies:
            set_requires_grad_matching(model, [r"heads\.energy\.reference"], True)
        if args.unfreeze_regex:
            changed = set_requires_grad_matching(model, [args.unfreeze_regex], True)
            LOG.info("Unfroze %d parameters via --unfreeze-regex", changed)

    LOG.info(
        "Parameters: total=%d trainable=%d",
        count_parameters(model, trainable_only=False),
        count_parameters(model, trainable_only=True),
    )
    return model.to(device), atoms_adapter


def make_dataset(
    name: str,
    path: Path,
    atoms_adapter,
    *,
    include_stress: bool,
    augmentation: bool,
) -> AseSqliteDataset:
    graph_targets = ["energy", "stress"] if include_stress else ["energy"]
    target_config = property_definitions.instantiate_property_config(
        {"graph": graph_targets, "node": ["forces"]}
    )
    aug = [augmentations.rotate_randomly] if augmentation else []
    return AseSqliteDataset(
        name=name,
        path=path,
        atoms_adapter=atoms_adapter,
        target_config=target_config,
        augmentations=aug,
    )


def make_train_loader(args: argparse.Namespace, atoms_adapter, include_stress: bool) -> DataLoader:
    target = make_dataset(
        args.dataset,
        args.train_db,
        atoms_adapter,
        include_stress=include_stress,
        augmentation=not args.no_augmentation,
    )
    dataset = target
    sampler = None

    samples_per_epoch = args.num_steps * args.batch_size if args.num_steps else None
    if args.replay_db:
        replay = make_dataset(
            f"{args.dataset}-replay",
            args.replay_db,
            atoms_adapter,
            include_stress=include_stress,
            augmentation=not args.no_augmentation,
        )
        dataset = ConcatDataset([target, replay])
        replay_fraction = args.replay_ratio / (1.0 + args.replay_ratio)
        target_weight = (1.0 - replay_fraction) / max(len(target), 1)
        replay_weight = replay_fraction / max(len(replay), 1)
        weights = torch.cat(
            [
                torch.full((len(target),), target_weight, dtype=torch.double),
                torch.full((len(replay),), replay_weight, dtype=torch.double),
            ]
        )
        sampler = WeightedRandomSampler(
            weights,
            num_samples=samples_per_epoch or len(dataset),
            replacement=True,
        )
        LOG.info(
            "Replay enabled: target=%d replay=%d expected_replay_fraction=%.3f",
            len(target),
            len(replay),
            replay_fraction,
        )
    elif samples_per_epoch:
        sampler = RandomSampler(target, replacement=True, num_samples=samples_per_epoch)
    else:
        sampler = RandomSampler(target)

    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=False)
    return DataLoader(
        dataset,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
        collate_fn=atoms_adapter.batch,
        batch_sampler=batch_sampler,
        timeout=10 * 60 if args.num_workers > 0 else 0,
    )


def make_eval_loader(
    name: str,
    path: Path | None,
    args: argparse.Namespace,
    atoms_adapter,
    include_stress: bool,
) -> DataLoader | None:
    if not path:
        return None
    dataset = make_dataset(
        name,
        path,
        atoms_adapter,
        include_stress=include_stress,
        augmentation=False,
    )
    batch_sampler = BatchSampler(
        SequentialSampler(dataset),
        batch_size=args.eval_batch_size,
        drop_last=False,
    )
    return DataLoader(
        dataset,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
        collate_fn=atoms_adapter.batch,
        batch_sampler=batch_sampler,
        timeout=10 * 60 if args.num_workers > 0 else 0,
    )


def metric_to_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().mean().cpu())
    return float(value)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    *,
    device: torch.device,
    clip_grad: float | None,
    log_every: int,
    epoch: int,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    steps = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model.loss(batch)
        if torch.isnan(out.loss):
            raise ValueError("NaN loss encountered")
        out.loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(
                [param for param in model.parameters() if param.requires_grad],
                clip_grad,
            )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        steps += 1
        for key, value in out.log.items():
            totals[key] = totals.get(key, 0.0) + metric_to_float(value)
        if log_every > 0 and steps % log_every == 0:
            LOG.info(
                "epoch=%d step=%d loss=%.6g",
                epoch,
                steps,
                totals.get("loss", 0.0) / steps,
            )

    return {key: value / max(steps, 1) for key, value in totals.items()}


def evaluate(
    model: nn.Module,
    loader: DataLoader | None,
    *,
    device: torch.device,
    max_batches: int | None,
) -> dict[str, float]:
    if loader is None:
        return {}
    model.eval()
    totals: dict[str, float] = {}
    steps = 0
    with torch.enable_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = batch.to(device)
            out = model.loss(batch)
            steps += 1
            for key, value in out.log.items():
                totals[key] = totals.get(key, 0.0) + metric_to_float(value)
    return {key: value / max(steps, 1) for key, value in totals.items()}


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    args: argparse.Namespace,
    *,
    epoch: int,
    output_dir: Path,
    metrics: dict[str, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "metrics": metrics,
    }
    torch.save(payload, output_dir / f"checkpoint_epoch{epoch}.pt")
    if args.lora:
        trainable_state = {
            name: param.detach().cpu()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        torch.save(
            {
                "epoch": epoch,
                "base_model": args.base_model,
                "weights_path": args.weights_path,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "lora_target_regex": args.lora_target_regex,
                "lora_exclude_regex": args.lora_exclude_regex,
                "trainable_state_dict": trainable_state,
                "metrics": metrics,
            },
            output_dir / f"lora_trainable_epoch{epoch}.pt",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-db", required=True, type=Path)
    parser.add_argument("--valid-db", type=Path, default=None)
    parser.add_argument("--test-db", type=Path, default=None)
    parser.add_argument("--replay-db", type=Path, default=None)
    parser.add_argument("--replay-ratio", type=float, default=1.0)
    parser.add_argument("--dataset", default="cs-pb-i-neb")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-model", default="orb-v3-conservative-inf-omat")
    parser.add_argument("--weights-path", type=Path, default=None)
    parser.add_argument("--precision", default="float32-high")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=5.0e-7)
    parser.add_argument("--gradient-clip-val", type=float, default=0.5)
    parser.add_argument("--energy-loss-weight", type=float, default=40.0)
    parser.add_argument("--forces-loss-weight", type=float, default=100.0)
    parser.add_argument("--stress-loss-weight", type=float, default=0.0)
    parser.add_argument("--confidence-loss-weight", type=float, default=0.0)
    parser.add_argument("--equigrad-loss-weight", type=float, default=None)
    parser.add_argument("--trainable-reference-energies", action="store_true")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora-target-regex",
        default=r"(^model\.|^heads\.(energy|forces|stress)\.)",
    )
    parser.add_argument("--lora-exclude-regex", default=r"(reference|confidence)")
    parser.add_argument(
        "--unfreeze-regex",
        default=None,
        help="Optional parameter-name regex to unfreeze in addition to LoRA parameters.",
    )
    args = parser.parse_args()
    if args.replay_ratio <= 0:
        raise ValueError("--replay-ratio must be positive")
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    seed_everything(args.random_seed)
    set_torch_precision(args.precision)
    device = init_device(args.device_id)
    LOG.info("Using device: %s", device)
    LOG.info("Run name: %s", args.run_name)

    model, atoms_adapter = make_orb_model(args, device)
    include_stress = bool(getattr(model, "has_stress", False)) and args.stress_loss_weight > 0.0
    train_loader = make_train_loader(args, atoms_adapter, include_stress)
    valid_loader = make_eval_loader("valid", args.valid_db, args, atoms_adapter, include_stress)
    test_loader = make_eval_loader("test", args.test_db, args, atoms_adapter, include_stress)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters are enabled")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(args.max_epochs * len(train_loader), 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            indent=2,
        )
        + "\n"
    )

    for epoch in range(args.max_epochs):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device=device,
            clip_grad=args.gradient_clip_val,
            log_every=args.log_every,
            epoch=epoch,
        )
        valid_metrics = evaluate(
            model,
            valid_loader,
            device=device,
            max_batches=args.max_eval_batches,
        )
        metrics = {
            **{f"train/{key}": value for key, value in train_metrics.items()},
            **{f"valid/{key}": value for key, value in valid_metrics.items()},
        }
        LOG.info(
            "epoch=%d train_loss=%.6g valid_loss=%s",
            epoch,
            train_metrics.get("loss", float("nan")),
            f"{valid_metrics['loss']:.6g}" if "loss" in valid_metrics else "n/a",
        )
        if epoch % args.save_every == 0 or epoch == args.max_epochs - 1:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                args,
                epoch=epoch,
                output_dir=run_dir,
                metrics=metrics,
            )

    test_metrics = evaluate(
        model,
        test_loader,
        device=device,
        max_batches=args.max_eval_batches,
    )
    if test_metrics:
        (run_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2) + "\n")
        LOG.info("test_loss=%.6g", test_metrics.get("loss", float("nan")))


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)
    main()
