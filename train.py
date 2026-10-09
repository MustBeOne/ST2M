from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import json
import os
import random
from time import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from project.utils import config as cfg
from project.utils.utils import *
from project.model.dataset import ProcessedCrystalTableDataset, collate_crystal_tables
from project.model.ST2M import ST2MDiffusion
from project.logging_utils import tee_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train joint D3PM/DDPM crystal diffusion with U-NET denoisers.")
    cfg.add_common_args(parser)
    cfg.add_model_args(parser)
    cfg.add_train_args(parser)
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(args: argparse.Namespace) -> torch.nn.Module:
    return ST2MDiffusion(
        num_classes=args.num_discrete_classes,
        max_sites=args.max_wyckoff_sites,
        max_occ=args.max_inf_occ,
        lattice_dim=args.lattice_dim,
        timesteps=args.timesteps,
        discrete_schedule=args.discrete_schedule,
        continuous_schedule=args.continuous_schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        d_model=args.transformer_d_model,
        nhead=args.transformer_nhead,
        num_layers=args.transformer_layers,
        dim_feedforward=args.transformer_ff_dim,
        dropout=args.transformer_dropout,
        denoiser_mode=args.transformer_denoiser_mode,
        discrete_loss_weight=args.discrete_loss_weight,
        continuous_loss_weight=args.continuous_loss_weight,
        lattice_loss_weight=args.lattice_loss_weight,
        use_count_losses=args.use_count_losses,
        lambda_contrastive_con=args.lambda_contrastive_con,
        lambda_contrastive_dis=args.lambda_contrastive_dis,
        contrastive_margin=args.contrastive_margin,
        contrastive_use_x0=args.contrastive_use_x0,
    )


def print_training_summary(
    args: argparse.Namespace,
    model: torch.nn.Module,
    device: torch.device,
    distributed: bool,
    world_size: int,
    train_size: int,
    val_size: int,
    start_epoch: int,
    best_val: float,
) -> None:
    base_model = unwrap_model(model)
    total_params = sum(param.numel() for param in base_model.parameters())
    trainable_params = sum(param.numel() for param in base_model.parameters() if param.requires_grad)

    print("==== training configuration ====")
    print(
        f"model={args.model_type} device={device} distributed={distributed} "
        f"world_size={world_size}"
    )
    print(
        f"epochs={args.epochs} start_batch_size={args.batch_size} "
        f"effective_batch_size={args.batch_size * max(world_size, 1)} "
        f"train_samples={train_size} val_samples={val_size} "
        f"start_epoch={start_epoch} resume={args.resume or 'none'} best_val={best_val:.6g}"
    )
    print(
        f"optimizer=AdamW lr={args.lr:g} weight_decay={args.weight_decay:g} "
        f"grad_clip={args.grad_clip:g} seed={args.seed}"
    )
    print(
        f"timesteps={args.timesteps} discrete_schedule={args.discrete_schedule} "
        f"continuous_schedule={args.continuous_schedule} "
        f"beta_start={args.beta_start:g} beta_end={args.beta_end:g}"
    )
    print(
        "loss_weights="
        f"discrete:{args.discrete_loss_weight:g}, "
        f"continuous:{args.continuous_loss_weight:g}, "
        f"lattice:{args.lattice_loss_weight:g}, "
        f"contrastive_continuous:{args.lambda_contrastive_con:g}, "
        f"contrastive_discrete:{args.lambda_contrastive_dis:g}"
    )
    print(
        f"contrastive_margin={args.contrastive_margin:g} "
        f"contrastive_continuous_target={'x0' if args.contrastive_use_x0 else 'noise'} "
        "negative_conditions=permuted_within_space_group_and_equal_orbit_count"
    )
    print(
        f"count_atom_margin={args.count_loss_atom_margin:g} "
        f"count_smooth_tau={args.count_loss_smooth_tau:g}"
    )
    print(
        f"transformer=d_model:{args.transformer_d_model}, nhead:{args.transformer_nhead}, "
        f"layers:{args.transformer_layers}, ff_dim:{args.transformer_ff_dim}, "
        f"dropout:{args.transformer_dropout:g}, denoiser_mode:{args.transformer_denoiser_mode}"
    )
    if args.transformer_denoiser_mode == "separate":
        discrete_params = sum(param.numel() for param in base_model.denoiser.discrete_model.parameters())
        continuous_params = sum(param.numel() for param in base_model.denoiser.continuous_model.parameters())
        print(
            f"separate_objectives=L_D(diff_discrete+count+lambda_dis*CL_dis), "
            f"L_C(diff_continuous+lattice+lambda_con*CL_con) "
            f"branch_parameters=discrete:{discrete_params:,}, continuous:{continuous_params:,}"
        )
    print(f"parameters=total:{total_params:,} trainable:{trainable_params:,}")
    print("================================")


def make_loader(
    root: str,
    args: argparse.Namespace,
    batch_size: int,
    shuffle: bool,
    max_samples: int,
    distributed: bool = False,
) -> tuple[DataLoader, DistributedSampler | None]:
    dataset = ProcessedCrystalTableDataset(
        root=root,
        lattice_dim=args.lattice_dim,
        num_classes=args.num_discrete_classes,
        max_occ=args.max_inf_occ,
        max_samples=max_samples,
    )
    sampler = DistributedSampler(dataset, shuffle=shuffle) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and torch.cuda.is_available(),
        collate_fn=collate_crystal_tables,
    )
    return loader, sampler

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "continuous_objective": 0.0,
        "discrete_objective": 0.0,
        "discrete_loss": 0.0,
        "continuous_loss": 0.0,
        "lattice_loss": 0.0,
        "contrastive_con_loss": 0.0,
        "contrastive_dis_loss": 0.0,
    }
    count = 0
    for batch in loader:
        out = forward_batch(model, batch, device)
        batch_size = batch.lattice.shape[0]
        totals["loss"] += out.loss.item() * batch_size
        totals["continuous_objective"] += out.continuous_objective.item() * batch_size
        totals["discrete_objective"] += out.discrete_objective.item() * batch_size
        totals["discrete_loss"] += out.discrete_loss.item() * batch_size
        totals["continuous_loss"] += out.continuous_loss.item() * batch_size
        totals["lattice_loss"] += out.lattice_loss.item() * batch_size
        totals["contrastive_con_loss"] += out.contrastive_con_loss.item() * batch_size
        totals["contrastive_dis_loss"] += out.contrastive_dis_loss.item() * batch_size
        count += batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    best_val: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "best_val": best_val,
        },
        path,
    )


def run(args: argparse.Namespace) -> None:
    distributed, local_rank, rank, world_size = setup_distributed()
    main_process = is_main_process(distributed, rank)
    set_seed(args.seed)
    device = torch.device(f"cuda:{local_rank}") if distributed and torch.cuda.is_available() else resolve_device(args.device)

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint = resolve_resume_checkpoint(args)
    if resume_checkpoint is not None:
        args.resume = str(resume_checkpoint)
    if main_process:
        with open(output_dir / "args.json", "w", encoding="utf-8") as handle:
            json.dump(vars(args), handle, indent=2)
        print(f"log_file={args.log_file if not args.no_log_file else 'disabled'}")
        if distributed:
            print(
                f"distributed=True backend={dist.get_backend()} "
                f"world_size={world_size} per_gpu_batch_size={args.batch_size}"
            )

    train_loader, train_sampler = make_loader(
        args.train_dir,
        args,
        batch_size=args.batch_size,
        shuffle=True,
        max_samples=args.max_train_samples,
        distributed=distributed,
    )
    val_loader, _ = make_loader(
        args.val_dir,
        args,
        batch_size=args.batch_size,
        shuffle=False,
        max_samples=args.max_val_samples,
        distributed=False,
    )

    model = make_model(args).to(device)
    start_epoch = 0
    best_val = float("inf")
    checkpoint = None
    if resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val = float(checkpoint.get("best_val", best_val))
    if main_process:
        print_training_summary(
            args,
            model,
            device,
            distributed,
            world_size,
            len(train_loader.dataset),
            len(val_loader.dataset),
            start_epoch,
            best_val,
        )
    if distributed:
        if torch.cuda.is_available():
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        else:
            model = DDP(model)

    parameter_groups = trainable_parameter_groups(model)
    optimizer = torch.optim.AdamW(
        [{"params": parameters, "name": name} for name, parameters in parameter_groups],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = 0
    try:
        for epoch in range(start_epoch, args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            running = 0.0
            tic = time()
            for step, batch in enumerate(train_loader):
                out = forward_batch(model, batch, device)

                optimizer.zero_grad(set_to_none=True)
                out.loss.backward()
                if args.grad_clip > 0:
                    for _, parameters in parameter_groups:
                        torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
                optimizer.step()

                running += out.loss.item()
                global_step += 1
                if main_process and args.log_every > 0 and global_step % args.log_every == 0:
                    avg = running / max(step + 1, 1)
                    print(
                        f"epoch={epoch:04d} step={step:05d} global={global_step:07d} "
                        f"loss={out.loss.item():.5f} avg={avg:.5f} "
                        f"L_C={out.continuous_objective.item():.5f} "
                        f"L_D={out.discrete_objective.item():.5f} "
                        f"d3pm={out.discrete_loss.item():.5f} "
                        f"ddpm={out.continuous_loss.item():.5f} "
                        f"lat={out.lattice_loss.item():.5f} "
                        f"cl_con={out.contrastive_con_loss.item():.5f} "
                        f"cl_dis={out.contrastive_dis_loss.item():.5f}"
                    )

            if main_process:
                print(
                    f"epoch={epoch:04d} train_loss={running / max(len(train_loader), 1):.5f} "
                    f"time={time() - tic:.1f}s"
                )

            if args.val_every > 0 and (epoch + 1) % args.val_every == 0:
                if main_process:
                    metrics = evaluate(unwrap_model(model), val_loader, device)
                    print(
                        f"epoch={epoch:04d} val_loss={metrics['loss']:.5f} "
                        f"val_L_C={metrics['continuous_objective']:.5f} "
                        f"val_L_D={metrics['discrete_objective']:.5f} "
                        f"val_d3pm={metrics['discrete_loss']:.5f} "
                        f"val_ddpm={metrics['continuous_loss']:.5f} "
                        f"val_lat={metrics['lattice_loss']:.5f} "
                        f"cl_con={metrics["contrastive_con_loss"]:.5f}"
                        f"cl_dis={metrics["contrastive_dis_loss"]:.5f}"
                    )
                    if metrics["loss"] < best_val:
                        best_val = metrics["loss"]
                        save_checkpoint(
                            checkpoint_dir / "best.pt",
                            unwrap_model(model),
                            optimizer,
                            args,
                            epoch,
                            best_val,
                        )
                        print(f"saved best checkpoint: {checkpoint_dir / 'best.pt'}")
                if distributed:
                    dist.barrier()

            if main_process and args.save_every > 0 and (epoch + 1) % args.save_every == 0:
                save_checkpoint(
                    checkpoint_dir / f"epoch_{epoch:04d}.pt",
                    unwrap_model(model),
                    optimizer,
                    args,
                    epoch,
                    best_val,
                )

            if main_process:
                save_checkpoint(
                    checkpoint_dir / "latest.pt",
                    unwrap_model(model),
                    optimizer,
                    args,
                    epoch,
                    best_val,
                )
            if distributed:
                dist.barrier()
    finally:
        cleanup_distributed(distributed)


def main() -> None:
    args = build_parser().parse_args()
    log_path = resolve_log_file(args)
    env_rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    if log_path is None or env_rank != 0:
        run(args)
        return
    args.log_file = str(log_path)
    with tee_output(log_path) as path:
        print(f"logging to: {path}")
        run(args)


if __name__ == "__main__":
    main()
