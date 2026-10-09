from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import zipfile
import os
import argparse
import json
import random
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from project.utils import config as cfg
from project.logging_utils import tee_output
from project.train import (
    cleanup_distributed,
    is_main_process,
    make_model,
    resolve_device,
    set_seed,
    setup_distributed,
)
from project.utils.spg_info import spg_wyckoff_degrees_of_freedom, wyckoff_label_to_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample from joint D3PM/DDPM Wyckoff diffusion.")
    cfg.add_common_args(parser)
    cfg.add_model_args(parser)
    cfg.add_test_args(parser)
    return parser


def load_model(args: argparse.Namespace, device: torch.device):
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    saved_args = checkpoint.get("args", {})
    merged = vars(args).copy()
    for key, value in saved_args.items():
        if key in merged and key not in {"checkpoint_path", "eval_split", "batch_size"}:
            merged[key] = value
    model_args = argparse.Namespace(**merged)
    model = make_model(model_args).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def _torch_load(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_spacegroup_masks(path: str | Path, device: torch.device) -> dict[int, dict[str, Any]]:
    template_path = Path(path)
    if not template_path.exists():
        raise FileNotFoundError(
            f"Space-group template file not found: {template_path}. "
            "Pass --spacegroup-template-path to the file you generated."
        )
    suffix = template_path.suffix.lower()
    if suffix == ".json":
        with open(template_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    elif suffix in {".npy", ".npz"}:
        loaded = np.load(template_path, allow_pickle=True)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            raw = {key: loaded[key] for key in loaded.files}
        elif isinstance(loaded, np.ndarray) and loaded.shape == ():
            raw = loaded.item()
        else:
            raw = loaded
    else:
        raw = _torch_load(template_path, device)
    if isinstance(raw, dict) and "templates" in raw:
        raw = raw["templates"]
    if isinstance(raw, list):
        raw = {idx: value for idx, value in enumerate(raw) if value is not None}
    masks = {int(key): value for key, value in raw.items()}
    for spg, group_masks in masks.items():
        if not isinstance(group_masks, dict):
            raise ValueError(f"Space-group {spg} mask must map Wyckoff letters to 3-vector masks.")
    return masks


def load_orbit_template_pool(root: str | Path) -> dict[int, list[tuple[tuple[str, ...], tuple[int, ...]]]]:
    pool: dict[int, list[tuple[tuple[str, ...], tuple[int, ...]]]] = {}
    for path in Path(root).glob("*.npy"):
        raw = np.load(path, allow_pickle=True)
        record = raw.item() if isinstance(raw, np.ndarray) and raw.shape == () else raw
        if not isinstance(record, dict):
            continue
        spg = int(record["spg_number"])
        letters = tuple(str(value) for value in record["wyckoff_letters"])
        dofs = tuple(int(value) for value in np.asarray(record["wyckoff_dofs"]).reshape(-1))
        if letters and len(letters) == len(dofs):
            pool.setdefault(spg, []).append((letters, dofs))
    if not pool:
        raise FileNotFoundError(f"No occupied-orbit templates found under {root}.")
    return pool


def make_template_batch(
    masks: dict[int, dict[str, Any]],
    template_pool: dict[int, list[tuple[tuple[str, ...], tuple[int, ...]]]],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, torch.Tensor | list[list[str]]]:
    keys = sorted(set(masks) & set(template_pool))
    requested_spgs = list(dict.fromkeys(int(spg) for spg in args.sample_space_groups))
    if args.sample_space_group > 0 and requested_spgs:
        raise ValueError("--sample-space-group and --sample-space-groups cannot be used together.")
    if requested_spgs:
        invalid_spgs = [spg for spg in requested_spgs if not 1 <= spg <= 230]
        if invalid_spgs:
            raise ValueError(f"Space-group numbers must be in 1..230, got {invalid_spgs}.")
        missing_spgs = [spg for spg in requested_spgs if spg not in keys]
        if missing_spgs:
            raise KeyError(f"Space groups are not in the template file: {missing_spgs}.")
        sampled_spgs = [random.choice(requested_spgs) for _ in range(args.sample_batch_size)]
    elif args.sample_space_group > 0:
        if args.sample_space_group not in keys:
            raise KeyError(f"space group {args.sample_space_group} has no mask or training orbit template.")
        sampled_spgs = [args.sample_space_group] * args.sample_batch_size
    else:
        sampled_spgs = [random.choice(keys) for _ in range(args.sample_batch_size)]

    batch_index = []
    continuous_masks = []
    letter_ids = []
    token_dofs = []
    sampled_letters: list[list[str]] = []

    for sample_index, spg in enumerate(sampled_spgs):
        letters, saved_dofs = random.choice(template_pool[spg])
        dof_map = spg_wyckoff_degrees_of_freedom[str(spg)]
        occurrences: dict[str, int] = {}
        sampled_letters.append(list(letters))
        for letter, saved_dof in zip(letters, saved_dofs):
            dof = int(dof_map[letter])
            if dof != int(saved_dof):
                raise ValueError(f"Template dof mismatch for space group {spg}, letter {letter}.")
            if letter not in masks[spg]:
                raise KeyError(f"No mask for space group {spg}, letter {letter}.")
            coord_mask = np.asarray(masks[spg][letter], dtype=np.float32).reshape(3)
            if int(np.rint(coord_mask.sum())) != dof:
                raise ValueError(f"Mask for space group {spg}, letter {letter} does not match dof={dof}.")
            slot = occurrences.get(letter, 0)
            if slot >= args.max_inf_occ:
                raise ValueError(f"Template has more than max_inf_occ={args.max_inf_occ} copies of {letter}.")
            occurrences[letter] = slot + 1
            batch_index.append(sample_index)
            continuous_masks.append(coord_mask)
            letter_ids.append(int(wyckoff_label_to_index[letter]))
            token_dofs.append(dof)

    return {
        "space_group": torch.as_tensor(sampled_spgs, dtype=torch.long, device=device),
        "batch_index": torch.as_tensor(batch_index, dtype=torch.long, device=device),
        "continuous_mask": torch.as_tensor(np.asarray(continuous_masks), dtype=torch.float32, device=device),
        "wyckoff_letter": torch.as_tensor(letter_ids, dtype=torch.long, device=device),
        "wyckoff_dof": torch.as_tensor(token_dofs, dtype=torch.long, device=device),
        "wyckoff_labels": sampled_letters,
    }


@torch.no_grad()
def sample_wyckoff_transformer(
    model,
    args: argparse.Namespace,
    device: torch.device,
    batch_ids: list[int] | None = None,
) -> dict[str, Any]:
    masks = load_spacegroup_masks(args.spacegroup_template_path, device)
    template_pool = load_orbit_template_pool(args.train_dir)
    chunks = []
    label_chunks = []
    if batch_ids is None:
        batch_ids = list(range(args.num_sample_batches))
    for i in batch_ids:
        template_batch = make_template_batch(masks, template_pool, args, device)
        sample = model.sample(template_batch, device=device)
        offset = len(label_chunks)
        cpu_sample = {key: value.cpu() for key, value in sample.items()}
        if "batch_index" in cpu_sample and cpu_sample["batch_index"].dim() == 1:
            cpu_sample["batch_index"] = cpu_sample["batch_index"] + offset
        chunks.append(cpu_sample)
        label_chunks.extend(template_batch["wyckoff_labels"])
        print(f"Sample: {i}/{args.num_sample_batches}")
    if not chunks:
        return {}
    tensor_keys = chunks[0].keys()
    merged = {key: torch.cat([chunk[key] for chunk in chunks], dim=0) for key in tensor_keys}
    merged["wyckoff_labels"] = label_chunks
    return merged


_SAMPLE_KEYS = {
    "lattice",
    "space_group",
}


_TOKEN_KEYS = {
    "discrete_labels",
    "continuous",
    "batch_index",
    "continuous_mask",
    "wyckoff_letter",
    "wyckoff_dof",
}


def _num_samples(samples: dict[str, Any]) -> int:
    if not samples:
        return 0
    return int(samples["space_group"].shape[0])


def merge_sample_shards(shards: list[dict[str, Any]]) -> dict[str, Any]:
    shards = [shard for shard in shards if shard]
    if not shards:
        return {}

    merged: dict[str, Any] = {}
    sample_offset = 0
    token_parts: dict[str, list[torch.Tensor]] = {key: [] for key in _TOKEN_KEYS}
    sample_parts: dict[str, list[torch.Tensor]] = {key: [] for key in _SAMPLE_KEYS}
    labels: list[list[str]] = []

    for shard in shards:
        shard_samples = _num_samples(shard)
        for key in _SAMPLE_KEYS:
            if key in shard:
                sample_parts[key].append(shard[key])
        for key in _TOKEN_KEYS:
            if key not in shard:
                continue
            value = shard[key].clone()
            if key == "batch_index":
                value = value + sample_offset
            token_parts[key].append(value)
        labels.extend(shard.get("wyckoff_labels", []))
        sample_offset += shard_samples

    for key, values in sample_parts.items():
        if values:
            merged[key] = torch.cat(values, dim=0)
    for key, values in token_parts.items():
        if values:
            merged[key] = torch.cat(values, dim=0)
    merged["wyckoff_labels"] = labels
    return merged

def lattice6_recover(lattice: torch.Tensor) -> torch.Tensor:
    values = lattice.detach().cpu().float()
    a, b, c = torch.exp(values[:3]).tolist()
    alpha, beta, gamma = 180*torch.arccos(values[3:].clamp(-0.999, 0.999))/torch.pi
    return torch.tensor([a, b, c, alpha, beta, gamma], dtype=torch.float32)

def wyckoff_sample_to_tables(
    samples: dict[str, Any],
    index: int,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    token_mask = samples["batch_index"] == index
    labels = samples["discrete_labels"][token_mask].detach().cpu().long()
    continuous = samples["continuous"][token_mask].detach().cpu().float()
    dofs = samples["wyckoff_dof"][token_mask].detach().cpu().long()
    letters = list(samples["wyckoff_labels"][index])
    if not (labels.numel() == continuous.shape[0] == dofs.numel() == len(letters)):
        raise ValueError("Generated packed orbit fields have inconsistent lengths.")

    occupied = (labels > cfg.VACANCY_CLASS) & (labels < cfg.NUM_DISCRETE_CLASSES)
    return (
        labels[occupied].numpy(),
        continuous[occupied].numpy().astype(np.float32, copy=False),
        [letter for letter, keep in zip(letters, occupied.tolist()) if keep],
        dofs[occupied].numpy(),
    )


def save_wyckoff_generated_cifs(samples: dict[str, Any], args: argparse.Namespace) -> None:
    from project.utils.dataset_utils import tables_to_crystal

    num_save = min(args.num_save_cifs, samples["space_group"].shape[0])
    cif_dir = Path(args.sample_dir) / "cif_no_CL_9760"
    cif_dir.mkdir(parents=True, exist_ok=True)
    file_list = os.listdir(cif_dir)
    for file_name in file_list:
        os.remove(cif_dir / file_name)
    total={}
    saved = 0
    mask_table = _torch_load(Path(args.spacegroup_template_path), torch.device("cpu"))
    for i in range(num_save):
        spg = int(samples["space_group"][i].item())
        discrete_tab, continuous_tab, wyckoff_letters, wyckoff_dofs = wyckoff_sample_to_tables(
            samples, i
        )
        if len(discrete_tab) == 0:
            continue
        lattice_matrix = lattice6_recover(samples["lattice"][i]).numpy()
        try:
            ret = tables_to_crystal(
                discrete_tab=discrete_tab,
                continuous_tab=continuous_tab,
                wyckoff_letters=wyckoff_letters,
                wyckoff_dofs=wyckoff_dofs,
                spg_number=spg,
                lattice_const=lattice_matrix,
                mask=mask_table,
            )
            if ret is None:
                continue
            structure, info=ret
            structure.to(filename=str(cif_dir / f"sample_{i:04d}_sg{spg}.cif"))
            total[f"sample_{i:04d}_sg{spg}.cif"]=info
            saved += 1
        except Exception as exc:
            print(f"skip CIF sample {i}: {exc}")
    with open(Path(args.sample_dir) / "total.json", "w+", encoding="utf-8") as f:
        json.dump(
            total,
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"saved CIFs: {saved}/{num_save} -> {cif_dir}")


def _resolve_log_file(args: argparse.Namespace) -> Path | None:
    if args.no_log_file or not args.log_file:
        return None
    log_path = Path(args.log_file)
    if not log_path.is_absolute():
        log_path = Path(args.output_dir) / log_path
    return log_path


def run(args: argparse.Namespace) -> None:
    distributed, local_rank, rank, world_size = setup_distributed()
    main_process = is_main_process(distributed, rank)
    set_seed(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}") if distributed and torch.cuda.is_available() else resolve_device(args.device)
    if main_process:
        print(f"log_file={args.log_file if not args.no_log_file else 'disabled'}")
        if args.sample_space_groups:
            print(f"sample_space_groups={list(dict.fromkeys(args.sample_space_groups))}")
        elif args.sample_space_group > 0:
            print(f"sample_space_group={args.sample_space_group}")
        else:
            print("sample_space_groups=all groups available in the template file")
        if distributed:
            print(
                f"distributed=True backend={dist.get_backend()} "
                f"world_size={world_size} per_gpu_sample_batch_size={args.sample_batch_size}"
            )

    model, checkpoint = load_model(args, device)
    if main_process:
        print(device)

    try:
        if args.num_sample_batches > 0:
            sample_dir = Path(args.sample_dir)
            sample_dir.mkdir(parents=True, exist_ok=True)
            local_batch_ids = [idx for idx in range(args.num_sample_batches) if idx % world_size == rank]
            local_samples = sample_wyckoff_transformer(model, args, device, batch_ids=local_batch_ids)
            shard_path = sample_dir / f"samples_{args.eval_split}_rank{rank}.pt"
            if shard_path.exists():
                shard_path.unlink()
            if local_samples:
                torch.save(
                    {
                        "samples": local_samples,
                        "checkpoint_epoch": checkpoint.get("epoch"),
                        "args": vars(args),
                        "rank": rank,
                        "world_size": world_size,
                        "batch_ids": local_batch_ids,
                    },
                    shard_path,
                )
                print(f"rank={rank} saved sample shard: {shard_path}")
            elif distributed:
                print(f"rank={rank} has no sample batches.")

            if distributed:
                dist.barrier()

            if main_process:
                if distributed:
                    shards = []
                    for shard_rank in range(world_size):
                        path = sample_dir / f"samples_{args.eval_split}_rank{shard_rank}.pt"
                        if path.exists():
                            shards.append(torch.load(path, map_location="cpu", weights_only=False)["samples"])
                    samples = merge_sample_shards(shards)
                else:
                    samples = local_samples

                if not samples:
                    print("No samples were generated.")
                    return

                print(
                    "samples:",
                    "discrete_labels",
                    tuple(samples["discrete_labels"].shape),
                    "continuous",
                    tuple(samples["continuous"].shape),
                    "lattice",
                    tuple(samples["lattice"].shape),
                    "num_structures",
                    _num_samples(samples),
                )
                output_path = sample_dir / f"samples_{args.eval_split}.pt"
                torch.save(
                    {
                        "samples": samples,
                        "checkpoint_epoch": checkpoint.get("epoch"),
                        "args": vars(args),
                    },
                    output_path,
                )
                print(f"saved merged samples: {output_path}")
                if args.save_cifs:
                    save_wyckoff_generated_cifs(samples, args)
                    with zipfile.ZipFile(Path(args.sample_dir) / "cifs.zip", "w") as zipObj:
                        for root, dirs, files in os.walk(Path(args.sample_dir) / "cif"):
                            for file in files:
                                zipObj.write(os.path.join(root, file))

            if distributed:
                dist.barrier()
    finally:
        cleanup_distributed(distributed)


def main() -> None:
    args = build_parser().parse_args()
    log_path = _resolve_log_file(args)
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
