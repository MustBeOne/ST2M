from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import json
from typing import Any

import torch

from project.model.model_utils import (
    _complete_lattice_from_space_group,
    _d3pm_p_sample_packed,
    _lattice_mask_from_space_group,
)
from project.generate import (
    build_parser as build_test_parser,
    lattice6_recover,
    load_model,
    load_orbit_template_pool,
    load_spacegroup_masks,
    make_template_batch,
    wyckoff_sample_to_tables,
)
from project.train import resolve_device, set_seed
from project.utils.dataset_utils import tables_to_crystal


PROJECT_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = build_test_parser()
    parser.description = "Generate one crystal and save five reverse-diffusion snapshots as CIF files."
    parser.add_argument(
        "--steps-output-dir",
        type=str,
        default=str(PROJECT_DIR / "temp"),
        help="Directory for the five CIF snapshots and their tensor data.",
    )
    return parser


def _snapshot(
    discrete: torch.Tensor,
    continuous: torch.Tensor,
    lattice: torch.Tensor,
    template: dict[str, Any],
) -> dict[str, Any]:
    """Build the same packed sample dictionary used by generate.py."""
    return {
        "discrete_labels": discrete.detach().cpu().clone(),
        "continuous": continuous.detach().cpu().clone(),
        "lattice": lattice.detach().cpu().clone(),
        "batch_index": template["batch_index"].detach().cpu().clone(),
        "continuous_mask": template["continuous_mask"].detach().cpu().clone(),
        "space_group": template["space_group"].detach().cpu().clone(),
        "wyckoff_letter": template["wyckoff_letter"].detach().cpu().clone(),
        "wyckoff_dof": template["wyckoff_dof"].detach().cpu().clone(),
        "wyckoff_labels": [list(template["wyckoff_labels"][0])],
    }


@torch.no_grad()
def generate_five_steps(model, template: dict[str, Any], device: torch.device) -> dict[int, dict[str, Any]]:
    """Generate one sample and retain five equally spaced x0 estimates."""
    batch_index = template["batch_index"].to(device=device, dtype=torch.long)
    continuous_mask = template["continuous_mask"].to(device=device, dtype=torch.float32).reshape(-1, 3)
    token_letter = template["wyckoff_letter"].to(device=device, dtype=torch.long)
    token_dof = template["wyckoff_dof"].to(device=device, dtype=torch.long)
    space_group = template["space_group"].to(device=device, dtype=torch.long)
    batch = int(space_group.shape[0])
    token_count = int(batch_index.numel())
    if batch != 1:
        raise ValueError(f"generate_steps requires exactly one structure, got {batch}.")
    if token_count == 0:
        raise ValueError("The sampled Wyckoff template contains no orbit tokens.")

    discrete_t = model.discrete_diffusion.transition.sample_prior(
        batch_size=1, num_rows=token_count, device=device
    ).reshape(-1)
    continuous_t = torch.randn(token_count, 3, device=device) * continuous_mask
    lattice_mask = _lattice_mask_from_space_group(space_group, model.lattice_dim)
    lattice_t = torch.randn(batch, model.lattice_dim, device=device) * lattice_mask

    total_steps = int(model.timesteps)
    save_at = {round(total_steps * fraction / 5) for fraction in range(1, 6)}
    snapshots: dict[int, dict[str, Any]] = {}

    for reverse_step in reversed(range(total_steps)):
        t = torch.full((batch,), reverse_step, dtype=torch.long, device=device)
        token_t = t[batch_index]
        discrete_logits, predicted_noise, predicted_lattice_noise = model.denoiser(
            discrete_t,
            continuous_t,
            lattice_t,
            t,
            batch_index=batch_index,
            letter=token_letter,
            space_group=space_group,
        )

        # These are the model's clean-structure (x0) estimates at this point.
        discrete_x0 = discrete_logits.argmax(dim=-1)
        continuous_x0 = model.continuous_diffusion.predict_x0_from_eps(
            continuous_t, token_t, predicted_noise
        ).clamp(-1.0, 1.0) * continuous_mask
        lattice_x0 = model.continuous_diffusion.predict_x0_from_eps(
            lattice_t, t, predicted_lattice_noise
        )
        lattice_x0 = _complete_lattice_from_space_group(lattice_x0, space_group)

        continuous_t = model.continuous_diffusion.p_sample(
            continuous_t, token_t, predicted_noise, mask=continuous_mask, clip_x0=True
        )
        lattice_t = model.continuous_diffusion.p_sample(
            lattice_t, t, predicted_lattice_noise, mask=lattice_mask, clip_x0=False
        )
        discrete_t = _d3pm_p_sample_packed(
            model.discrete_diffusion.transition, discrete_t, token_t, discrete_logits
        )
        continuous_t = continuous_t * continuous_mask

        progress = total_steps - reverse_step
        if progress in save_at:
            if progress == total_steps:
                final_lattice = _complete_lattice_from_space_group(lattice_t, space_group)
                snapshots[progress] = _snapshot(discrete_t, continuous_t, final_lattice, template)
            else:
                snapshots[progress] = _snapshot(discrete_x0, continuous_x0, lattice_x0, template)
            print(f"captured generation step {progress}/{total_steps}")

    if len(snapshots) != 5:
        raise RuntimeError(f"Expected five snapshots, captured {sorted(snapshots)}.")
    return snapshots


def save_snapshots_as_cifs(
    snapshots: dict[int, dict[str, Any]],
    output_dir: Path,
    mask_table: Any,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {}
    failures: list[str] = []

    for progress, sample in sorted(snapshots.items()):
        spg = int(sample["space_group"][0].item())
        discrete, continuous, letters, dofs = wyckoff_sample_to_tables(sample, 0)
        filename = f"step_{progress:04d}_sg{spg}.cif"
        if len(discrete) == 0:
            failures.append(f"{filename}: model predicted only vacancies")
            continue
        try:
            result = tables_to_crystal(
                discrete_tab=discrete,
                continuous_tab=continuous,
                wyckoff_letters=letters,
                wyckoff_dofs=dofs,
                spg_number=spg,
                lattice_const=lattice6_recover(sample["lattice"][0]).numpy(),
                mask=mask_table,
            )
            if result is None:
                failures.append(f"{filename}: tables_to_crystal returned None")
                continue
            structure, info = result
            structure.to(filename=str(output_dir / filename))
            metadata[filename] = info
            print(f"saved {output_dir / filename}")
        except Exception as exc:
            failures.append(f"{filename}: {exc}")

    with (output_dir / "total.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    if failures:
        with (output_dir / "failed_steps.txt").open("w", encoding="utf-8") as handle:
            handle.write("\n".join(failures) + "\n")
        print("Some intermediate estimates could not be converted to CIF:")
        for failure in failures:
            print(f"  {failure}")
    print(f"saved CIFs: {len(metadata)}/5 -> {output_dir}")


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    model, checkpoint = load_model(args, device)

    # This script deliberately generates one structure, irrespective of generate.py's batch default.
    args.sample_batch_size = 1
    masks = load_spacegroup_masks(args.spacegroup_template_path, device)
    template_pool = load_orbit_template_pool(args.train_dir)
    template = make_template_batch(masks, template_pool, args, device)
    snapshots = generate_five_steps(model, template, device)

    output_dir = Path(args.steps_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "snapshots": snapshots,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "args": vars(args),
        },
        output_dir / "generation_steps.pt",
    )
    mask_table = torch.load(args.spacegroup_template_path, map_location="cpu", weights_only=False)
    save_snapshots_as_cifs(snapshots, output_dir, mask_table)


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()