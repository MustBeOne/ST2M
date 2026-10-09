from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import pyxtal
import os
import torch
import torch.distributed as dist
import socket
import subprocess
import argparse
from pathlib import Path
from project.utils.spg_info import spg_wyckoff


def extract_wyckoff_orbit_parameters(
    structure_raw,
    spg_true=None,
    symprec=1e-3,
    angle_tolerance=5,
):
    sga = SpacegroupAnalyzer(
        structure_raw,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )
    structure = sga.get_conventional_standard_structure()
    xtal = pyxtal.pyxtal()
    xtal.from_seed(structure)
    structure_xtal=xtal.to_pymatgen()
    spacegroup_symbol = xtal.group.symbol
    spacegroup_number = xtal.group.number
    info={'spg_number': spacegroup_number, 
          'spg_symbol': spacegroup_symbol,
          'lattice_system': xtal.group.lattice_type,
          'lattice_matrix': structure_xtal.lattice.matrix,
          'lattice_abc': structure_xtal.lattice.abc,
          'lattice_angles': structure_xtal.lattice.angles}
    if spg_true is not None:
        if spg_true!=spacegroup_number:
            return None
    
    results = {i:[] for i in spg_wyckoff[str(spacegroup_number)].keys()}


    for n,site in enumerate(xtal.atom_sites):        
        species=site.specie
        wyckoff_letter = site.wp.letter
        multiplicity = site.wp.multiplicity 
        wyckoff_label = f"{multiplicity}{wyckoff_letter}"     
        representative_coord= site.position           
        result = {
            "species": species,
            "wyckoff_letter": wyckoff_letter,
            "wyckoff_label": wyckoff_label,
            "multiplicity": multiplicity,
            "orbit_id": n,
            "representative_coord": representative_coord,
        }
        results[wyckoff_letter].append(result)
    return structure, results, spacegroup_number, info


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def setup_distributed() -> tuple[bool, int, int, int]:
    """Initialize torchrun or Slurm-launched DDP if requested."""

    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ.get("RANK", local_rank))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    elif "SLURM_PROCID" in os.environ and int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        os.environ.setdefault("LOCAL_RANK", str(local_rank))
        os.environ.setdefault("RANK", str(rank))
        os.environ.setdefault("WORLD_SIZE", str(world_size))
        os.environ.setdefault("MASTER_ADDR", _slurm_master_addr())
        os.environ.setdefault("MASTER_PORT", _default_master_port())
    else:
        return False, 0, 0, 1

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return True, local_rank, rank, world_size


def _slurm_master_addr() -> str:
    nodelist = os.environ.get("SLURM_NODELIST")
    if not nodelist:
        return "127.0.0.1"
    try:
        output = subprocess.check_output(
            ["scontrol", "show", "hostname", nodelist],
            text=True,
        )
        return output.splitlines()[0].strip()
    except Exception:
        return socket.gethostname()


def _default_master_port() -> str:
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id and job_id.isdigit():
        return str(10000 + int(job_id[-4:]))
    return "29500"


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def trainable_parameter_groups(
    model: torch.nn.Module,
) -> list[tuple[str, list[torch.nn.Parameter]]]:
    """Return explicit, disjoint optimizer groups for separate denoisers."""
    base_model = unwrap_model(model)
    if getattr(base_model, "denoiser_mode", "shared") != "separate":
        return [("shared", [parameter for parameter in base_model.parameters() if parameter.requires_grad])]

    groups = [
        (
            "discrete",
            [parameter for parameter in base_model.denoiser.discrete_model.parameters() if parameter.requires_grad],
        ),
        (
            "continuous",
            [parameter for parameter in base_model.denoiser.continuous_model.parameters() if parameter.requires_grad],
        ),
    ]
    grouped_ids = {id(parameter) for _, parameters in groups for parameter in parameters}
    trainable_ids = {id(parameter) for parameter in base_model.parameters() if parameter.requires_grad}
    if grouped_ids != trainable_ids:
        raise RuntimeError("Separate optimizer groups do not cover the trainable model parameters exactly.")
    return groups

def is_main_process(distributed: bool, rank: int) -> bool:
    return (not distributed) or rank == 0

def _to_device(value, device: torch.device):
    return None if value is None else value.to(device, non_blocking=True)

def forward_batch(model: torch.nn.Module, batch, device: torch.device):
    return model(
        batch.discrete.to(device, non_blocking=True),
        batch.continuous.to(device, non_blocking=True),
        batch.lattice.to(device, non_blocking=True),
        batch.lattice_mask.to(device, non_blocking=True),
        _to_device(batch.batch_index, device),
        _to_device(batch.discrete_loss_mask, device),
        _to_device(batch.continuous_mask, device),
        _to_device(batch.wyckoff_letter, device),
        _to_device(batch.wyckoff_multiplicity, device),
        _to_device(batch.space_group, device),
    )

def resolve_log_file(args: argparse.Namespace) -> Path | None:
    if args.no_log_file or not args.log_file:
        return None
    log_path = Path(args.log_file)
    if not log_path.is_absolute():
        log_path = Path(args.output_dir) / log_path
    return log_path


def resolve_resume_checkpoint(args: argparse.Namespace) -> Path | None:
    if args.resume_best and args.resume:
        raise ValueError("Use either --resume-best or --resume <checkpoint>, not both.")
    if args.resume_best:
        path = Path(args.checkpoint_dir) / "best.pt"
    elif args.resume:
        path = Path(args.resume)
    else:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    return path