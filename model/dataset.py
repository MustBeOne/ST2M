from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


from project.utils import config as cfg

from project.model.model_utils import (
    _load_npy_dict,
    _torch_load,
    _prepare_lattice,
    _prepare_lattice_mask,
    _prepare_packed_tensors,
)


@dataclass
class CrystalTableBatch:
    discrete: torch.Tensor
    continuous: torch.Tensor
    lattice: torch.Tensor
    lattice_mask: torch.Tensor
    space_group: torch.Tensor | None
    paths: list[str]
    batch_index: torch.Tensor | None = None
    continuous_mask: torch.Tensor | None = None
    discrete_loss_mask: torch.Tensor | None = None
    wyckoff_letter: torch.Tensor | None = None
    wyckoff_multiplicity: torch.Tensor | None = None


class ProcessedCrystalTableDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        lattice_dim: int = cfg.LATTICE_DIM,
        num_classes: int = cfg.NUM_DISCRETE_CLASSES,
        max_occ: int = cfg.DEFAULT_MAX_INF_OCC,
        max_samples: int = 0,
        spacegroup_template_path: str | Path = cfg.SPACEGROUP_TEMPLATE_PATH,
    ) -> None:
        self.root = Path(root)
        self.lattice_dim = int(lattice_dim)
        self.num_classes = int(num_classes)
        self.max_occ = int(max_occ)
        self.mask_table = _torch_load(spacegroup_template_path)
        self.paths = sorted(str(path) for path in self.root.glob("*.npy"))
        if max_samples and max_samples > 0:
            self.paths = self.paths[: int(max_samples)]
        if not self.paths:
            raise FileNotFoundError(f"No .npy files found under {self.root}.")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[index]
        record = _load_npy_dict(path)
        spg = record.get("spg_number", record.get("space_group", None))
        if isinstance(spg, np.ndarray):
            spg = int(np.asarray(spg).reshape(-1)[0])
        elif spg is not None:
            spg = int(spg)
        if spg is None:
            raise ValueError(f"Processed record has no spg_number/space_group: {path}")
        record["spg_number"] = spg

        packed = _prepare_packed_tensors(record, self.num_classes, self.max_occ, self.mask_table)
        lattice = _prepare_lattice(record, self.lattice_dim)
        lattice_mask = _prepare_lattice_mask(record, self.lattice_dim, spg)

        return {
            "lattice": torch.from_numpy(lattice),
            "lattice_mask": torch.from_numpy(lattice_mask),
            "space_group": spg,
            "path": path,
            **{key: torch.from_numpy(value) for key, value in packed.items()},
        }


def collate_crystal_tables(items: list[dict[str, Any]]) -> CrystalTableBatch:
    if not items:
        raise ValueError("Cannot collate an empty batch.")

    num_classes = items[0]["discrete"].shape[-1]
    slot_counts = [item["discrete"].shape[0] for item in items]
    total_slots = int(sum(slot_counts))

    discrete = torch.cat([item["discrete"].float() for item in items], dim=0)
    if discrete.shape != (total_slots, num_classes):
        raise ValueError(f"Packed discrete tensor has unexpected shape {tuple(discrete.shape)}.")
    continuous = torch.cat([item["continuous"].float() for item in items], dim=0)
    batch_index = torch.cat(
        [torch.full((count,), i, dtype=torch.long) for i, count in enumerate(slot_counts)],
        dim=0,
    )
    lattice_tensor = torch.stack([item["lattice"].float() for item in items], dim=0)
    lattice_mask_tensor = torch.stack([item["lattice_mask"].float() for item in items], dim=0)
    spg_tensor = torch.tensor([item["space_group"] for item in items], dtype=torch.long)

    return CrystalTableBatch(
        discrete=discrete,
        continuous=continuous,
        lattice=lattice_tensor,
        lattice_mask=lattice_mask_tensor,
        space_group=spg_tensor,
        paths=[item["path"] for item in items],
        batch_index=batch_index,
        continuous_mask=torch.cat([item["continuous_mask"].float() for item in items], dim=0),
        discrete_loss_mask=torch.cat([item["discrete_loss_mask"].float() for item in items], dim=0),
        wyckoff_letter=torch.cat([item["wyckoff_letter"].long() for item in items], dim=0),
        wyckoff_multiplicity=torch.cat([item["wyckoff_multiplicity"].long() for item in items], dim=0),
    )
