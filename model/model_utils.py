"""Shared dataset conversion, tensor, lattice and packed-sampling helpers.

Dataset classes remain in dataset.py, diffusion processes in diffusion.py,
and model architecture classes in ST2M.py.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from project.utils.spg_info import (
    spg_wyckoff_degrees_of_freedom,
    spg_wyckoff_multiplicities,
    wyckoff_label_to_index,
)

if TYPE_CHECKING:
    from project.model.diffusion import D3PMTransition

def _load_npy_dict(path: str | Path) -> dict[str, Any]:
    raw = np.load(path, allow_pickle=True)
    if isinstance(raw, np.lib.npyio.NpzFile):
        return {key: raw[key] for key in raw.files}
    if isinstance(raw, np.ndarray) and raw.shape == ():
        item = raw.item()
        if isinstance(item, dict):
            return item
    if isinstance(raw, dict):
        return raw
    raise ValueError(f"Unsupported processed table format in {path}.")


def _torch_load(path: str | Path) -> Any:
    template_path = Path(path)
    if not template_path.exists():
        return None
    try:
        return torch.load(template_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(template_path, map_location="cpu")


def _to_numpy(value: Any, dtype: np.dtype | type | None = None) -> np.ndarray:
    arr = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    return arr.astype(dtype, copy=False) if dtype is not None else arr


def _lookup_orbit_mask(mask_table: Any, spg: int, letter: str) -> np.ndarray | None:
    if not isinstance(mask_table, dict):
        return None
    group_masks = mask_table.get(str(spg), mask_table.get(int(spg)))
    if not isinstance(group_masks, dict) or letter not in group_masks:
        return None
    return _to_numpy(group_masks[letter], np.float32).reshape(3)


def _labels_to_one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    labels = np.clip(labels, 0, num_classes - 1)
    one_hot = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    one_hot[np.arange(labels.shape[0]), labels] = 1.0
    return one_hot


def _prepare_lattice(record: dict[str, Any], lattice_dim: int) -> np.ndarray:
    if lattice_dim == 6 and "lattice_abc" in record and "lattice_angles" in record:
        abc = np.asarray(record["lattice_abc"], dtype=np.float32).reshape(-1)
        angles = np.asarray(record["lattice_angles"], dtype=np.float32).reshape(-1)
        if abc.size < 3 or angles.size < 3:
            raise ValueError("lattice_abc and lattice_angles must each contain 3 values.")
        cos_angles = np.cos(np.deg2rad(angles[:3])).astype(np.float32)
        return np.concatenate([np.log(np.clip(abc[:3], 1e-8, None)), cos_angles]).astype(np.float32)

    values = record["lattice_matrix"]
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if lattice_dim == 6:
        matrix = arr.reshape(3, 3)
        lengths = np.linalg.norm(matrix, axis=1).clip(1e-8, None)
        cos_alpha = np.dot(matrix[1], matrix[2]) / (lengths[1] * lengths[2])
        cos_beta = np.dot(matrix[0], matrix[2]) / (lengths[0] * lengths[2])
        cos_gamma = np.dot(matrix[0], matrix[1]) / (lengths[0] * lengths[1])
        return np.array(
            [np.log(lengths[0]), np.log(lengths[1]), np.log(lengths[2]), cos_alpha, cos_beta, cos_gamma],
            dtype=np.float32,
        )
    if arr.size != lattice_dim:
        raise ValueError(f"lattice_matrix must flatten to length {lattice_dim}, got {arr.size}.")
    return arr.astype(np.float32, copy=False)


_LATTICE_SYSTEM_MASKS = {
    "triclinic": (1, 1, 1, 1, 1, 1),
    "monoclinic": (1, 1, 1, 0, 1, 0),
    "orthorhombic": (1, 1, 1, 0, 0, 0),
    "tetragonal": (1, 0, 1, 0, 0, 0),
    "trigonal": (1, 0, 1, 0, 0, 0),
    "hexagonal": (1, 0, 1, 0, 0, 0),
    "cubic": (1, 0, 0, 0, 0, 0),
}


def _lattice_system_from_space_group(spg: int | None) -> str | None:
    if spg is None:
        return None
    spg = int(spg)
    if 1 <= spg <= 2:
        return "triclinic"
    if 3 <= spg <= 15:
        return "monoclinic"
    if 16 <= spg <= 74:
        return "orthorhombic"
    if 75 <= spg <= 142:
        return "tetragonal"
    if 143 <= spg <= 167:
        return "trigonal"
    if 168 <= spg <= 194:
        return "hexagonal"
    if 195 <= spg <= 230:
        return "cubic"
    return None


def _normalize_lattice_system(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "rhombohedral": "trigonal",
        "orthorhombic_p": "orthorhombic",
        "monoclinic_p": "monoclinic",
    }
    return aliases.get(text, text)


def _prepare_lattice_mask(record: dict[str, Any], lattice_dim: int, spg: int | None) -> np.ndarray:
    if lattice_dim != 6:
        return np.ones(lattice_dim, dtype=np.float32)
    lattice_system = _normalize_lattice_system(record.get("lattice_system"))
    if lattice_system not in _LATTICE_SYSTEM_MASKS:
        lattice_system = _lattice_system_from_space_group(spg)
    mask = _LATTICE_SYSTEM_MASKS.get(str(lattice_system), _LATTICE_SYSTEM_MASKS["triclinic"])
    return np.asarray(mask, dtype=np.float32)


def _prepare_packed_tensors(
    record: dict[str, Any],
    num_classes: int,
    max_occ: int,
    mask_table: Any,
) -> dict[str, np.ndarray]:
    spg = int(record["spg_number"])
    dof_map = spg_wyckoff_degrees_of_freedom[str(spg)]
    mult_map = spg_wyckoff_multiplicities[str(spg)]
    discrete_labels = _to_numpy(record["discrete_tab"], np.int64).reshape(-1)
    continuous = _to_numpy(record["continuous_tab"], np.float32).reshape(-1, 3)
    orbit_letters = [str(value) for value in record["wyckoff_letters"]]
    saved_dofs = _to_numpy(record["wyckoff_dofs"], np.int64).reshape(-1)
    orbit_count = len(orbit_letters)
    if not (len(discrete_labels) == len(continuous) == len(saved_dofs) == orbit_count):
        raise ValueError("Orbit tables, letters, and dofs must have the same first dimension.")
    if int(record.get("orbit_count", orbit_count)) != orbit_count:
        raise ValueError("orbit_count does not match the packed orbit tables.")
    if np.any((discrete_labels < 1) | (discrete_labels >= num_classes)):
        raise ValueError("Packed occupied-orbit labels must be non-vacancy element classes.")

    masks = np.zeros((orbit_count, 3), dtype=np.float32)
    letter_ids = np.zeros(orbit_count, dtype=np.int64)
    multiplicities = np.zeros(orbit_count, dtype=np.int64)
    occurrence_counts: dict[str, int] = {}
    fixed_letters: set[str] = set()

    for index, (letter, saved_dof) in enumerate(zip(orbit_letters, saved_dofs.tolist())):
        if letter not in dof_map:
            raise ValueError(f"Wyckoff letter {letter!r} is invalid for space group {spg}.")
        dof = int(dof_map[letter])
        if int(saved_dof) != dof:
            raise ValueError(f"Saved dof for {letter} in space group {spg} is {saved_dof}, expected {dof}.")
        if dof == 0 and letter in fixed_letters:
            raise ValueError(f"Fixed Wyckoff position {letter} occurs more than once in space group {spg}.")
        if dof == 0:
            fixed_letters.add(letter)

        mask = _lookup_orbit_mask(mask_table, spg, letter)
        if mask is None:
            raise KeyError(f"No continuous mask for space group {spg}, Wyckoff letter {letter}.")
        if int(np.rint(mask.sum())) != dof:
            raise ValueError(f"Mask {mask.tolist()} for space group {spg}, letter {letter} does not match dof={dof}.")

        slot = occurrence_counts.get(letter, 0)
        if slot >= max_occ:
            raise ValueError(f"Wyckoff letter {letter} has more than max_occ={max_occ} occupied orbits.")
        occurrence_counts[letter] = slot + 1
        masks[index] = mask
        letter_ids[index] = int(wyckoff_label_to_index[letter])
        multiplicities[index] = int(mult_map[letter])

    ones = np.ones(orbit_count, dtype=np.float32)
    return {
        "discrete": _labels_to_one_hot(discrete_labels, num_classes),
        "continuous": continuous,
        "continuous_mask": masks,
        "discrete_loss_mask": ones.copy(),
        "wyckoff_letter": letter_ids,
        "wyckoff_multiplicity": multiplicities,
    }


def _labels(discrete: torch.Tensor) -> torch.Tensor:
    if discrete.dim() == 1:
        return discrete.long()
    if discrete.dim() == 2:
        return discrete.argmax(dim=-1).long()
    if discrete.dim() == 3:
        if discrete.dtype.is_floating_point:
            return discrete.argmax(dim=-1).long()
        return discrete.long()
    if discrete.dim() == 5:
        return discrete.argmax(dim=-1).long()
    if discrete.dim() == 4:
        return discrete.argmax(dim=-1).long()
    raise ValueError(f"discrete must be [B,L,K] or [B,L,K,C], got {tuple(discrete.shape)}.")


def _num_batches(batch_index: torch.Tensor) -> int:
    if batch_index.numel() == 0:
        return 0
    return int(batch_index.max().item()) + 1


def _sum_by_batch(values: torch.Tensor, batch_index: torch.Tensor, batch_size: int | None = None) -> torch.Tensor:
    if batch_size is None:
        batch_size = _num_batches(batch_index)
    out = values.new_zeros(batch_size)
    if values.numel() > 0:
        out.index_add_(0, batch_index.long(), values)
    return out


def _lattice_mask_from_space_group(space_group: torch.Tensor, lattice_dim: int) -> torch.Tensor:
    if lattice_dim != 6:
        return torch.ones(space_group.shape[0], lattice_dim, dtype=torch.float32, device=space_group.device)
    sg = space_group.long()
    mask = torch.ones(sg.shape[0], 6, dtype=torch.float32, device=sg.device)
    mask[(3 <= sg) & (sg <= 15)] = torch.tensor([1, 1, 1, 0, 1, 0], dtype=torch.float32, device=sg.device)
    mask[(16 <= sg) & (sg <= 74)] = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.float32, device=sg.device)
    mask[(75 <= sg) & (sg <= 142)] = torch.tensor([1, 0, 1, 0, 0, 0], dtype=torch.float32, device=sg.device)
    mask[(143 <= sg) & (sg <= 167)] = torch.tensor([1, 0, 1, 0, 0, 0], dtype=torch.float32, device=sg.device)
    mask[(168 <= sg) & (sg <= 194)] = torch.tensor([1, 0, 1, 0, 0, 0], dtype=torch.float32, device=sg.device)
    mask[(195 <= sg) & (sg <= 230)] = torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.float32, device=sg.device)
    return mask


def _complete_lattice_from_space_group(lattice: torch.Tensor, space_group: torch.Tensor) -> torch.Tensor:
    if lattice.shape[-1] != 6:
        return lattice
    out = lattice.clone()
    sg = space_group.long().to(device=out.device)

    monoclinic = (3 <= sg) & (sg <= 15)
    out[monoclinic, 3] = 0.0
    out[monoclinic, 5] = 0.0

    orthorhombic = (16 <= sg) & (sg <= 74)
    out[orthorhombic, 3:6] = 0.0

    tetragonal = (75 <= sg) & (sg <= 142)
    out[tetragonal, 1] = out[tetragonal, 0]
    out[tetragonal, 3:6] = 0.0

    trigonal_hexagonal = ((143 <= sg) & (sg <= 167)) | ((168 <= sg) & (sg <= 194))
    out[trigonal_hexagonal, 1] = out[trigonal_hexagonal, 0]
    out[trigonal_hexagonal, 3] = 0.0
    out[trigonal_hexagonal, 4] = 0.0
    out[trigonal_hexagonal, 5] = -0.5

    cubic = (195 <= sg) & (sg <= 230)
    out[cubic, 1] = out[cubic, 0]
    out[cubic, 2] = out[cubic, 0]
    out[cubic, 3:6] = 0.0
    return out


def _sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def _smooth_excess(value: torch.Tensor, limit: torch.Tensor, tau: float) -> torch.Tensor:
    tau = max(float(tau), 1e-6)
    centered = tau * F.softplus((value - limit) / tau) - tau * math.log(2.0)
    return F.relu(centered)


def _d3pm_q_sample_packed(
    transition: D3PMTransition,
    labels: torch.Tensor,
    token_t: torch.Tensor,
) -> torch.Tensor:
    labels_onehot = F.one_hot(labels.long(), transition.num_classes).float()
    qbar = transition.qt_bar.to(labels.device)[token_t.long()]
    probs = torch.einsum("nc,ncd->nd", labels_onehot, qbar)
    return sample_categorical(probs)


def _packed_per_sample_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
    batch_index: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    weights = weights.to(device=values.device, dtype=values.dtype).reshape(-1)
    numerators = _sum_by_batch(values * weights, batch_index, batch_size)
    denominators = _sum_by_batch(weights, batch_index, batch_size)
    return numerators / denominators.clamp_min(1.0)


def _d3pm_vb_loss_packed(
    transition: D3PMTransition,
    x0: torch.Tensor,
    x_t: torch.Tensor,
    token_t: torch.Tensor,
    x0_logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    num_classes = transition.num_classes
    x0_log_probs = F.log_softmax(x0_logits, dim=-1)
    x0_probs = x0_log_probs.exp()

    ce = F.nll_loss(
        x0_log_probs.reshape(-1, num_classes),
        x0.long().reshape(-1),
        reduction="none",
    ).reshape_as(x0.float())

    device = x_t.device
    token_t = token_t.long().to(device)
    x_t_onehot = F.one_hot(x_t.long(), num_classes).float()
    qt = transition.qt.to(device)[token_t]
    qbar_t = transition.qt_bar.to(device)[token_t]
    eye = transition.identity.to(device).expand(token_t.shape[0], -1, -1)
    t_minus_1 = (token_t - 1).clamp_min(0)
    qbar_prev = transition.qt_bar.to(device)[t_minus_1]
    qbar_prev = torch.where((token_t == 0).view(-1, 1, 1), eye, qbar_prev)

    left = torch.einsum("nj,nkj->nk", x_t_onehot, qt)
    numerator = left.unsqueeze(1) * qbar_prev
    denominator = torch.einsum("nj,nij->ni", x_t_onehot, qbar_t)
    true_post = (
        numerator / denominator.unsqueeze(-1).clamp_min(1e-12)
    ).gather(1, x0.long().view(-1, 1, 1).expand(-1, 1, num_classes)).squeeze(1).clamp_min(1e-12)

    model_post = torch.einsum(
        "ni,nik->nk",
        x0_probs,
        numerator / denominator.unsqueeze(-1).clamp_min(1e-12),
    ).clamp_min(1e-12)
    kl = (true_post * (true_post.log() - model_post.log())).sum(dim=-1)
    loss = torch.where(token_t == 0, ce, kl)
    mask = mask.to(dtype=loss.dtype)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def _d3pm_posterior_over_x0_packed(
    transition: D3PMTransition,
    x_t: torch.Tensor,
    token_t: torch.Tensor,
) -> torch.Tensor:
    num_classes = transition.num_classes
    device = x_t.device
    token_t = token_t.long().to(device)
    x_t_onehot = F.one_hot(x_t.long(), num_classes).float()
    qt = transition.qt.to(device)[token_t]
    qbar_t = transition.qt_bar.to(device)[token_t]
    eye = transition.identity.to(device).expand(token_t.shape[0], -1, -1)
    t_minus_1 = (token_t - 1).clamp_min(0)
    qbar_prev = transition.qt_bar.to(device)[t_minus_1]
    qbar_prev = torch.where((token_t == 0).view(-1, 1, 1), eye, qbar_prev)

    left = torch.einsum("nj,nkj->nk", x_t_onehot, qt)
    numerator = left.unsqueeze(1) * qbar_prev
    denominator = torch.einsum("nj,nij->ni", x_t_onehot, qbar_t)
    return numerator / denominator.unsqueeze(-1).clamp_min(1e-12)


@torch.no_grad()
def _d3pm_p_sample_packed(
    transition: D3PMTransition,
    x_t: torch.Tensor,
    token_t: torch.Tensor,
    x0_logits: torch.Tensor,
) -> torch.Tensor:
    x0_probs = F.softmax(x0_logits, dim=-1)
    posterior_all = _d3pm_posterior_over_x0_packed(transition, x_t, token_t)
    posterior = torch.einsum("ni,nik->nk", x0_probs, posterior_all)
    posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    x_prev = sample_categorical(posterior)
    x0_sample = sample_categorical(x0_probs)
    return torch.where(token_t == 0, x0_sample, x_prev)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def extract(values: torch.Tensor, t: torch.Tensor, x_shape: torch.Size | tuple[int, ...]) -> torch.Tensor:
    out = values.gather(0, t.long())
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1))).to(dtype=torch.float32)


def sample_categorical(probs: torch.Tensor) -> torch.Tensor:
    original_shape = probs.shape[:-1]
    flat = probs.reshape(-1, probs.shape[-1]).clamp_min(0.0)
    flat = flat / flat.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    samples = torch.multinomial(flat, num_samples=1).squeeze(-1)
    return samples.reshape(original_shape)
