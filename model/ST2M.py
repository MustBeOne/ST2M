from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from project.model.diffusion import ContinuousDDPM, D3PMDiffusion, D3PMTransition

from project.model.model_utils import (
    _labels,
    _sum_by_batch,
    _lattice_mask_from_space_group,
    _complete_lattice_from_space_group,
    _sinusoidal_timestep_embedding,
    _smooth_excess,
    _d3pm_q_sample_packed,
    _packed_per_sample_mean,
    _d3pm_vb_loss_packed,
    _d3pm_p_sample_packed,
    MLP,
)

@dataclass
class ST2MOutput:
    loss: torch.Tensor
    continuous_objective: torch.Tensor
    discrete_objective: torch.Tensor
    discrete_loss: torch.Tensor
    continuous_loss: torch.Tensor
    lattice_loss: torch.Tensor
    contrastive_con_loss: torch.Tensor
    contrastive_dis_loss: torch.Tensor
    t: torch.Tensor
    discrete_t: torch.Tensor
    continuous_t: torch.Tensor
    lattice_t: torch.Tensor
    discrete_logits: torch.Tensor
    predicted_noise: torch.Tensor
    predicted_lattice_noise: torch.Tensor


class DiscreteCountRegularizer(nn.Module):
    """Penalize over-populated predicted Wyckoff orbit slots."""

    def __init__(
        self,
        vacancy_class: int = 0,
        orbit_margin: float = 0.0,
        atom_margin: float = 0.0,
        smooth_tau: float = 0.5,
    ) -> None:
        super().__init__()
        self.vacancy_class = int(vacancy_class)
        self.orbit_margin = float(orbit_margin)
        self.atom_margin = float(atom_margin)
        self.smooth_tau = float(smooth_tau)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        slot_mask: torch.Tensor,
        multiplicity: torch.Tensor,
        batch_index: torch.Tensor,
        batch_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probs = F.softmax(logits, dim=-1)
        slot_mask = slot_mask.to(dtype=probs.dtype)
        p_occ = (1.0 - probs[:, self.vacancy_class]) * slot_mask
        true_occ = (labels != self.vacancy_class).to(dtype=probs.dtype) * slot_mask

        pred_orbits = _sum_by_batch(p_occ, batch_index, batch_size)
        true_orbits = _sum_by_batch(true_occ, batch_index, batch_size)
        orbit_limit = true_orbits + self.orbit_margin
        orbit_excess = _smooth_excess(pred_orbits, orbit_limit, self.smooth_tau)
        orbit_loss = (orbit_excess / orbit_limit.clamp_min(1.0)).pow(2).mean()

        mult = multiplicity.to(dtype=probs.dtype).clamp_min(0.0)
        pred_atoms = _sum_by_batch(p_occ * mult, batch_index, batch_size)
        true_atoms = _sum_by_batch(true_occ * mult, batch_index, batch_size)
        atom_limit = true_atoms + self.atom_margin
        atom_excess = _smooth_excess(pred_atoms, atom_limit, self.smooth_tau)
        multiplicity_loss = (atom_excess / atom_limit.clamp_min(1.0)).pow(2).mean()
        return orbit_loss, multiplicity_loss


class ST2MBranch(nn.Module):
    """One fully independent conditional denoising branch."""

    def __init__(
        self,
        branch: str,
        num_classes: int,
        max_sites: int,
        max_occ: int,
        lattice_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_spacegroup: int,
        max_multiplicity: int,
        max_dof: int,
        max_letter_index: int,
    ) -> None:
        super().__init__()
        if branch not in {"discrete", "continuous"}:
            raise ValueError("branch must be 'discrete' or 'continuous'.")
        self.branch = branch
        self.num_classes = int(num_classes)
        self.d_model = int(d_model)

        self.single_discrete_mlp = MLP(num_classes, d_model, d_model, dropout)
        self.single_continuous_mlp = MLP(3, d_model, d_model, dropout)
        self.lattice_mlp = MLP(lattice_dim, d_model, d_model, dropout)
        self.time_mlp = MLP(d_model, d_model * 2, d_model, dropout)
        self.letter_emb = nn.Embedding(max_letter_index + 1, d_model, padding_idx=0)
        self.multiplicity_emb = nn.Embedding(max_multiplicity + 1, d_model, padding_idx=0)
        self.dof_emb = nn.Embedding(max_dof + 1, d_model)
        self.spacegroup_emb = nn.Embedding(max_spacegroup + 1, d_model, padding_idx=0)
        self.position_emb = nn.Embedding(max_sites, d_model)
        self.slot_emb = nn.Embedding(max_occ, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)
        if branch == "discrete":
            self.discrete_head = nn.Linear(d_model, num_classes)
        else:
            self.continuous_head = nn.Linear(d_model, 3)
            self.lattice_head = MLP(d_model, d_model, lattice_dim, dropout)

    def _embed(
        self,
        noisy_discrete: torch.Tensor,
        noisy_continuous: torch.Tensor,
        noisy_lattice: torch.Tensor,
        t: torch.Tensor,
        batch_index: torch.Tensor,
        letter: torch.Tensor,
        space_group: torch.Tensor,
    ) -> torch.Tensor:
        token_t = t.long()[batch_index]
        token_space_group = space_group.long()[batch_index]
        return (
            self.single_discrete_mlp(noisy_discrete.float())
            + self.single_continuous_mlp(noisy_continuous.float())
            + self.lattice_mlp(noisy_lattice.float())[batch_index]
            + self.time_mlp(_sinusoidal_timestep_embedding(token_t, self.d_model))
            + self.letter_emb(letter.clamp_min(0))
            + self.spacegroup_emb(token_space_group.clamp_min(0))
        )

    def forward(
        self,
        noisy_discrete: torch.Tensor,
        noisy_continuous: torch.Tensor,
        noisy_lattice: torch.Tensor,
        t: torch.Tensor,
        batch_index: torch.Tensor,
        letter: torch.Tensor,
        space_group: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        tokens = self._embed(
            noisy_discrete,
            noisy_continuous,
            noisy_lattice,
            t,
            batch_index,
            letter,
            space_group,
        )
        attention_mask = batch_index[:, None] != batch_index[None, :]
        hidden = self.final_norm(
            self.transformer(tokens.unsqueeze(0), mask=attention_mask)
        ).squeeze(0)
        if self.branch == "discrete":
            return self.discrete_head(hidden)

        continuous_noise = self.continuous_head(hidden)
        batch_size = noisy_lattice.shape[0]
        pooled = hidden.new_zeros(batch_size, hidden.shape[-1])
        pooled.index_add_(0, batch_index, hidden)
        counts = torch.bincount(batch_index, minlength=batch_size).to(dtype=hidden.dtype).clamp_min(1.0)
        lattice_noise = self.lattice_head(pooled / counts.unsqueeze(-1))
        return continuous_noise, lattice_noise


class ST2MDenoiser(nn.Module):
    """Predict denoising targets with either shared or fully separate networks."""

    def __init__(
        self,
        num_classes: int = 101,
        max_sites: int = 27,
        max_occ: int = 30,
        lattice_dim: int = 6,
        timesteps: int = 1000,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        denoiser_mode: str = "shared",
        max_spacegroup: int = 230,
        max_multiplicity: int = 256,
        max_dof: int = 3,
        max_letter_index: int = 52,
    ) -> None:
        super().__init__()
        if denoiser_mode not in {"shared", "separate"}:
            raise ValueError("denoiser_mode must be 'shared' or 'separate'.")
        self.num_classes = int(num_classes)
        self.max_sites = int(max_sites)
        self.max_occ = int(max_occ)
        self.lattice_dim = int(lattice_dim)
        self.timesteps = int(timesteps)
        self.d_model = int(d_model)
        self.denoiser_mode = denoiser_mode

        if denoiser_mode == "shared":
            self.single_discrete_mlp = MLP(num_classes, d_model, d_model, dropout)
            self.single_continuous_mlp = MLP(3, d_model, d_model, dropout)
            self.lattice_mlp = MLP(lattice_dim, d_model, d_model, dropout)
            self.time_mlp = MLP(d_model, d_model * 2, d_model, dropout)
            self.letter_emb = nn.Embedding(max_letter_index + 1, d_model, padding_idx=0)
            self.multiplicity_emb = nn.Embedding(max_multiplicity + 1, d_model, padding_idx=0)
            self.dof_emb = nn.Embedding(max_dof + 1, d_model)
            self.spacegroup_emb = nn.Embedding(max_spacegroup + 1, d_model, padding_idx=0)
            self.position_emb = nn.Embedding(max_sites, d_model)
            self.slot_emb = nn.Embedding(max_occ, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.shared_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.discrete_head = nn.Linear(d_model, num_classes)
            self.continuous_head = nn.Linear(d_model, 3)
            self.lattice_head = MLP(d_model, d_model, lattice_dim, dropout)
            self.final_norm = nn.LayerNorm(d_model)
            self.discrete_model = None
            self.continuous_model = None
        else:
            branch_args = dict(
                num_classes=num_classes,
                max_sites=max_sites,
                max_occ=max_occ,
                lattice_dim=lattice_dim,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                max_spacegroup=max_spacegroup,
                max_multiplicity=max_multiplicity,
                max_dof=max_dof,
                max_letter_index=max_letter_index,
            )
            self.discrete_model = ST2MBranch(branch="discrete", **branch_args)
            self.continuous_model = ST2MBranch(branch="continuous", **branch_args)

    def _embed(
        self,
        noisy_discrete: torch.Tensor,
        noisy_continuous: torch.Tensor,
        noisy_lattice: torch.Tensor,
        t: torch.Tensor,
        batch_index: torch.Tensor,
        letter: torch.Tensor,
        space_group: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_discrete.dim() == 1:
            noisy_discrete = F.one_hot(noisy_discrete.long(), num_classes=self.num_classes).float()
        batch_index = batch_index.long()
        token_t = t.long()[batch_index]
        token_space_group = space_group.long()[batch_index]
        return (
            self.single_discrete_mlp(noisy_discrete.float())
            + self.single_continuous_mlp(noisy_continuous.float())
            + self.lattice_mlp(noisy_lattice.float())[batch_index]
            + self.time_mlp(_sinusoidal_timestep_embedding(token_t, self.d_model))
            + self.letter_emb(letter.clamp_min(0))
            + self.spacegroup_emb(token_space_group.clamp_min(0))
        )

    def forward(
        self,
        noisy_discrete: torch.Tensor,
        noisy_continuous: torch.Tensor,
        noisy_lattice: torch.Tensor,
        t: torch.Tensor,
        batch_index: torch.Tensor,
        letter: torch.Tensor,
        space_group: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if noisy_discrete.dim() == 1:
            noisy_discrete = F.one_hot(noisy_discrete.long(), num_classes=self.num_classes).float()
        elif noisy_discrete.dim() != 2:
            raise ValueError(f"packed noisy_discrete must be [N] or [N,C], got {tuple(noisy_discrete.shape)}.")
        noisy_discrete = noisy_discrete.float()
        noisy_continuous = noisy_continuous.float()
        noisy_lattice = noisy_lattice.float()
        batch_index = batch_index.long()

        if self.denoiser_mode == "shared":
            tokens = self._embed(
                noisy_discrete,
                noisy_continuous,
                noisy_lattice,
                t,
                batch_index,
                letter.long(),
                space_group.long(),
            )
            attention_mask = batch_index[:, None] != batch_index[None, :]
            tokens = tokens.unsqueeze(0)
            hidden = self.final_norm(self.shared_transformer(tokens, mask=attention_mask)).squeeze(0)
            discrete_logits = self.discrete_head(hidden)
            continuous_noise = self.continuous_head(hidden)
            continuous_hidden = hidden
        else:
            branch_inputs = (
                noisy_discrete,
                noisy_continuous,
                noisy_lattice,
                t,
                batch_index,
                letter.long(),
                space_group.long(),
            )
            discrete_logits = self.discrete_model(*branch_inputs)
            continuous_noise, lattice_noise = self.continuous_model(*branch_inputs)
            return discrete_logits, continuous_noise, lattice_noise

        batch_size = noisy_lattice.shape[0]
        pooled = continuous_hidden.new_zeros(batch_size, continuous_hidden.shape[-1])
        pooled.index_add_(0, batch_index, continuous_hidden)
        counts = torch.bincount(batch_index, minlength=batch_size).to(dtype=continuous_hidden.dtype).clamp_min(1.0)
        pooled = pooled / counts.unsqueeze(-1)
        lattice_noise = self.lattice_head(pooled)
        return discrete_logits, continuous_noise, lattice_noise


class ST2MDiffusion(nn.Module):
    """Joint D3PM/DDPM wrapper around the Wyckoff-site Transformer denoiser."""

    def __init__(
        self,
        num_classes: int = 101,
        max_sites: int = 27,
        max_occ: int = 30,
        lattice_dim: int = 6,
        timesteps: int = 1000,
        discrete_schedule: str = "cosine",
        continuous_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        denoiser_mode: str = "shared",
        discrete_loss_weight: float = 1.0,
        continuous_loss_weight: float = 1.0,
        lattice_loss_weight: float = 1.0,
        use_count_losses: bool = False,
        lambda_contrastive_con: float = 0.0,
        lambda_contrastive_dis: float = 0.0,
        contrastive_margin: float = 1.0,
        contrastive_use_x0: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.max_sites = int(max_sites)
        self.max_occ = int(max_occ)
        self.lattice_dim = int(lattice_dim)
        self.timesteps = int(timesteps)
        self.denoiser_mode = denoiser_mode
        self.discrete_loss_weight = float(discrete_loss_weight)
        self.continuous_loss_weight = float(continuous_loss_weight)
        self.lattice_loss_weight = float(lattice_loss_weight)
        self.use_count_losses = bool(use_count_losses)
        self.lambda_contrastive_con = float(lambda_contrastive_con)
        self.lambda_contrastive_dis = float(lambda_contrastive_dis)
        self.contrastive_margin = float(contrastive_margin)
        self.contrastive_use_x0 = bool(contrastive_use_x0)

        self.denoiser = ST2MDenoiser(
            num_classes=num_classes,
            max_sites=max_sites,
            max_occ=max_occ,
            lattice_dim=lattice_dim,
            timesteps=timesteps,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            denoiser_mode=denoiser_mode,
        )
        transition = D3PMTransition(
            timesteps=timesteps,
            num_classes=num_classes,
            schedule=discrete_schedule,  # type: ignore[arg-type]
            beta_start=beta_start,
            beta_end=beta_end,
        )
        self.discrete_diffusion = D3PMDiffusion(self.denoiser, transition=transition)
        self.continuous_diffusion = ContinuousDDPM(
            denoiser=nn.Identity(),
            timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            schedule=continuous_schedule,  # type: ignore[arg-type]
        )

    @staticmethod
    def _same_orbit_count_permutation(
        batch_index: torch.Tensor,
        space_group: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Permute conditions within each space group and orbit-count stratum."""
        batch_size = space_group.shape[0]
        permutation = torch.arange(batch_size, device=batch_index.device)
        active = torch.zeros(batch_size, dtype=torch.bool, device=batch_index.device)
        orbit_counts = torch.bincount(batch_index.long(), minlength=batch_size)
        for group in torch.unique(space_group):
            group_indices = torch.nonzero(space_group == group, as_tuple=False).flatten()
            group_counts = orbit_counts[group_indices]
            for orbit_count in torch.unique(group_counts):
                indices = group_indices[group_counts == orbit_count]
                if indices.numel() > 1:
                    shift = torch.randint(1, indices.numel(), (1,), device=batch_index.device)
                    permutation[indices] = indices.roll(int(shift.item()))
                    active[indices] = True
        return permutation, active

    @staticmethod
    def _permute_packed_by_sample(
        values: torch.Tensor,
        batch_index: torch.Tensor,
        permutation: torch.Tensor,
    ) -> torch.Tensor:
        """Move whole packed sample blocks between equal-length samples."""
        result = values.clone()
        for target_index, source_index in enumerate(permutation.tolist()):
            if target_index == source_index:
                continue
            target_rows = torch.nonzero(batch_index == target_index, as_tuple=False).flatten()
            source_rows = torch.nonzero(batch_index == source_index, as_tuple=False).flatten()
            if target_rows.numel() != source_rows.numel():
                raise ValueError("negative conditions require equal orbit counts.")
            result[target_rows] = values[source_rows]
        return result

    def _contrastive_losses(
        self,
        labels_0: torch.Tensor,
        continuous_0: torch.Tensor,
        lattice_0: torch.Tensor,
        lattice_mask: torch.Tensor,
        batch_index: torch.Tensor,
        discrete_loss_mask: torch.Tensor,
        continuous_mask: torch.Tensor,
        t: torch.Tensor,
        token_t: torch.Tensor,
        discrete_t: torch.Tensor,
        continuous_t: torch.Tensor,
        lattice_t: torch.Tensor,
        discrete_logits: torch.Tensor,
        predicted_noise: torch.Tensor,
        predicted_lattice_noise: torch.Tensor,
        target_noise: torch.Tensor,
        target_lattice_noise: torch.Tensor,
        letter: torch.Tensor,
        space_group: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Use continuous noise or x0 errors; discrete distances always use x0 CE."""
        zero = continuous_0.new_tensor(0.0)
        permutation, active = self._same_orbit_count_permutation(batch_index, space_group)
        if not bool(active.any()):
            return zero, zero

        negative_labels_0 = self._permute_packed_by_sample(labels_0, batch_index, permutation)
        negative_continuous_0 = self._permute_packed_by_sample(continuous_0, batch_index, permutation)
        negative_lattice_0 = lattice_0[permutation]

        negative_discrete_t = _d3pm_q_sample_packed(
            self.discrete_diffusion.transition,
            negative_labels_0,
            token_t,
        )
        negative_continuous_t, _ = self.continuous_diffusion.q_sample(
            negative_continuous_0,
            token_t,
            noise=target_noise,
            mask=continuous_mask,
        )
        negative_lattice_t, _ = self.continuous_diffusion.q_sample(
            negative_lattice_0,
            t,
            noise=target_lattice_noise,
            mask=lattice_mask,
        )

        _, negative_continuous_noise, negative_lattice_noise = self.denoiser(
            negative_discrete_t,
            continuous_t,
            lattice_t,
            t,
            batch_index=batch_index,
            letter=letter,
            space_group=space_group,
        )
        negative_discrete_logits, _, _ = self.denoiser(
            discrete_t,
            negative_continuous_t,
            negative_lattice_t,
            t,
            batch_index=batch_index,
            letter=letter,
            space_group=space_group,
        )

        if self.contrastive_use_x0:
            # Both conditions reconstruct the same target from its noisy state.
            predicted_noise = self.continuous_diffusion.predict_x0_from_eps(
                continuous_t, token_t, predicted_noise
            )
            negative_continuous_noise = self.continuous_diffusion.predict_x0_from_eps(
                continuous_t, token_t, negative_continuous_noise
            )
            predicted_lattice_noise = self.continuous_diffusion.predict_x0_from_eps(
                lattice_t, t, predicted_lattice_noise
            )
            negative_lattice_noise = self.continuous_diffusion.predict_x0_from_eps(
                lattice_t, t, negative_lattice_noise
            )
            target_noise = continuous_0
            target_lattice_noise = lattice_0

        coordinate_weights = continuous_mask.sum(dim=-1)
        positive_coordinate_error = ((target_noise - predicted_noise).square() * continuous_mask).sum(dim=-1)
        negative_coordinate_error = ((target_noise - negative_continuous_noise).square() * continuous_mask).sum(dim=-1)
        coordinate_counts = _sum_by_batch(coordinate_weights, batch_index, lattice_0.shape[0])
        positive_errors = _sum_by_batch(positive_coordinate_error, batch_index, lattice_0.shape[0])
        negative_errors = _sum_by_batch(negative_coordinate_error, batch_index, lattice_0.shape[0])
        lattice_counts = lattice_mask.sum(dim=-1)
        positive_errors = positive_errors + (
            (target_lattice_noise - predicted_lattice_noise).square() * lattice_mask
        ).sum(dim=-1)
        negative_errors = negative_errors + (
            (target_lattice_noise - negative_lattice_noise).square() * lattice_mask
        ).sum(dim=-1)
        continuous_distance_positive = positive_errors / (coordinate_counts + lattice_counts).clamp_min(1.0)
        continuous_distance_negative = negative_errors / (coordinate_counts + lattice_counts).clamp_min(1.0)

        discrete_distance_positive = _packed_per_sample_mean(
            F.cross_entropy(discrete_logits, labels_0, reduction="none"),
            discrete_loss_mask,
            batch_index,
            lattice_0.shape[0],
        )
        discrete_distance_negative = _packed_per_sample_mean(
            F.cross_entropy(negative_discrete_logits, labels_0, reduction="none"),
            discrete_loss_mask,
            batch_index,
            lattice_0.shape[0],
        )
        active_weights = active.to(dtype=continuous_0.dtype)
        continuous_triplets = F.relu(
            continuous_distance_positive - continuous_distance_negative + self.contrastive_margin
        )
        discrete_triplets = F.relu(
            discrete_distance_positive - discrete_distance_negative + self.contrastive_margin
        )
        continuous_loss = (continuous_triplets * active_weights).sum() / active_weights.sum().clamp_min(1.0)
        discrete_loss = (discrete_triplets * active_weights).sum() / active_weights.sum().clamp_min(1.0)
        return continuous_loss, discrete_loss

    def forward(
        self,
        discrete_0: torch.Tensor,
        continuous_0: torch.Tensor,
        lattice_0: torch.Tensor,
        lattice_mask: torch.Tensor | None,
        batch_index: torch.Tensor,
        discrete_loss_mask: torch.Tensor,
        continuous_mask: torch.Tensor,
        letter: torch.Tensor,
        multiplicity: torch.Tensor,
        space_group: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> ST2MOutput:
        labels_0 = _labels(discrete_0)
        continuous_0 = continuous_0.float()
        lattice_0 = lattice_0.reshape(lattice_0.shape[0], -1).float()
        if lattice_mask is None:
            lattice_mask = _lattice_mask_from_space_group(space_group, self.lattice_dim)
        lattice_mask = lattice_mask.reshape_as(lattice_0).to(device=lattice_0.device, dtype=lattice_0.dtype)
        batch_index = batch_index.long().to(labels_0.device)
        discrete_loss_mask = discrete_loss_mask.float()
        continuous_mask = continuous_mask.to(device=continuous_0.device, dtype=continuous_0.dtype)
        batch = lattice_0.shape[0]

        if t is None:
            t = torch.randint(self.timesteps, (batch,), device=labels_0.device)
        else:
            t = t.long().to(labels_0.device)

        token_t = t[batch_index]
        discrete_t = _d3pm_q_sample_packed(self.discrete_diffusion.transition, labels_0, token_t)

        cont_mask = continuous_mask.reshape_as(continuous_0)
        continuous_t, target_noise = self.continuous_diffusion.q_sample(continuous_0, token_t, mask=cont_mask)
        lattice_t, target_lattice_noise = self.continuous_diffusion.q_sample(lattice_0, t, mask=lattice_mask)

        discrete_logits, predicted_noise, predicted_lattice_noise = self.denoiser(
            discrete_t,
            continuous_t,
            lattice_t,
            t,
            batch_index=batch_index,
            letter=letter,
            space_group=space_group,
        )
        discrete_loss = _d3pm_vb_loss_packed(
            self.discrete_diffusion.transition,
            labels_0,
            discrete_t,
            token_t,
            discrete_logits,
            discrete_loss_mask,
        )
        continuous_loss = self.continuous_diffusion.mse_loss(predicted_noise, target_noise, mask=cont_mask)
        lattice_loss = self.continuous_diffusion.mse_loss(
            predicted_lattice_noise,
            target_lattice_noise,
            mask=lattice_mask,
        )
        orbit_count_loss = continuous_0.new_tensor(0.0)
        multiplicity_count_loss = continuous_0.new_tensor(0.0)
        
        if self.lambda_contrastive_con > 0 or self.lambda_contrastive_dis > 0:
            contrastive_con_loss, contrastive_dis_loss = self._contrastive_losses(
                labels_0,
                continuous_0,
                lattice_0,
                lattice_mask,
                batch_index,
                discrete_loss_mask,
                cont_mask,
                t,
                token_t,
                discrete_t,
                continuous_t,
                lattice_t,
                discrete_logits,
                predicted_noise,
                predicted_lattice_noise,
                target_noise,
                target_lattice_noise,
                letter,
                space_group,
            )
        else:
            contrastive_con_loss = continuous_0.new_tensor(0.0)
            contrastive_dis_loss = continuous_0.new_tensor(0.0)
        continuous_objective = (
            self.continuous_loss_weight * continuous_loss
            + self.lattice_loss_weight * lattice_loss
            + self.lambda_contrastive_con * contrastive_con_loss
        )
        discrete_objective = (
            self.discrete_loss_weight * discrete_loss
            + self.lambda_contrastive_dis * contrastive_dis_loss
        )
        loss = continuous_objective + discrete_objective
        return ST2MOutput(
            loss=loss,
            continuous_objective=continuous_objective,
            discrete_objective=discrete_objective,
            discrete_loss=discrete_loss,
            continuous_loss=continuous_loss,
            lattice_loss=lattice_loss,
            contrastive_con_loss=contrastive_con_loss,
            contrastive_dis_loss=contrastive_dis_loss,
            t=t,
            discrete_t=discrete_t,
            continuous_t=continuous_t,
            lattice_t=lattice_t,
            discrete_logits=discrete_logits,
            predicted_noise=predicted_noise,
            predicted_lattice_noise=predicted_lattice_noise,
        )

    @torch.no_grad()
    def sample(
        self,
        template_batch: dict[str, torch.Tensor],
        device: torch.device | str,
    ) -> dict[str, torch.Tensor]:
        """Run reverse diffusion on an occupied-orbit packed template."""

        batch_index = template_batch["batch_index"].to(device=device, dtype=torch.long)
        continuous_mask = template_batch["continuous_mask"].to(device=device, dtype=torch.float32)
        token_letter = template_batch["wyckoff_letter"].to(device=device, dtype=torch.long)
        token_dof = template_batch["wyckoff_dof"].to(device=device, dtype=torch.long)
        space_group = template_batch["space_group"].to(device=device, dtype=torch.long)

        batch = space_group.shape[0]
        if batch_index.numel() == 0:
            raise ValueError("Template batch contains no occupied-orbit tokens.")
        token_count = batch_index.numel()
        token_shapes = {
            "continuous_mask": continuous_mask.shape[0],
            "letter": token_letter.shape[0],
            "dof": token_dof.shape[0],
        }
        if any(size != token_count for size in token_shapes.values()):
            raise ValueError(f"Packed template token counts do not match: {token_shapes}.")

        discrete_t = self.discrete_diffusion.transition.sample_prior(
            batch_size=1,
            num_rows=token_count,
            device=device,
        ).reshape(-1)
        continuous_mask = continuous_mask.reshape(token_count, 3)
        continuous_t = torch.randn(token_count, 3, device=device) * continuous_mask
        lattice_mask = _lattice_mask_from_space_group(space_group, self.lattice_dim)
        lattice_t = torch.randn(batch, self.lattice_dim, device=device) * lattice_mask

        for step in reversed(range(self.timesteps)):
            t = torch.full((batch,), step, dtype=torch.long, device=device)
            token_t = t[batch_index]
            discrete_logits, predicted_noise, predicted_lattice_noise = self.denoiser(
                discrete_t,
                continuous_t,
                lattice_t,
                t,
                batch_index=batch_index,
                letter=token_letter,
                space_group=space_group,
            )
            continuous_t = self.continuous_diffusion.p_sample(
                continuous_t,
                token_t,
                predicted_noise,
                mask=continuous_mask,
                clip_x0=True,
            )
            lattice_t = self.continuous_diffusion.p_sample(
                lattice_t,
                t,
                predicted_lattice_noise,
                mask=lattice_mask,
                clip_x0=False,
            )
            discrete_t = _d3pm_p_sample_packed(
                self.discrete_diffusion.transition,
                discrete_t,
                token_t,
                discrete_logits,
            )
            continuous_t = continuous_t * continuous_mask

        lattice_t = _complete_lattice_from_space_group(lattice_t, space_group)

        return {
            "discrete_labels": discrete_t,
            "continuous": continuous_t,
            "lattice": lattice_t,
            "batch_index": batch_index,
            "continuous_mask": continuous_mask,
            "space_group": space_group,
            "wyckoff_letter": token_letter,
            "wyckoff_dof": token_dof,
        }
