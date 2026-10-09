from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from project.utils.config import NUM_DISCRETE_CLASSES

from project.model.model_utils import (
    extract,
    sample_categorical,
)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = torch.arange(timesteps + 1, dtype=torch.float64)
    x = steps / timesteps
    alpha_bar = torch.cos((x + s) / (1 + s) * torch.pi / 2).pow(2)
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(1e-8, 0.999).float()


def linear_beta_schedule(
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


class D3PMTransition(nn.Module):
    def __init__(
        self,
        timesteps: int = 1000,
        num_classes: int = NUM_DISCRETE_CLASSES,
        marginal: torch.Tensor | None = None,
        schedule: Literal["cosine", "linear"] = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        absorbing_class: int = 0,
    ) -> None:
        super().__init__()
        self.timesteps = int(timesteps)
        self.num_classes = int(num_classes)
        self.absorbing_class = int(absorbing_class)
        if self.absorbing_class < 0 or self.absorbing_class >= self.num_classes:
            raise ValueError(
                f"absorbing_class must be in [0, {self.num_classes - 1}], "
                f"got {self.absorbing_class}."
            )

        if schedule == "cosine":
            betas = cosine_beta_schedule(self.timesteps)
        elif schedule == "linear":
            betas = linear_beta_schedule(self.timesteps, beta_start, beta_end)
        else:
            raise ValueError(f"Unknown discrete schedule {schedule!r}.")
        betas[-1] = 1.0

        if marginal is None:
            marginal = torch.zeros(self.num_classes, dtype=torch.float32)
            marginal[self.absorbing_class] = 1.0
        marginal = marginal.float().flatten()
        if marginal.numel() != self.num_classes:
            raise ValueError(
                f"marginal must have {self.num_classes} entries, got {marginal.numel()}."
            )
        marginal = marginal.clamp_min(0.0)
        if marginal.sum() <= 0:
            raise ValueError("marginal must have positive total probability.")
        marginal = marginal / marginal.sum()

        self.register_buffer("betas", betas.float())
        self.register_buffer("marginal", marginal.float())
        self.register_buffer("identity", torch.eye(self.num_classes, dtype=torch.float32))

        qt = []
        for beta in self.betas:
            qt.append((1 - beta) * self.identity + beta * self.marginal[None, :])
        qt_tensor = torch.stack(qt, dim=0)
        self.register_buffer("qt", qt_tensor)
        self.register_buffer("qt_bar", torch.empty_like(qt_tensor), persistent=False)
        self._rebuild_qt_bar()

    def _rebuild_qt_bar(self) -> None:
        cumulative = []
        running = self.qt[0]
        cumulative.append(running)
        for idx in range(1, self.timesteps):
            running = running @ self.qt[idx]
            cumulative.append(running)
        self.qt_bar = torch.stack(cumulative, dim=0)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample ``x_t`` from ``q(x_t | x_0)`` for integer labels ``x0``."""
        if x0.dim() != 2:
            raise ValueError(f"x0 must be [B, N], got {tuple(x0.shape)}.")
        t = t.long().to(x0.device)
        x0_onehot = F.one_hot(x0.long(), self.num_classes).float()
        qbar = self.qt_bar.to(x0.device)[t]
        probs = torch.einsum("bnc,bcd->bnd", x0_onehot, qbar)
        return sample_categorical(probs)

    def q_posterior_over_x0(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x_t.dim() != 2:
            raise ValueError(f"x_t must be [B, N], got {tuple(x_t.shape)}.")

        device = x_t.device
        t = t.long().to(device)
        x_t_onehot = F.one_hot(x_t.long(), self.num_classes).float()
        qt = self.qt.to(device)[t]
        qbar_t = self.qt_bar.to(device)[t]

        eye = self.identity.to(device).expand(t.shape[0], -1, -1)
        t_minus_1 = (t - 1).clamp_min(0)
        qbar_prev = self.qt_bar.to(device)[t_minus_1]
        qbar_prev = torch.where((t == 0).view(-1, 1, 1), eye, qbar_prev)

        left = torch.einsum("bnj,bkj->bnk", x_t_onehot, qt)
        numerator = left.unsqueeze(2) * qbar_prev.unsqueeze(1)
        denominator = torch.einsum("bnj,bij->bni", x_t_onehot, qbar_t)
        return numerator / denominator.unsqueeze(-1).clamp_min(1e-12)

    def model_posterior_probs(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x0_probs: torch.Tensor,
    ) -> torch.Tensor:
        posterior = self.q_posterior_over_x0(x_t, t)
        probs = torch.einsum("bni,bnik->bnk", x0_probs, posterior)
        return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def sample_prior(self, batch_size: int, num_rows: int, device: torch.device | str) -> torch.Tensor:
        if self.marginal[self.absorbing_class].item() == 1.0:
            return torch.full(
                (batch_size, num_rows),
                self.absorbing_class,
                dtype=torch.long,
                device=device,
            )
        probs = self.marginal.to(device).view(1, 1, self.num_classes)
        probs = probs.expand(batch_size, num_rows, self.num_classes)
        return sample_categorical(probs)


class D3PMDiffusion(nn.Module):
    def __init__(
        self,
        denoiser: nn.Module,
        transition: D3PMTransition | None = None,
        timesteps: int = 1000,
        num_classes: int = NUM_DISCRETE_CLASSES,
    ) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.transition = transition or D3PMTransition(
            timesteps=timesteps,
            num_classes=num_classes,
        )
        self.timesteps = self.transition.timesteps
        self.num_classes = self.transition.num_classes

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.transition.q_sample(x0, t)

    def predict_x0_logits(self, *args, **kwargs) -> torch.Tensor:
        return self.denoiser(*args, **kwargs)

    def vb_loss(
        self,
        x0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x0_logits: torch.Tensor,
        row_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x0_log_probs = F.log_softmax(x0_logits, dim=-1)
        x0_probs = x0_log_probs.exp()

        ce = F.nll_loss(
            x0_log_probs.reshape(-1, self.num_classes),
            x0.long().reshape(-1),
            reduction="none",
        ).reshape_as(x0.float())

        posterior_all = self.transition.q_posterior_over_x0(x_t, t)
        gather_idx = x0.long().unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, self.num_classes)
        true_post = posterior_all.gather(2, gather_idx).squeeze(2).clamp_min(1e-12)
        model_post = self.transition.model_posterior_probs(x_t, t, x0_probs).clamp_min(1e-12)
        kl = (true_post * (true_post.log() - model_post.log())).sum(dim=-1)

        use_ce = (t == 0).view(-1, 1)
        loss = torch.where(use_ce, ce, kl)
        if row_mask is not None:
            loss = loss * row_mask.float()
            return loss.sum() / row_mask.float().sum().clamp_min(1.0)
        return loss.mean()

    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x0_logits: torch.Tensor,
    ) -> torch.Tensor:
        x0_probs = F.softmax(x0_logits, dim=-1)
        posterior = self.transition.model_posterior_probs(x_t, t, x0_probs)
        x_prev = sample_categorical(posterior)
        x0_sample = sample_categorical(x0_probs)
        return torch.where((t == 0).view(-1, 1), x0_sample, x_prev)


class ContinuousDDPM(nn.Module):
    def __init__(
        self,
        denoiser: nn.Module,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        schedule: Literal["linear", "cosine"] = "linear",
    ) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.timesteps = int(timesteps)

        if schedule == "linear":
            betas = linear_beta_schedule(self.timesteps, beta_start, beta_end)
        elif schedule == "cosine":
            betas = cosine_beta_schedule(self.timesteps)
        else:
            raise ValueError(f"Unknown continuous schedule {schedule!r}.")

        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alphas_bar", alphas_bar.float())
        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar).float())
        self.register_buffer(
            "sqrt_one_minus_alphas_bar",
            torch.sqrt(1.0 - alphas_bar).float(),
        )
        self.register_buffer("sqrt_recip_alphas_bar", torch.sqrt(1.0 / alphas_bar).float())
        self.register_buffer(
            "sqrt_recipm1_alphas_bar",
            torch.sqrt(1.0 / alphas_bar - 1.0).float(),
        )
        posterior_var = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
        self.register_buffer("posterior_var", posterior_var.float())
        self.register_buffer(
            "posterior_log_var_clipped",
            torch.log(torch.cat([posterior_var[1:2], posterior_var[1:]])).float(),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            (torch.sqrt(alphas_bar_prev) * betas / (1.0 - alphas_bar)).float(),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (torch.sqrt(alphas) * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)).float(),
        )

    @staticmethod
    def _mask_like(mask: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor:
        if mask is None:
            return torch.ones_like(x)
        while mask.dim() < x.dim():
            mask = mask.unsqueeze(0)
        return mask.to(device=x.device, dtype=x.dtype)

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        mask_t = self._mask_like(mask, x0)
        noise = noise * mask_t
        x_t = (
            extract(self.sqrt_alphas_bar, t, x0.shape).to(x0.device) * x0
            + extract(self.sqrt_one_minus_alphas_bar, t, x0.shape).to(x0.device) * noise
        )
        return x_t * mask_t, noise

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape).to(x_t.device) * x_t
            - extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape).to(x_t.device) * eps
        )

    def q_posterior_mean_variance(
        self,
        x0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape).to(x_t.device) * x0
            + extract(self.posterior_mean_coef2, t, x_t.shape).to(x_t.device) * x_t
        )
        log_var = extract(self.posterior_log_var_clipped, t, x_t.shape).to(x_t.device)
        return mean, log_var

    def mse_loss(
        self,
        predicted_noise: torch.Tensor,
        target_noise: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask_t = self._mask_like(mask, predicted_noise)
        loss = (predicted_noise - target_noise).pow(2) * mask_t
        return loss.sum() / mask_t.sum().clamp_min(1.0)

    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        predicted_noise: torch.Tensor,
        mask: torch.Tensor | None = None,
        clip_x0: bool = True,
    ) -> torch.Tensor:
        mask_t = self._mask_like(mask, x_t)
        x0 = self.predict_x0_from_eps(x_t, t, predicted_noise)
        if clip_x0:
            x0 = x0.clamp(-1.0, 1.0)
        mean, log_var = self.q_posterior_mean_variance(x0, x_t, t)
        noise = torch.randn_like(x_t)
        nonzero = (t > 0).float().view(t.shape[0], *((1,) * (x_t.dim() - 1)))
        sample = mean + nonzero * torch.exp(0.5 * log_var) * noise
        return sample * mask_t
