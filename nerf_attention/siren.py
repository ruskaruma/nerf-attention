"""SIREN: Sinusoidal Representation Network for KV cache compression.

Maps f(position) -> KV_vector via periodic activations (Sitzmann et al., NeurIPS 2020).
fit_siren() is a pure function: tensor + config in, FitResult out. No side effects.
"""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from nerf_attention.types import FitResult, SIRENConfig


class SineLayer(nn.Module):

    def __init__(self, in_features: int, out_features: int,
                 omega_0: float = 30.0, is_first: bool = False):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)

        with torch.no_grad():
            if is_first:
                bound = 1.0 / in_features
            else:
                bound = math.sqrt(6.0 / in_features) / omega_0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):

    def __init__(self, config: SIRENConfig, out_features: int):
        super().__init__()
        self.siren_config = config

        layers: list[nn.Module] = [
            SineLayer(1, config.hidden_features, omega_0=config.omega_0, is_first=True)
        ]
        for _ in range(config.hidden_layers):
            layers.append(
                SineLayer(config.hidden_features, config.hidden_features, omega_0=config.omega_0)
            )

        final = nn.Linear(config.hidden_features, out_features)
        with torch.no_grad():
            bound = math.sqrt(6.0 / config.hidden_features) / config.omega_0
            final.weight.uniform_(-bound, bound)
            final.bias.uniform_(-bound, bound)
        layers.append(final)

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def size_bytes(self) -> int:
        return self.count_parameters() * 4  # SIREN params are float32


def fit_siren(
    kv_tensor: torch.Tensor,
    config: SIRENConfig,
    epochs: int = 5000,
    lr: float = 1e-4,
    device: str = 'cuda',
    log_every: int = 500,
    verbose: bool = True,
) -> FitResult:
    """Fit a SIREN to a single (seq_len, d_head) KV tensor. Pure function."""
    seq_len, d_head = kv_tensor.shape

    positions = torch.linspace(0, 1, seq_len).unsqueeze(1).to(device)
    targets = kv_tensor.to(device)

    target_mean = targets.mean(dim=0, keepdim=True)
    target_std = targets.std(dim=0, keepdim=True).clamp(min=1e-3)
    targets_norm = (targets - target_mean) / target_std

    model = SIREN(config, out_features=d_head).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01,
    )

    losses: list[float] = []
    start = time.time()

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred_norm = model(positions)
        loss = F.mse_loss(pred_norm, targets_norm)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if verbose and (epoch + 1) % log_every == 0:
            with torch.no_grad():
                pred_real = pred_norm * target_std + target_mean
                real_mse = F.mse_loss(pred_real, targets).item()
                cos_sim = F.cosine_similarity(pred_real, targets, dim=1).mean().item()
            print(f"  Epoch {epoch+1}/{epochs} | "
                  f"NormMSE: {loss.item():.6f} | "
                  f"RealMSE: {real_mse:.6f} | "
                  f"CosSim: {cos_sim:.4f}")

    train_time = time.time() - start

    model.eval()
    with torch.no_grad():
        pred_norm = model(positions)
        pred_real = pred_norm * target_std + target_mean
        final_mse = F.mse_loss(pred_real, targets).item()
        cosine_sims = F.cosine_similarity(pred_real, targets, dim=1)
        per_pos_mse = ((pred_real - targets) ** 2).mean(dim=1)

    raw_size = seq_len * d_head * 2  # KV cache is float16 in the model
    siren_size = model.size_bytes()

    return FitResult(
        model=model,
        config=config,
        target_mean=target_mean.cpu(),
        target_std=target_std.cpu(),
        losses=losses,
        final_mse=final_mse,
        final_cosine_mean=cosine_sims.mean().item(),
        final_cosine_min=cosine_sims.min().item(),
        final_cosine_std=cosine_sims.std().item(),
        per_pos_mse=per_pos_mse.cpu().numpy(),
        cosine_sims=cosine_sims.cpu().numpy(),
        compression_ratio=raw_size / siren_size,
        raw_size_bytes=raw_size,
        siren_size_bytes=siren_size,
        train_time_seconds=train_time,
        seq_len=seq_len,
        d_head=d_head,
        num_parameters=model.count_parameters(),
    )
