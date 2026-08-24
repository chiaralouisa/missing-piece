"""Network blocks and training loops shared by the baselines and flow models."""

from __future__ import annotations

import copy
import math
from typing import Sequence

import numpy as np
import torch
from torch import nn

__all__ = [
    "ResidualMLP",
    "TimeConditionedMLP",
    "SinusoidalTimeEmbedding",
    "train_supervised",
]


class SinusoidalTimeEmbedding(nn.Module):
    """Standard transformer-style embedding of the continuous flow time t."""

    def __init__(self, dim: int = 128):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("time embedding dim must be even")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.reshape(-1).float()
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device).float() / half
        )
        args = t[:, None] * freqs[None, :] * 1000.0
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class _ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float, cond_dim: int = 0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        # FiLM conditioning: the time/context embedding modulates the block
        # rather than being concatenated once at the input, which keeps the
        # time signal available at every depth.
        self.film = nn.Linear(cond_dim, 2 * dim) if cond_dim else None

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm(x)
        if self.film is not None and cond is not None:
            scale, shift = self.film(cond).chunk(2, dim=-1)
            h = h * (1 + scale) + shift
        h = self.drop(self.act(self.fc1(h)))
        h = self.fc2(h)
        return x + h


class ResidualMLP(nn.Module):
    """Plain feed-forward trunk with residual blocks, for discriminative use."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Sequence[int] = (512, 256),
        dropout: float = 0.2,
        output_bias_init: np.ndarray | None = None,
    ):
        super().__init__()
        width = hidden[0]
        layers: list[nn.Module] = [nn.Linear(in_dim, width), nn.SiLU()]
        for h in hidden[1:]:
            layers += [nn.Linear(width, h), nn.SiLU(), nn.Dropout(dropout)]
            width = h
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(width, out_dim)
        if output_bias_init is not None:
            with torch.no_grad():
                self.head.bias.copy_(torch.as_tensor(output_bias_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))


class TimeConditionedMLP(nn.Module):
    """Velocity / denoiser network v(x_t, t, c) for the flow models.

    ``x_t`` is the current state of the target block, ``c`` the conditioning
    vector from the observed panel, and ``t`` the flow time. Conditioning enters
    through FiLM in every residual block.
    """

    def __init__(
        self,
        state_dim: int,
        cond_dim: int,
        width: int = 512,
        depth: int = 4,
        dropout: float = 0.1,
        time_dim: int = 128,
        out_dim: int | None = None,
        output_bias_init: np.ndarray | None = None,
    ):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.state_proj = nn.Linear(state_dim, width)
        self.blocks = nn.ModuleList(
            [_ResBlock(width, dropout, cond_dim=width) for _ in range(depth)]
        )
        self.norm_out = nn.LayerNorm(width)
        self.head = nn.Linear(width, out_dim or state_dim)
        if output_bias_init is not None:
            with torch.no_grad():
                self.head.bias.copy_(
                    torch.as_tensor(output_bias_init, dtype=torch.float32)
                )

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        c = self.cond_proj(cond) + self.time_proj(self.time_embed(t))
        h = self.state_proj(x_t)
        for block in self.blocks:
            h = block(h, c)
        return self.head(self.norm_out(h))


def train_supervised(
    net: nn.Module,
    X: np.ndarray,
    Y: np.ndarray,
    val: tuple[np.ndarray, np.ndarray] | None = None,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 200,
    batch_size: int = 128,
    patience: int = 20,
    device: str = "cpu",
    verbose: bool = False,
) -> nn.Module:
    """Train a multi-label classifier with BCE and early stopping on val loss."""
    x = torch.as_tensor(X, dtype=torch.float32, device=device)
    y = torch.as_tensor(Y, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    if val is not None:
        xv = torch.as_tensor(val[0], dtype=torch.float32, device=device)
        yv = torch.as_tensor(val[1], dtype=torch.float32, device=device)

    best_loss = float("inf")
    best_state = copy.deepcopy(net.state_dict())
    bad_epochs = 0
    n = x.shape[0]

    for epoch in range(epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(net(x[idx]), y[idx])
            loss.backward()
            opt.step()

        if val is None:
            continue
        net.eval()
        with torch.no_grad():
            v = float(loss_fn(net(xv), yv))
        if v < best_loss - 1e-6:
            best_loss, best_state, bad_epochs = v, copy.deepcopy(net.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f"early stop at epoch {epoch} (val loss {best_loss:.5f})")
                break

    if val is not None:
        net.load_state_dict(best_state)
    return net
