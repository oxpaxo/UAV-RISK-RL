from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class GpsiHeadAConfig:
    ego_dim: int = 10
    obs_dim: int = 12
    history_feature_dim: int = 6
    history_hidden_dim: int = 128
    current_hidden_dim: int = 128
    fusion_hidden_dim: int = 128
    z_dim: int = 64
    num_horizons: int = 3
    state_dim: int = 3
    activation: str = "relu"


def activation_layer(name: str) -> nn.Module:
    lowered = name.lower()
    if lowered == "tanh":
        return nn.Tanh()
    if lowered == "gelu":
        return nn.GELU()
    if lowered == "relu":
        return nn.ReLU()
    raise ValueError(f"unsupported activation: {name}")


class GpsiHeadA(nn.Module):
    """Per-obstacle HeadA network.

    Inputs are inference-available fields only:
    ego_current, obs_current, history_rel_pos, history_rel_vel, history_valid_mask.
    Labels and future masks are intentionally not part of the forward signature.
    """

    def __init__(self, config: GpsiHeadAConfig | dict | None = None) -> None:
        super().__init__()
        if config is None:
            config = GpsiHeadAConfig()
        if isinstance(config, dict):
            config = GpsiHeadAConfig(**config)
        self.config = config
        act = activation_layer(config.activation)

        self.history_gru = nn.GRU(
            input_size=config.history_feature_dim,
            hidden_size=config.history_hidden_dim,
            batch_first=True,
        )
        self.current_encoder = nn.Sequential(
            nn.Linear(config.ego_dim + config.obs_dim, config.current_hidden_dim),
            act,
            nn.Linear(config.current_hidden_dim, config.current_hidden_dim),
            activation_layer(config.activation),
        )
        self.fusion = nn.Sequential(
            nn.Linear(config.history_hidden_dim + config.current_hidden_dim, config.fusion_hidden_dim),
            activation_layer(config.activation),
            nn.Linear(config.fusion_hidden_dim, config.z_dim),
            activation_layer(config.activation),
        )
        output_dim = config.num_horizons * config.state_dim
        self.delta_head = nn.Linear(config.z_dim, output_dim)
        self.logvar_head = nn.Linear(config.z_dim, output_dim)

    def forward(
        self,
        ego_current: torch.Tensor,
        obs_current: torch.Tensor,
        history_rel_pos: torch.Tensor,
        history_rel_vel: torch.Tensor,
        history_valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        history = torch.cat([history_rel_pos, history_rel_vel], dim=-1)
        history = history * history_valid_mask.unsqueeze(-1)
        _seq, hidden = self.history_gru(history)
        history_embedding = hidden[-1]
        current_embedding = self.current_encoder(torch.cat([ego_current, obs_current], dim=-1))
        z = self.fusion(torch.cat([history_embedding, current_embedding], dim=-1))
        shape = (-1, self.config.num_horizons, self.config.state_dim)
        delta_hat = self.delta_head(z).view(*shape)
        logvar_hat = self.logvar_head(z).view(*shape)
        return {"z": z, "delta_hat": delta_hat, "logvar_hat": logvar_hat}
