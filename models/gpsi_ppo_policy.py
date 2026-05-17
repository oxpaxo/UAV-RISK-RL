from __future__ import annotations

import math

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class GpsiObstacleSetExtractor(BaseFeaturesExtractor):
    """Masked-attention extractor for Gpsi-augmented obstacle observations."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        hidden_dim: int = 64,
        obs_dim: int | None = None,
        use_risk_bias: bool = False,
        lambda_bias: float = 0.0,
        risk_bias_eps: float = 1e-6,
    ) -> None:
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError("GpsiObstacleSetExtractor requires a Dict observation space")
        self.hidden_dim = int(hidden_dim)
        self.obs_dim = int(obs_dim or observation_space["obs"].shape[-1])
        self.use_risk_bias = bool(use_risk_bias)
        self.lambda_bias = float(lambda_bias)
        self.risk_bias_eps = float(risk_bias_eps)
        self.latest_attention_weights: torch.Tensor | None = None
        self.latest_attention_scores: torch.Tensor | None = None
        self.latest_attention_entropy: torch.Tensor | None = None

        features_dim = self.hidden_dim * 2 + 2
        super().__init__(observation_space, features_dim=features_dim)

        self.obs_encoder = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )
        self.ego_encoder = nn.Sequential(
            nn.Linear(10, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )
        self.W_q = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.W_k = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        ego = observations["ego"]
        obs = observations["obs"]
        mask = observations["mask"]
        global_risk = observations["global_risk"]

        batch_size, max_obs, obs_dim = obs.shape
        if obs_dim != self.obs_dim:
            raise RuntimeError(f"Gpsi obs dim mismatch: extractor={self.obs_dim}, observation={obs_dim}")
        h = self.obs_encoder(obs.reshape(batch_size * max_obs, obs_dim)).reshape(batch_size, max_obs, self.hidden_dim)
        ego_emb = self.ego_encoder(ego)
        query = self.W_q(ego_emb).unsqueeze(1)
        keys = self.W_k(h)
        scores = (query * keys).sum(dim=-1) / math.sqrt(self.hidden_dim)
        if self.use_risk_bias:
            risk = obs[..., 11].clamp(min=0.0)
            scores = scores + self.lambda_bias * torch.log(risk + self.risk_bias_eps)
        scores = scores.masked_fill(mask <= 0.0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        self._cache_weights(weights=weights, scores=scores)
        context = (weights.unsqueeze(-1) * h).sum(dim=1)
        return torch.cat([ego_emb, context, global_risk], dim=-1)

    def _cache_weights(self, weights: torch.Tensor, scores: torch.Tensor) -> None:
        detached_weights = weights.detach()
        self.latest_attention_weights = detached_weights
        self.latest_attention_scores = scores.detach()
        entropy = -(detached_weights * torch.log(detached_weights.clamp(min=1e-8))).sum(dim=-1)
        self.latest_attention_entropy = entropy.detach()


class GpsiBlockProjectedNoZExtractor(BaseFeaturesExtractor):
    """Block-wise no-z extractor for [obs_i, delta_hat_scaled, logvar_scaled]."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        hidden_dim: int = 64,
        obs_block_dim: int = 12,
        delta_block_dim: int = 9,
        logvar_block_dim: int = 9,
        obs_project_dim: int = 32,
        delta_project_dim: int = 16,
        logvar_project_dim: int = 16,
        activation: str = "tanh",
        use_risk_bias: bool = False,
        lambda_bias: float = 0.0,
        risk_bias_eps: float = 1e-6,
    ) -> None:
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError("GpsiBlockProjectedNoZExtractor requires a Dict observation space")
        self.hidden_dim = int(hidden_dim)
        self.obs_block_dim = int(obs_block_dim)
        self.delta_block_dim = int(delta_block_dim)
        self.logvar_block_dim = int(logvar_block_dim)
        self.projected_dim = int(obs_project_dim) + int(delta_project_dim) + int(logvar_project_dim)
        self.obs_dim = self.obs_block_dim + self.delta_block_dim + self.logvar_block_dim
        actual_obs_dim = int(observation_space["obs"].shape[-1])
        if actual_obs_dim != self.obs_dim:
            raise ValueError(f"block-projected no-z extractor expected obs_dim={self.obs_dim}, got {actual_obs_dim}")
        self.use_risk_bias = bool(use_risk_bias)
        self.lambda_bias = float(lambda_bias)
        self.risk_bias_eps = float(risk_bias_eps)
        self.latest_attention_weights: torch.Tensor | None = None
        self.latest_attention_scores: torch.Tensor | None = None
        self.latest_attention_entropy: torch.Tensor | None = None
        self.latest_adapter_output: torch.Tensor | None = None

        features_dim = self.hidden_dim * 2 + 2
        super().__init__(observation_space, features_dim=features_dim)

        if str(activation).lower() == "relu":
            activation_layer: type[nn.Module] = nn.ReLU
        elif str(activation).lower() == "gelu":
            activation_layer = nn.GELU
        else:
            activation_layer = nn.Tanh

        self.obs_projector = nn.Sequential(
            nn.Linear(self.obs_block_dim, int(obs_project_dim)),
            nn.LayerNorm(int(obs_project_dim)),
            activation_layer(),
        )
        self.delta_projector = nn.Sequential(
            nn.Linear(self.delta_block_dim, int(delta_project_dim)),
            nn.LayerNorm(int(delta_project_dim)),
            activation_layer(),
        )
        self.logvar_projector = nn.Sequential(
            nn.Linear(self.logvar_block_dim, int(logvar_project_dim)),
            nn.LayerNorm(int(logvar_project_dim)),
            activation_layer(),
        )
        if self.projected_dim == self.hidden_dim:
            self.block_fusion: nn.Module = nn.Identity()
        else:
            self.block_fusion = nn.Sequential(
                nn.Linear(self.projected_dim, self.hidden_dim),
                nn.Tanh(),
            )
        self.ego_encoder = nn.Sequential(
            nn.Linear(10, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )
        self.W_q = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.W_k = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        ego = observations["ego"]
        obs = observations["obs"]
        mask = observations["mask"]
        global_risk = observations["global_risk"]

        batch_size, max_obs, obs_dim = obs.shape
        if obs_dim != self.obs_dim:
            raise RuntimeError(f"Gpsi block obs dim mismatch: extractor={self.obs_dim}, observation={obs_dim}")
        flat = obs.reshape(batch_size * max_obs, obs_dim)
        obs_block = flat[:, : self.obs_block_dim]
        delta_start = self.obs_block_dim
        logvar_start = delta_start + self.delta_block_dim
        delta_block = flat[:, delta_start:logvar_start]
        logvar_block = flat[:, logvar_start : logvar_start + self.logvar_block_dim]
        projected = torch.cat(
            [
                self.obs_projector(obs_block),
                self.delta_projector(delta_block),
                self.logvar_projector(logvar_block),
            ],
            dim=-1,
        )
        self.latest_adapter_output = projected.detach().reshape(batch_size, max_obs, self.projected_dim)
        h = self.block_fusion(projected).reshape(batch_size, max_obs, self.hidden_dim)
        ego_emb = self.ego_encoder(ego)
        query = self.W_q(ego_emb).unsqueeze(1)
        keys = self.W_k(h)
        scores = (query * keys).sum(dim=-1) / math.sqrt(self.hidden_dim)
        if self.use_risk_bias:
            risk = obs[..., 11].clamp(min=0.0)
            scores = scores + self.lambda_bias * torch.log(risk + self.risk_bias_eps)
        scores = scores.masked_fill(mask <= 0.0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        self._cache_weights(weights=weights, scores=scores)
        context = (weights.unsqueeze(-1) * h).sum(dim=1)
        return torch.cat([ego_emb, context, global_risk], dim=-1)

    def _cache_weights(self, weights: torch.Tensor, scores: torch.Tensor) -> None:
        detached_weights = weights.detach()
        self.latest_attention_weights = detached_weights
        self.latest_attention_scores = scores.detach()
        entropy = -(detached_weights * torch.log(detached_weights.clamp(min=1e-8))).sum(dim=-1)
        self.latest_attention_entropy = entropy.detach()


class GpsiGatedResidualExtractor(BaseFeaturesExtractor):
    """Attention-like gated residual extractor for [obs_i, delta_hat_scaled, logvar_scaled].

    This class intentionally does not claim a trained attention_full fallback by
    itself. Unless a caller explicitly warm-starts or distills the base branch,
    gate ~= 0 falls back to a randomly initialized attention-like obs branch.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        hidden_dim: int = 64,
        obs_block_dim: int = 12,
        delta_block_dim: int = 9,
        logvar_block_dim: int = 9,
        gpsi_hidden_dim: int = 64,
        gate_init_logit: float = -5.0,
        activation: str = "tanh",
        use_risk_bias: bool = False,
        lambda_bias: float = 0.0,
        risk_bias_eps: float = 1e-6,
    ) -> None:
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError("GpsiGatedResidualExtractor requires a Dict observation space")
        self.hidden_dim = int(hidden_dim)
        self.obs_block_dim = int(obs_block_dim)
        self.delta_block_dim = int(delta_block_dim)
        self.logvar_block_dim = int(logvar_block_dim)
        self.obs_dim = self.obs_block_dim + self.delta_block_dim + self.logvar_block_dim
        actual_obs_dim = int(observation_space["obs"].shape[-1])
        if actual_obs_dim != self.obs_dim:
            raise ValueError(f"gated residual extractor expected obs_dim={self.obs_dim}, got {actual_obs_dim}")
        self.use_risk_bias = bool(use_risk_bias)
        self.lambda_bias = float(lambda_bias)
        self.risk_bias_eps = float(risk_bias_eps)
        self.latest_attention_weights: torch.Tensor | None = None
        self.latest_attention_scores: torch.Tensor | None = None
        self.latest_attention_entropy: torch.Tensor | None = None
        self.latest_adapter_output: torch.Tensor | None = None
        self.latest_base_output: torch.Tensor | None = None
        self.latest_gpsi_output: torch.Tensor | None = None
        self.latest_gated_contribution: torch.Tensor | None = None
        self.latest_gate_value: torch.Tensor | None = None

        features_dim = self.hidden_dim * 2 + 2
        super().__init__(observation_space, features_dim=features_dim)

        if str(activation).lower() == "relu":
            activation_layer: type[nn.Module] = nn.ReLU
        elif str(activation).lower() == "gelu":
            activation_layer = nn.GELU
        else:
            activation_layer = nn.Tanh

        self.base_obs_encoder = nn.Sequential(
            nn.Linear(self.obs_block_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )
        self.gpsi_encoder = nn.Sequential(
            nn.Linear(self.delta_block_dim + self.logvar_block_dim, int(gpsi_hidden_dim)),
            nn.LayerNorm(int(gpsi_hidden_dim)),
            activation_layer(),
            nn.Linear(int(gpsi_hidden_dim), self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            activation_layer(),
        )
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init_logit), dtype=torch.float32))
        self.ego_encoder = nn.Sequential(
            nn.Linear(10, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )
        self.W_q = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.W_k = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        ego = observations["ego"]
        obs = observations["obs"]
        mask = observations["mask"]
        global_risk = observations["global_risk"]

        batch_size, max_obs, obs_dim = obs.shape
        if obs_dim != self.obs_dim:
            raise RuntimeError(f"Gpsi gated residual obs dim mismatch: extractor={self.obs_dim}, observation={obs_dim}")
        flat = obs.reshape(batch_size * max_obs, obs_dim)
        obs_block = flat[:, : self.obs_block_dim]
        delta_start = self.obs_block_dim
        logvar_start = delta_start + self.delta_block_dim
        delta_block = flat[:, delta_start:logvar_start]
        logvar_block = flat[:, logvar_start : logvar_start + self.logvar_block_dim]

        base_h = self.base_obs_encoder(obs_block)
        gpsi_h = self.gpsi_encoder(torch.cat([delta_block, logvar_block], dim=-1))
        gate = torch.sigmoid(self.gate_logit)
        gated = gate * gpsi_h
        h_flat = base_h + gated
        h = h_flat.reshape(batch_size, max_obs, self.hidden_dim)

        self.latest_base_output = base_h.detach().reshape(batch_size, max_obs, self.hidden_dim)
        self.latest_gpsi_output = gpsi_h.detach().reshape(batch_size, max_obs, self.hidden_dim)
        self.latest_gated_contribution = gated.detach().reshape(batch_size, max_obs, self.hidden_dim)
        self.latest_adapter_output = h.detach()
        self.latest_gate_value = gate.detach().reshape(1)

        ego_emb = self.ego_encoder(ego)
        query = self.W_q(ego_emb).unsqueeze(1)
        keys = self.W_k(h)
        scores = (query * keys).sum(dim=-1) / math.sqrt(self.hidden_dim)
        if self.use_risk_bias:
            risk = obs[..., 11].clamp(min=0.0)
            scores = scores + self.lambda_bias * torch.log(risk + self.risk_bias_eps)
        scores = scores.masked_fill(mask <= 0.0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        self._cache_weights(weights=weights, scores=scores)
        context = (weights.unsqueeze(-1) * h).sum(dim=1)
        return torch.cat([ego_emb, context, global_risk], dim=-1)

    def gated_debug(self) -> dict[str, float]:
        def mean_l2(tensor: torch.Tensor | None) -> float:
            if tensor is None:
                return float("nan")
            arr = tensor.detach().float().reshape(-1, tensor.shape[-1])
            if arr.numel() == 0:
                return float("nan")
            return float(torch.linalg.norm(arr, dim=-1).mean().cpu())

        return {
            "gate_value": float(torch.sigmoid(self.gate_logit.detach()).cpu()),
            "base_branch_l2": mean_l2(self.latest_base_output),
            "gpsi_branch_l2": mean_l2(self.latest_gpsi_output),
            "gated_contribution_l2": mean_l2(self.latest_gated_contribution),
            "fused_obs_emb_l2": mean_l2(self.latest_adapter_output),
        }

    def _cache_weights(self, weights: torch.Tensor, scores: torch.Tensor) -> None:
        detached_weights = weights.detach()
        self.latest_attention_weights = detached_weights
        self.latest_attention_scores = scores.detach()
        entropy = -(detached_weights * torch.log(detached_weights.clamp(min=1e-8))).sum(dim=-1)
        self.latest_attention_entropy = entropy.detach()
