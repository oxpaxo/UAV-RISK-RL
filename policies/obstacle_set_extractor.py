from __future__ import annotations

import math

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class ObstacleSetExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        agg_mode: str = "risk",
        hidden_dim: int = 64,
        beta: float = 1.0,
        r_ref: float = 1.0,
        use_rbar: bool = True,
        rbar_floor: float = 0.0,
        use_risk_bias: bool = False,
        lambda_bias: float = 0.2,
        risk_bias_eps: float = 1e-6,
    ) -> None:
        if agg_mode not in {"risk", "attention", "mean"}:
            raise ValueError(f"unsupported agg_mode: {agg_mode}")
        self.hidden_dim = int(hidden_dim)
        self.agg_mode = agg_mode
        self.beta = float(beta)
        self.r_ref = float(r_ref)
        self.use_rbar = bool(use_rbar)
        self.rbar_floor = float(rbar_floor)
        self.use_risk_bias = bool(use_risk_bias)
        self.lambda_bias = float(lambda_bias)
        self.risk_bias_eps = float(risk_bias_eps)
        self.latest_attention_weights: torch.Tensor | None = None
        self.latest_attention_scores: torch.Tensor | None = None
        self.latest_attention_entropy: torch.Tensor | None = None

        features_dim = self.hidden_dim * 2 + 2
        super().__init__(observation_space, features_dim=features_dim)

        self.obs_encoder = nn.Sequential(
            nn.Linear(12, self.hidden_dim),
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

        if self.agg_mode == "attention":
            self.W_q = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.W_k = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        ego = observations["ego"]
        obs = observations["obs"]
        mask = observations["mask"]
        global_risk = observations["global_risk"]

        batch_size, max_obs, obs_dim = obs.shape
        h = self.obs_encoder(obs.reshape(batch_size * max_obs, obs_dim)).reshape(batch_size, max_obs, self.hidden_dim)
        ego_emb = self.ego_encoder(ego)

        if self.agg_mode == "risk":
            risk = obs[..., -1]
            weights = (risk.clamp(min=0.0) ** self.beta) * mask
            weight_sum = weights.sum(dim=-1, keepdim=True)
            fallback = (mask / mask.sum(dim=-1, keepdim=True).clamp(min=1.0)).to(weights.dtype)
            weights = torch.where(weight_sum > 1e-8, weights / weight_sum.clamp(min=1e-8), fallback)
            self._cache_weights(weights=weights, scores=None)
            context = (weights.unsqueeze(-1) * h).sum(dim=1)
            if self.use_rbar:
                R_sum = global_risk[:, 1:2]
                R_bar = torch.tanh(R_sum / self.r_ref)
                if self.rbar_floor > 0.0:
                    R_bar = torch.clamp(R_bar, min=self.rbar_floor)
                context = R_bar * context
        elif self.agg_mode == "attention":
            query = self.W_q(ego_emb).unsqueeze(1)
            keys = self.W_k(h)
            scores = (query * keys).sum(dim=-1) / math.sqrt(self.hidden_dim)
            if self.use_risk_bias:
                risk = obs[..., -1].clamp(min=0.0)
                scores = scores + self.lambda_bias * torch.log(risk + self.risk_bias_eps)
            scores = scores.masked_fill(mask <= 0.0, -1e9)
            weights = torch.softmax(scores, dim=-1)
            self._cache_weights(weights=weights, scores=scores)
            context = (weights.unsqueeze(-1) * h).sum(dim=1)
        else:
            weights = mask / mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            self._cache_weights(weights=weights, scores=None)
            context = (weights.unsqueeze(-1) * h).sum(dim=1)

        return torch.cat([ego_emb, context, global_risk], dim=-1)

    def _cache_weights(self, weights: torch.Tensor, scores: torch.Tensor | None) -> None:
        detached_weights = weights.detach()
        self.latest_attention_weights = detached_weights
        self.latest_attention_scores = scores.detach() if scores is not None else None
        entropy = -(detached_weights * torch.log(detached_weights.clamp(min=1e-8))).sum(dim=-1)
        self.latest_attention_entropy = entropy.detach()
