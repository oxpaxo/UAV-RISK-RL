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


class GpsiNearestKExtractor(BaseFeaturesExtractor):
    """Non-attention ordered top-K obstacle MLP extractor.

    The extractor keeps the repaired Gpsi wrapper schema, but can run as an
    obs-only control by consuming only the first 12 obstacle features.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        hidden_dim: int = 64,
        obs_block_dim: int = 12,
        delta_block_dim: int = 9,
        logvar_block_dim: int = 9,
        feature_mode: str = "gpsi",
        k: int = 6,
        rank_key: str = "risk_ttc_distance",
        rank_eps: float = 1e-6,
    ) -> None:
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError("GpsiNearestKExtractor requires a Dict observation space")
        self.hidden_dim = int(hidden_dim)
        self.obs_block_dim = int(obs_block_dim)
        self.delta_block_dim = int(delta_block_dim)
        self.logvar_block_dim = int(logvar_block_dim)
        self.feature_mode = str(feature_mode)
        self.k = int(k)
        self.rank_key = str(rank_key)
        self.rank_eps = float(rank_eps)
        self.full_obs_dim = self.obs_block_dim + self.delta_block_dim + self.logvar_block_dim
        actual_obs_dim = int(observation_space["obs"].shape[-1])
        if actual_obs_dim != self.full_obs_dim:
            raise ValueError(f"nearest-k extractor expected obs_dim={self.full_obs_dim}, got {actual_obs_dim}")
        if self.feature_mode not in {"obs_only", "gpsi"}:
            raise ValueError("feature_mode must be 'obs_only' or 'gpsi'")
        if self.k <= 0:
            raise ValueError("k must be positive")
        self.input_dim = self.obs_block_dim if self.feature_mode == "obs_only" else self.full_obs_dim
        self.latest_topk_indices: torch.Tensor | None = None
        self.latest_topk_scores: torch.Tensor | None = None
        self.latest_topk_active: torch.Tensor | None = None
        self.latest_selected_features: torch.Tensor | None = None
        self.latest_selected_distance: torch.Tensor | None = None
        self.latest_selected_ttc: torch.Tensor | None = None
        self.latest_selected_risk: torch.Tensor | None = None
        self.latest_flatten_l2: torch.Tensor | None = None
        self.latest_nonfinite_count: torch.Tensor | None = None

        features_dim = 128
        super().__init__(observation_space, features_dim=features_dim)

        self.ego_encoder = nn.Sequential(
            nn.Linear(10, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )
        self.obstacle_mlp = nn.Sequential(
            nn.Linear(self.k * (self.input_dim + 1), 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
        )
        self.global_encoder = nn.Sequential(nn.Linear(2, 16), nn.Tanh())
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim + 128 + 16, features_dim),
            nn.LayerNorm(features_dim),
            nn.Tanh(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        ego = observations["ego"]
        obs = observations["obs"]
        mask = observations["mask"].float()
        global_risk = observations["global_risk"]

        batch_size, max_obs, obs_dim = obs.shape
        if obs_dim != self.full_obs_dim:
            raise RuntimeError(f"nearest-k obs dim mismatch: extractor={self.full_obs_dim}, observation={obs_dim}")
        feature_block = obs[..., : self.input_dim]
        active = mask > 0.0
        score, risk, inv_distance, inv_ttc = self._rank_score(obs, active)
        score = score.masked_fill(~active, -1.0e9)
        k_eff = min(self.k, max_obs)
        top_scores, top_indices = torch.topk(score, k=k_eff, dim=1)
        gather_index = top_indices.unsqueeze(-1).expand(-1, -1, self.input_dim)
        selected = torch.gather(feature_block, 1, gather_index)
        selected_active = torch.gather(mask, 1, top_indices).unsqueeze(-1)
        selected = torch.where(selected_active > 0.0, selected, torch.zeros_like(selected))

        if k_eff < self.k:
            pad_features = torch.zeros(batch_size, self.k - k_eff, self.input_dim, dtype=selected.dtype, device=selected.device)
            pad_active = torch.zeros(batch_size, self.k - k_eff, 1, dtype=selected.dtype, device=selected.device)
            pad_scores = torch.full((batch_size, self.k - k_eff), -1.0e9, dtype=top_scores.dtype, device=top_scores.device)
            pad_indices = torch.full((batch_size, self.k - k_eff), -1, dtype=top_indices.dtype, device=top_indices.device)
            selected = torch.cat([selected, pad_features], dim=1)
            selected_active = torch.cat([selected_active, pad_active], dim=1)
            top_scores = torch.cat([top_scores, pad_scores], dim=1)
            top_indices = torch.cat([top_indices, pad_indices], dim=1)

        slot_features = torch.cat([selected, selected_active], dim=-1)
        flat = slot_features.reshape(batch_size, self.k * (self.input_dim + 1))
        ego_emb = self.ego_encoder(ego)
        obs_emb = self.obstacle_mlp(flat)
        global_emb = self.global_encoder(global_risk)
        self._cache(
            selected=selected,
            selected_active=selected_active,
            top_indices=top_indices,
            top_scores=top_scores,
            flat=flat,
            risk=risk,
            inv_distance=inv_distance,
            inv_ttc=inv_ttc,
            source_features=feature_block,
        )
        return self.fusion(torch.cat([ego_emb, obs_emb, global_emb], dim=-1))

    def _normalize_active(self, values: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        high = torch.full_like(values, 1.0e9)
        low = torch.full_like(values, -1.0e9)
        vmin = torch.where(active, values, high).min(dim=1, keepdim=True).values
        vmax = torch.where(active, values, low).max(dim=1, keepdim=True).values
        scaled = (values - vmin) / (vmax - vmin).clamp(min=self.rank_eps)
        return torch.where(active, scaled.clamp(0.0, 1.0), torch.zeros_like(values))

    def _rank_score(self, obs: torch.Tensor, active: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.rank_key != "risk_ttc_distance":
            raise RuntimeError(f"unsupported rank_key={self.rank_key!r}")
        risk = obs[..., 11].clamp(min=0.0)
        distance_norm = obs[..., 8].abs().clamp(min=self.rank_eps)
        planned_ttc_norm = obs[..., 7].abs().clamp(min=self.rank_eps)
        inv_distance = 1.0 / distance_norm
        inv_ttc = 1.0 / planned_ttc_norm
        score = self._normalize_active(risk, active) + self._normalize_active(inv_distance, active) + self._normalize_active(inv_ttc, active)
        return score, risk, inv_distance, inv_ttc

    def _cache(
        self,
        *,
        selected: torch.Tensor,
        selected_active: torch.Tensor,
        top_indices: torch.Tensor,
        top_scores: torch.Tensor,
        flat: torch.Tensor,
        risk: torch.Tensor,
        inv_distance: torch.Tensor,
        inv_ttc: torch.Tensor,
        source_features: torch.Tensor,
    ) -> None:
        self.latest_topk_indices = top_indices.detach()
        self.latest_topk_scores = top_scores.detach()
        self.latest_topk_active = selected_active.detach()
        self.latest_selected_features = selected.detach()
        self.latest_flatten_l2 = torch.linalg.norm(flat.detach().float(), dim=-1)
        self.latest_nonfinite_count = (~torch.isfinite(source_features.detach())).sum(dim=(1, 2)).float()
        safe_indices = top_indices.clamp(min=0)
        self.latest_selected_risk = torch.gather(risk, 1, safe_indices).detach() * selected_active.squeeze(-1).detach()
        self.latest_selected_distance = torch.gather(1.0 / inv_distance.clamp(min=self.rank_eps), 1, safe_indices).detach() * selected_active.squeeze(-1).detach()
        self.latest_selected_ttc = torch.gather(1.0 / inv_ttc.clamp(min=self.rank_eps), 1, safe_indices).detach() * selected_active.squeeze(-1).detach()


class GpsiDeepSetsExtractor(BaseFeaturesExtractor):
    """Non-attention DeepSets pooling extractor for obstacle sets."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        hidden_dim: int = 64,
        obs_block_dim: int = 12,
        delta_block_dim: int = 9,
        logvar_block_dim: int = 9,
        feature_mode: str = "gpsi",
    ) -> None:
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError("GpsiDeepSetsExtractor requires a Dict observation space")
        self.hidden_dim = int(hidden_dim)
        self.obs_block_dim = int(obs_block_dim)
        self.delta_block_dim = int(delta_block_dim)
        self.logvar_block_dim = int(logvar_block_dim)
        self.feature_mode = str(feature_mode)
        self.full_obs_dim = self.obs_block_dim + self.delta_block_dim + self.logvar_block_dim
        actual_obs_dim = int(observation_space["obs"].shape[-1])
        if actual_obs_dim != self.full_obs_dim:
            raise ValueError(f"DeepSets extractor expected obs_dim={self.full_obs_dim}, got {actual_obs_dim}")
        if self.feature_mode not in {"obs_only", "gpsi"}:
            raise ValueError("feature_mode must be 'obs_only' or 'gpsi'")
        self.input_dim = self.obs_block_dim if self.feature_mode == "obs_only" else self.full_obs_dim
        self.latest_active_count: torch.Tensor | None = None
        self.latest_phi_output: torch.Tensor | None = None
        self.latest_mean_pool: torch.Tensor | None = None
        self.latest_max_pool: torch.Tensor | None = None
        self.latest_pool_l2: torch.Tensor | None = None
        self.latest_nonfinite_count: torch.Tensor | None = None

        features_dim = 128
        super().__init__(observation_space, features_dim=features_dim)

        self.phi = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Tanh(),
        )
        self.ego_encoder = nn.Sequential(
            nn.Linear(10, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )
        self.global_encoder = nn.Sequential(nn.Linear(2, 16), nn.Tanh())
        self.rho = nn.Sequential(
            nn.Linear(self.hidden_dim * 3 + 16, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, features_dim),
            nn.LayerNorm(features_dim),
            nn.Tanh(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        ego = observations["ego"]
        obs = observations["obs"]
        mask = observations["mask"].float()
        global_risk = observations["global_risk"]

        batch_size, max_obs, obs_dim = obs.shape
        if obs_dim != self.full_obs_dim:
            raise RuntimeError(f"DeepSets obs dim mismatch: extractor={self.full_obs_dim}, observation={obs_dim}")
        feature_block = obs[..., : self.input_dim]
        flat = feature_block.reshape(batch_size * max_obs, self.input_dim)
        phi = self.phi(flat).reshape(batch_size, max_obs, self.hidden_dim)
        active = mask.unsqueeze(-1) > 0.0
        masked_phi = torch.where(active, phi, torch.zeros_like(phi))
        active_count = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean_pool = masked_phi.sum(dim=1) / active_count
        max_pool = torch.where(active, phi, torch.full_like(phi, -1.0e9)).max(dim=1).values
        any_active = (mask.sum(dim=1, keepdim=True) > 0.0)
        max_pool = torch.where(any_active, max_pool, torch.zeros_like(max_pool))
        ego_emb = self.ego_encoder(ego)
        global_emb = self.global_encoder(global_risk)
        self._cache(phi=phi, mean_pool=mean_pool, max_pool=max_pool, mask=mask, source_features=feature_block)
        return self.rho(torch.cat([mean_pool, max_pool, ego_emb, global_emb], dim=-1))

    def _cache(
        self,
        *,
        phi: torch.Tensor,
        mean_pool: torch.Tensor,
        max_pool: torch.Tensor,
        mask: torch.Tensor,
        source_features: torch.Tensor,
    ) -> None:
        self.latest_active_count = mask.detach().sum(dim=1)
        self.latest_phi_output = phi.detach()
        self.latest_mean_pool = mean_pool.detach()
        self.latest_max_pool = max_pool.detach()
        self.latest_pool_l2 = torch.linalg.norm(torch.cat([mean_pool.detach(), max_pool.detach()], dim=-1).float(), dim=-1)
        self.latest_nonfinite_count = (~torch.isfinite(source_features.detach())).sum(dim=(1, 2)).float()


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
