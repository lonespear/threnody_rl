"""Shared MLP policy with masked discrete action head + value head.

Single shared policy across all 25 faction matchups; the my-faction and
their-faction one-hots in the observation let it specialize per matchup.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


HIDDEN = 512


class MaskedActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = HIDDEN):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.policy_head = nn.Linear(hidden, action_dim)
        self.value_head  = nn.Linear(hidden, 1)

        # Orthogonal init — standard PPO trick
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value

    def act(self, obs: torch.Tensor, mask: torch.Tensor,
            deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample masked action.
        Returns (action, log_prob, value)."""
        logits, value = self(obs)
        masked_logits = logits.masked_fill(mask == 0, -1e9)
        dist = torch.distributions.Categorical(logits=masked_logits)
        action = masked_logits.argmax(dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate(self, obs: torch.Tensor, mask: torch.Tensor,
                 actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-compute log_prob / entropy / value for stored actions (PPO update)."""
        logits, value = self(obs)
        masked_logits = logits.masked_fill(mask == 0, -1e9)
        dist = torch.distributions.Categorical(logits=masked_logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, value
