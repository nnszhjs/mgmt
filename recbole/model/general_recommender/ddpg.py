# -*- coding: utf-8 -*-
r"""
DDPG
################################################

Reference:
    Lillicrap et al. "Continuous Control with Deep Reinforcement Learning." ICLR 2016.
    Adapted for recommendation: the actor outputs a continuous action vector in the
    item embedding space, and items are scored by their proximity to this action.

    Key components:
    1. Actor: state -> action (continuous embedding vector), with tanh output.
    2. Critic: (state, action) -> Q-value scalar.
    3. Target networks with soft update (Polyak averaging).
    4. Ornstein-Uhlenbeck noise for exploration.
"""

import copy

import torch
import torch.nn as nn

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType


class DDPG(GeneralRecommender):
    r"""DDPG: Deep Deterministic Policy Gradient for Recommendation.

    State  = average-pooled historical item embeddings for a user.
    Action = continuous embedding vector produced by the actor network.
    Score  = dot product between the action and candidate item embeddings.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(DDPG, self).__init__(config, dataset)

        # ---------- load parameters ----------
        self.embedding_size     = config["embedding_size"]
        self.actor_hidden_size  = config["actor_hidden_size"]
        self.critic_hidden_size = config["critic_hidden_size"]
        self.tau                = config["tau"]
        self.ou_sigma           = config["ou_sigma"]
        self.ou_theta           = config["ou_theta"]
        self.gamma              = config["gamma"]
        d = self.embedding_size

        # ---------- item embedding (shared) ----------
        self.item_embedding = nn.Embedding(self.n_items, d, padding_idx=0)

        # ---------- user history ----------
        history_item_id, history_item_value, _ = dataset.history_item_matrix()
        self.register_buffer("history_item_id", history_item_id)
        self.register_buffer("history_item_value", history_item_value)

        # ---------- actor ----------
        self.actor = nn.Sequential(
            nn.Linear(d, self.actor_hidden_size),
            nn.ReLU(),
            nn.Linear(self.actor_hidden_size, self.actor_hidden_size),
            nn.ReLU(),
            nn.Linear(self.actor_hidden_size, d),
            nn.Tanh(),
        )

        # ---------- critic ----------
        self.critic = nn.Sequential(
            nn.Linear(2 * d, self.critic_hidden_size),
            nn.ReLU(),
            nn.Linear(self.critic_hidden_size, self.critic_hidden_size),
            nn.ReLU(),
            nn.Linear(self.critic_hidden_size, 1),
        )

        # ---------- target networks ----------
        self.apply(xavier_normal_initialization)
        self.target_actor  = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        # Freeze target parameters (not updated by gradient)
        for p in self.target_actor.parameters():
            p.requires_grad = False
        for p in self.target_critic.parameters():
            p.requires_grad = False

        # ---------- OU noise ----------
        # No persistent state — noise is sampled fresh each call to avoid
        # shape mismatches when batch size varies across iterations.

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_state(self, user):
        """Average-pooled item history embeddings as user state.

        Args:
            user: (B,) user id tensor

        Returns:
            (B, d) state vectors
        """
        history_items = self.history_item_id[user]       # (B, max_hist)
        history_mask  = self.history_item_value[user]     # (B, max_hist)
        history_emb   = self.item_embedding(history_items)  # (B, max_hist, d)
        mask = history_mask.unsqueeze(-1)                 # (B, max_hist, 1)
        sum_emb = (history_emb * mask).sum(dim=1)         # (B, d)
        count   = mask.sum(dim=1).clamp(min=1)            # (B, 1)
        return sum_emb / count

    def _ou_noise(self, shape):
        """Sample Ornstein-Uhlenbeck-style exploration noise (stateless).

        Uses ou_theta for scale damping: noise ~ N(0, sigma / (1 + theta)).
        """
        scale = self.ou_sigma / (1.0 + self.ou_theta)
        return torch.randn(shape, device=self.item_embedding.weight.device) * scale

    def _critic_no_grad(self, x):
        """Forward through critic without propagating gradients to critic params."""
        for param in self.critic.parameters():
            param.requires_grad_(False)
        out = self.critic(x)
        for param in self.critic.parameters():
            param.requires_grad_(True)
        return out

    def _soft_update(self):
        """Polyak-average update of target networks."""
        for tp, sp in zip(self.target_actor.parameters(), self.actor.parameters()):
            tp.data.mul_(1.0 - self.tau).add_(sp.data, alpha=self.tau)
        for tp, sp in zip(self.target_critic.parameters(), self.critic.parameters()):
            tp.data.mul_(1.0 - self.tau).add_(sp.data, alpha=self.tau)

    # ------------------------------------------------------------------
    # RecBole required methods
    # ------------------------------------------------------------------

    def calculate_loss(self, interaction):
        user     = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        state = self._get_state(user)           # (B, d)

        pos_emb = self.item_embedding(pos_item)  # (B, d)
        neg_emb = self.item_embedding(neg_item)  # (B, d)

        # ---- Critic loss (TD-style) ----
        # In the recommendation setting there is no "next state" transition,
        # so we use single-step targets: r=1 for positive, r=0 for negative.
        # Target Q-values are computed via the target critic for stability.
        with torch.no_grad():
            q_target_pos = self.target_critic(
                torch.cat([state, pos_emb], dim=-1)
            ).squeeze(-1)
            q_target_neg = self.target_critic(
                torch.cat([state, neg_emb], dim=-1)
            ).squeeze(-1)
            # TD targets: reward + gamma * Q_target(s, a_target)
            # With gamma=0 (no next-state bootstrap), targets are just rewards.
            # We blend with target Q for training stability.
            td_target_pos = 1.0 + self.gamma * q_target_pos
            td_target_neg = 0.0 + self.gamma * q_target_neg

        q_pos = self.critic(torch.cat([state.detach(), pos_emb.detach()], dim=-1)).squeeze(-1)
        q_neg = self.critic(torch.cat([state.detach(), neg_emb.detach()], dim=-1)).squeeze(-1)
        critic_loss = (
            nn.functional.mse_loss(q_pos, td_target_pos)
            + nn.functional.mse_loss(q_neg, td_target_neg)
        )

        # ---- Actor loss: maximise Q(s, actor(s)) ----
        # Detach critic so its parameters are not updated by actor gradients.
        action = self.actor(state)
        # Add OU noise during training for exploration
        if self.training:
            noise = self._ou_noise(action.shape)
            action_noisy = action + noise
        else:
            action_noisy = action
        q_actor = self._critic_no_grad(
            torch.cat([state, action_noisy], dim=-1)
        ).squeeze(-1)
        actor_loss = -q_actor.mean()

        # ---- Soft-update target networks ----
        self._soft_update()

        return critic_loss + actor_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        state = self._get_state(user)
        action = self.actor(state)
        item_emb = self.item_embedding(item)
        return (action * item_emb).sum(dim=-1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        state = self._get_state(user)
        action = self.actor(state)               # (B, d)
        all_item_emb = self.item_embedding.weight  # (V, d)
        return torch.matmul(action, all_item_emb.T).view(-1)
