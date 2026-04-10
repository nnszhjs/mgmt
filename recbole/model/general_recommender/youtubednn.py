# -*- coding: utf-8 -*-
r"""
YouTubeDNN
################################################

Reference:
    Covington et al. "Deep Neural Networks for YouTube Recommendations." RecSys 2016.

    Candidate generation stage of the YouTube recommendation system.
    User tower: average-pool historical item embeddings, then pass through
    a multi-layer ReLU MLP to produce a user vector.
    Item tower: item embedding lookup.
    Scoring: dot product between user and item vectors.
"""

import torch
import torch.nn as nn

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType


class YouTubeDNN(GeneralRecommender):
    r"""YouTubeDNN: Deep Neural Networks for YouTube Recommendations.

    Two-tower model where the user tower average-pools historical item
    embeddings and feeds them through a multi-layer MLP, while the item
    tower is a simple embedding lookup.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(YouTubeDNN, self).__init__(config, dataset)

        # ---------- load parameters ----------
        self.embedding_size  = config["embedding_size"]
        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.dropout_prob    = config["dropout_prob"]
        d = self.embedding_size

        # ---------- item embedding ----------
        self.item_embedding = nn.Embedding(self.n_items, d, padding_idx=0)

        # ---------- user history ----------
        history_item_id, history_item_value, _ = dataset.history_item_matrix()
        self.register_buffer("history_item_id", history_item_id)
        self.register_buffer("history_item_value", history_item_value)

        # ---------- user tower MLP ----------
        layers = []
        input_size = d
        for i, hidden_size in enumerate(self.mlp_hidden_size):
            layers.append(nn.Linear(input_size, hidden_size))
            if i < len(self.mlp_hidden_size) - 1:
                layers.append(nn.ReLU())
                if self.dropout_prob > 0:
                    layers.append(nn.Dropout(self.dropout_prob))
            input_size = hidden_size
        # Final projection to embedding space
        if input_size != d:
            layers.append(nn.Linear(input_size, d))
        self.user_tower = nn.Sequential(*layers)

        # ---------- loss ----------
        self.loss = BPRLoss()

        # ---------- init ----------
        self.apply(xavier_normal_initialization)

    def _get_user_vector(self, user):
        """Compute user tower output: avg-pool history -> MLP.

        Args:
            user: (B,) user id tensor

        Returns:
            (B, d) user vectors
        """
        history_items = self.history_item_id[user]        # (B, max_hist)
        history_mask  = self.history_item_value[user]      # (B, max_hist)
        history_emb   = self.item_embedding(history_items)  # (B, max_hist, d)
        mask = history_mask.unsqueeze(-1)                  # (B, max_hist, 1)
        sum_emb = (history_emb * mask).sum(dim=1)          # (B, d)
        count   = mask.sum(dim=1).clamp(min=1)             # (B, 1)
        avg_emb = sum_emb / count                          # (B, d)
        return self.user_tower(avg_emb)                    # (B, d)

    def calculate_loss(self, interaction):
        user     = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_vec = self._get_user_vector(user)
        pos_emb  = self.item_embedding(pos_item)
        neg_emb  = self.item_embedding(neg_item)

        pos_score = (user_vec * pos_emb).sum(dim=-1)
        neg_score = (user_vec * neg_emb).sum(dim=-1)
        return self.loss(pos_score, neg_score)

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_vec = self._get_user_vector(user)
        item_emb = self.item_embedding(item)
        return (user_vec * item_emb).sum(dim=-1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_vec = self._get_user_vector(user)       # (B, d)
        all_item_emb = self.item_embedding.weight     # (V, d)
        return torch.matmul(user_vec, all_item_emb.T).view(-1)
