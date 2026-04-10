# -*- coding: utf-8 -*-
r"""
LightGCN_PDA
################################################

LightGCN with Popularity-bias Deconfounding and Adjusting.
"""

import torch
import torch.nn as nn

from recbole.model.general_recommender.lightgcn import LightGCN
from recbole.model.loss import BPRLoss


class LightGCN_PDA(LightGCN):
    r"""LightGCN + PDA.

    Adds user/item popularity embeddings.
    Training loss = LightGCN loss + pda_weight * popularity BPR loss.
    Inference uses only the original LightGCN (interest) scores.
    """

    def __init__(self, config, dataset):
        super(LightGCN_PDA, self).__init__(config, dataset)

        self.pda_weight = config["pda_weight"]
        d = self.latent_dim

        self.user_pop_embedding = nn.Embedding(self.n_users, d)
        self.item_pop_embedding = nn.Embedding(self.n_items, d)
        self.pop_loss = BPRLoss()

        nn.init.xavier_normal_(self.user_pop_embedding.weight)
        nn.init.xavier_normal_(self.item_pop_embedding.weight)

    def calculate_loss(self, interaction):
        user     = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        # Interest loss (original LightGCN: BPR + reg)
        interest_loss = super().calculate_loss(interaction)

        # Popularity loss
        u_pop   = self.user_pop_embedding(user)
        pi_pop  = self.item_pop_embedding(pos_item)
        ni_pop  = self.item_pop_embedding(neg_item)
        pos_pop = (u_pop * pi_pop).sum(dim=-1)
        neg_pop = (u_pop * ni_pop).sum(dim=-1)
        pop_loss = self.pop_loss(pos_pop, neg_pop)

        return interest_loss + self.pda_weight * pop_loss

    # predict / full_sort_predict inherited from LightGCN — interest only
