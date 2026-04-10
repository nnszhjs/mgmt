# -*- coding: utf-8 -*-
r"""
LightGCN_PC
################################################

LightGCN with Popularity Compensation post-processing.
"""

import numpy as np
import torch

from recbole.model.general_recommender.lightgcn import LightGCN


class LightGCN_PC(LightGCN):
    r"""LightGCN + Popularity Compensation.

    Training is identical to LightGCN.  At inference time:
    ``score' = score - pc_lambda * log(popularity + 1)``
    """

    def __init__(self, config, dataset):
        super(LightGCN_PC, self).__init__(config, dataset)

        self.pc_lambda = config["pc_lambda"]

        items = np.array(dataset.inter_feat[dataset.iid_field])
        pop = np.bincount(items, minlength=self.n_items).astype(np.float32)
        self.register_buffer("log_pop", torch.from_numpy(np.log(pop + 1.0)))

    def predict(self, interaction):
        scores = super().predict(interaction)
        item = interaction[self.ITEM_ID]
        return scores - self.pc_lambda * self.log_pop[item]

    def full_sort_predict(self, interaction):
        scores = super().full_sort_predict(interaction)
        batch_size = interaction[self.USER_ID].shape[0]
        scores = scores.view(batch_size, self.n_items)
        scores = scores - self.pc_lambda * self.log_pop.unsqueeze(0)
        return scores.view(-1)
