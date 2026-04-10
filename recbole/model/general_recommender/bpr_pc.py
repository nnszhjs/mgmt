# -*- coding: utf-8 -*-
r"""
BPR_PC
################################################

BPR with Popularity Compensation post-processing.
Inherits BPR and subtracts ``lambda * log(pop + 1)`` at inference time.
"""

import numpy as np
import torch

from recbole.model.general_recommender.bpr import BPR


class BPR_PC(BPR):
    r"""BPR + Popularity Compensation.

    Training is identical to BPR.  At inference time, item scores are adjusted:
    ``score' = score - pc_lambda * log(popularity + 1)``
    """

    def __init__(self, config, dataset):
        super(BPR_PC, self).__init__(config, dataset)

        self.pc_lambda = config["pc_lambda"]

        # Pre-compute log-popularity from training interactions
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
