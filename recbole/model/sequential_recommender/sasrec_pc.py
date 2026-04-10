# -*- coding: utf-8 -*-
r"""
SASRec_PC
################################################

SASRec with Popularity Compensation post-processing.
"""

import numpy as np
import torch

from recbole.model.sequential_recommender.sasrec import SASRec


class SASRec_PC(SASRec):
    r"""SASRec + Popularity Compensation.

    Training is identical to SASRec.  At inference time:
    ``score' = score - pc_lambda * log(popularity + 1)``
    """

    def __init__(self, config, dataset):
        super(SASRec_PC, self).__init__(config, dataset)

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
        batch_size = scores.shape[0] // self.n_items
        scores = scores.view(batch_size, self.n_items)
        scores = scores - self.pc_lambda * self.log_pop.unsqueeze(0)
        return scores.view(-1)
