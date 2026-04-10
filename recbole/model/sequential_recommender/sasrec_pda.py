# -*- coding: utf-8 -*-
r"""
SASRec_PDA
################################################

SASRec with Popularity-bias Deconfounding and Adjusting.

Paper footnote 13: "In the SASRec-PDA implementation, we apply causal learning
only to the prevalence of the most recently interacted item in each sequence."

This means the popularity score is derived from the *last interacted item's*
popularity embedding (not a separate user popularity embedding).
"""

import torch
import torch.nn as nn

from recbole.model.sequential_recommender.sasrec import SASRec
from recbole.model.loss import BPRLoss


class SASRec_PDA(SASRec):
    r"""SASRec + PDA.

    Adds item popularity embeddings.  The popularity score for a user-item
    pair is the dot product between the *last interacted item's* popularity
    embedding and the *candidate item's* popularity embedding.

    Training loss = SASRec loss + pda_weight * popularity BPR loss.
    Inference uses only the original SASRec (interest) scores.
    """

    def __init__(self, config, dataset):
        super(SASRec_PDA, self).__init__(config, dataset)

        self.pda_weight = config["pda_weight"]
        d = self.hidden_size

        # Item popularity embeddings (separate from interest item_embedding)
        self.item_pop_embedding = nn.Embedding(self.n_items, d, padding_idx=0)
        self.pop_loss = BPRLoss()

        nn.init.xavier_normal_(self.item_pop_embedding.weight)
        if self.item_pop_embedding.padding_idx is not None:
            self.item_pop_embedding.weight.data[0].fill_(0)

    def _get_last_item_pop(self, item_seq, item_seq_len):
        """Get popularity embedding of the most recently interacted item."""
        # item_seq_len is 1-based; gather the last valid item
        last_item_idx = item_seq_len - 1  # (B,)
        last_items = item_seq.gather(
            1, last_item_idx.unsqueeze(1)
        ).squeeze(1)  # (B,)
        return self.item_pop_embedding(last_items)  # (B, d)

    def calculate_loss(self, interaction):
        item_seq     = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items    = interaction[self.POS_ITEM_ID]

        # Interest loss (original SASRec)
        interest_loss = super().calculate_loss(interaction)

        # Popularity loss: last-item pop ⊙ candidate pop
        last_pop = self._get_last_item_pop(item_seq, item_seq_len)  # (B, d)
        pos_pop  = self.item_pop_embedding(pos_items)  # (B, d)
        pos_pop_score = (last_pop * pos_pop).sum(dim=-1)

        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            neg_pop = self.item_pop_embedding(neg_items)
            neg_pop_score = (last_pop * neg_pop).sum(dim=-1)
            pop_loss = self.pop_loss(pos_pop_score, neg_pop_score)
        else:  # CE
            all_pop_emb = self.item_pop_embedding.weight  # (V, d)
            pop_logits  = torch.matmul(last_pop, all_pop_emb.T)  # (B, V)
            pop_loss    = nn.CrossEntropyLoss()(pop_logits, pos_items)

        return interest_loss + self.pda_weight * pop_loss

    # predict / full_sort_predict inherited from SASRec — interest only
