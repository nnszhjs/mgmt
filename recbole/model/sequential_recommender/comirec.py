# -*- coding: utf-8 -*-
# @Time   : 2026/04/09
# @Author : RecBole Contributors

r"""
ComiRec-DR
################################################

Reference:
    Yukuo Cen et al. "Controllable Multi-Interest Framework for Recommendation." in KDD 2020.

This implements the Dynamic Routing (DR) variant, which uses a capsule network
with iterative routing to extract multiple interest vectors from a user's
interaction sequence.
"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, xavier_uniform_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from recbole.utils import InputType


class ComiRec(SequentialRecommender):
    r"""ComiRec-DR uses a dynamic routing capsule network to extract multiple
    interest representations from a user's historical interaction sequence.

    At each routing iteration, input item embeddings are projected into K
    capsule spaces via a shared bilinear map, and routing weights are refined
    by measuring agreement between predicted and actual output capsules.
    The final K interest vectors are used to score candidate items via a
    max-pooling mechanism over interests.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(ComiRec, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.n_interests = config["n_interests"]
        self.n_iterations = config["n_iterations"]
        self.loss_type = config["loss_type"]

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.embedding_size, padding_idx=0
        )

        # bilinear map for projecting input capsules into K output capsule spaces
        # shape: (embedding_size, n_interests, embedding_size)
        self.bilinear_map = nn.Parameter(
            torch.empty(self.embedding_size, self.n_interests, self.embedding_size)
        )

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        if hasattr(self, "bilinear_map") and module is self:
            xavier_uniform_(self.bilinear_map.data)

    def _squash(self, s):
        """Non-linear squash activation for capsule vectors.

        Args:
            s (torch.Tensor): Input tensor of any shape, squash applied on last dim.

        Returns:
            torch.Tensor: Squashed tensor with the same shape as input.
        """
        s_norm_sq = (s * s).sum(dim=-1, keepdim=True)  # ||s||^2
        s_norm = torch.sqrt(s_norm_sq + 1e-8)  # ||s||
        scale = s_norm_sq / (1.0 + s_norm_sq)  # ||s||^2 / (1 + ||s||^2)
        direction = s / s_norm  # s / ||s||
        return scale * direction

    def _dynamic_routing(self, item_seq_emb, mask):
        """Dynamic routing algorithm to extract K interest capsules.

        Args:
            item_seq_emb (torch.Tensor): Item sequence embeddings, shape (B, L, d).
            mask (torch.Tensor): Boolean mask where True indicates valid (non-padding)
                positions, shape (B, L).

        Returns:
            torch.Tensor: K interest capsule vectors, shape (B, K, d).
        """
        # item_seq_emb: (B, L, d)
        # bilinear_map: (d, K, d)
        # u_hat: (B, L, K, d) - predicted output vectors
        u_hat = torch.einsum("bld,dkh->blkh", item_seq_emb, self.bilinear_map)

        B, L, K, d = u_hat.shape

        # routing logits: (B, L, K)
        b = torch.zeros(B, L, K, device=item_seq_emb.device)

        # mask for padding positions: (B, L) -> (B, L, 1) for broadcasting
        # set routing logits at padding positions to -inf so softmax yields 0
        mask_expanded = (~mask).unsqueeze(-1)  # True where padding

        for i in range(self.n_iterations):
            # mask padding positions before softmax
            b_masked = b.masked_fill(mask_expanded, float("-inf"))
            c = torch.softmax(b_masked, dim=-1)  # (B, L, K)
            # handle fully-padded sequences: softmax(-inf) -> nan, replace with 0
            c = torch.nan_to_num(c, nan=0.0)

            # weighted sum: (B, K, d)
            z = torch.einsum("blk,blkd->bkd", c, u_hat)
            z_hat = self._squash(z)

            if i < self.n_iterations - 1:
                # update routing logits by agreement
                b = b + torch.einsum("blkd,bkd->blk", u_hat, z_hat)

        return z_hat  # (B, K, d)

    def forward(self, item_seq, item_seq_len):
        """Extract K interest vectors from the item sequence.

        Args:
            item_seq (torch.Tensor): Item sequence ids, shape (B, L).
            item_seq_len (torch.Tensor): Lengths of item sequences, shape (B,).

        Returns:
            torch.Tensor: K interest capsule vectors, shape (B, K, d).
        """
        item_seq_emb = self.item_embedding(item_seq)  # (B, L, d)
        mask = item_seq != 0  # (B, L), True for valid positions
        interests = self._dynamic_routing(item_seq_emb, mask)  # (B, K, d)
        return interests

    def _multi_interest_score(self, interests, item_emb):
        """Compute scores by taking max dot product across K interests.

        Args:
            interests (torch.Tensor): Interest vectors, shape (B, K, d).
            item_emb (torch.Tensor): Item embeddings, shape (B, d).

        Returns:
            torch.Tensor: Scores, shape (B,).
        """
        # (B, K, d) * (B, 1, d) -> (B, K, d) -> sum -> (B, K) -> max -> (B,)
        scores = (interests * item_emb.unsqueeze(1)).sum(-1).max(-1).values
        return scores

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        interests = self.forward(item_seq, item_seq_len)  # (B, K, d)
        pos_items = interaction[self.POS_ITEM_ID]

        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)  # (B, d)
            neg_items_emb = self.item_embedding(neg_items)  # (B, d)
            pos_score = self._multi_interest_score(interests, pos_items_emb)  # (B,)
            neg_score = self._multi_interest_score(interests, neg_items_emb)  # (B,)
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type == 'CE'
            test_item_emb = self.item_embedding.weight  # (n_items, d)
            # For each interest, compute scores with all items: (B, K, n_items)
            logits = torch.matmul(interests, test_item_emb.transpose(0, 1))
            # Max over K interests: (B, n_items)
            logits = logits.max(dim=1).values
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        interests = self.forward(item_seq, item_seq_len)  # (B, K, d)
        test_item_emb = self.item_embedding(test_item)  # (B, d)
        scores = self._multi_interest_score(interests, test_item_emb)  # (B,)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        interests = self.forward(item_seq, item_seq_len)  # (B, K, d)
        test_items_emb = self.item_embedding.weight  # (n_items, d)
        # Compute scores for each interest with all items: (B, K, n_items)
        scores = torch.matmul(interests, test_items_emb.transpose(0, 1))
        # Element-wise max across K interests: (B, n_items)
        scores = scores.max(dim=1).values
        return scores
