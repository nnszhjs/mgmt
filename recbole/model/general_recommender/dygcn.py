# -*- coding: utf-8 -*-
r"""
DyGCN
################################################

Reference:
    "Diversifying Recommendations on Digital Platforms: A Dynamic Graph Neural Network Approach"

    DyGCN is the backbone dynamic GNN of MECo-DGNN *without* the Matthew Effect
    control modules (L_reg and L_mat).  It models evolving user-item graphs via
    T cumulative time-step snapshots processed by a shared GNN, then aggregates
    the temporal sequence through an LSTM to produce final embeddings.

    Overall loss: L = L_BPR
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType


class DyGCN(GeneralRecommender):
    r"""DyGCN: Dynamic Graph Convolutional Network for Recommendation.

    Builds T cumulative graph snapshots from time-partitioned interactions,
    processes them through a shared GNN (LightGCN-style aggregation with MLP update),
    then feeds the temporal sequence through an LSTM to produce final embeddings.
    Trained with BPR loss only (no regularisation).
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(DyGCN, self).__init__(config, dataset)

        # ---------- load parameters ----------
        self.embedding_size = config["embedding_size"]
        self.n_layers       = config["n_layers"]
        self.n_time_steps   = config["n_time_steps"]
        self.lstm_hidden    = config["lstm_hidden"]
        self.mlp_hidden     = config["mlp_hidden"]
        self.use_compile    = config["use_compile"]
        self.use_tf32       = config["use_tf32"]
        d = self.embedding_size

        if self.use_tf32:
            torch.set_float32_matmul_precision("high")

        # ---------- build temporal graph snapshots ----------
        self.norm_adj_list = self._build_time_snapshots(dataset)
        self.norm_adj_list = [adj.to(self.device) for adj in self.norm_adj_list]

        # ---------- node embeddings ----------
        self.user_embedding = nn.Embedding(self.n_users, d)
        self.item_embedding = nn.Embedding(self.n_items, d)

        # ---------- GNN MLP update function f(z, h) -> h_next ----------
        self.gnn_mlp = nn.Sequential(
            nn.Linear(2 * d, self.mlp_hidden),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden, self.mlp_hidden),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden, d),
        )

        # ---------- Output projection o(): concat(h1..hL) -> d ----------
        self.output_proj = nn.Linear(self.n_layers * d, d)

        # ---------- LSTM: temporal aggregation ----------
        self.lstm = nn.LSTM(input_size=d, hidden_size=self.lstm_hidden, batch_first=False)
        self.lstm_proj = (
            nn.Linear(self.lstm_hidden, d) if self.lstm_hidden != d else nn.Identity()
        )

        # ---------- BPR loss ----------
        self.mf_loss = BPRLoss()

        # ---------- evaluation cache ----------
        self.restore_user_e = None
        self.restore_item_e = None

        # ---------- initialize parameters ----------
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

        # ---------- torch.compile ----------
        if self.use_compile and hasattr(torch, "compile"):
            self.gnn_mlp.forward = torch.compile(self.gnn_mlp.forward)

    # ------------------------------------------------------------------
    # Temporal graph construction
    # ------------------------------------------------------------------

    def _build_time_snapshots(self, dataset):
        """Partition interactions into T equal time windows and build
        cumulative normalized adjacency matrices for each snapshot.

        Returns:
            norm_adj_list : list[T] of sparse tensors (U+V, U+V)
        """
        uid_field  = dataset.uid_field
        iid_field  = dataset.iid_field
        time_field = dataset.time_field
        inter_feat = dataset.inter_feat

        users = np.array(inter_feat[uid_field])
        items = np.array(inter_feat[iid_field])
        T     = self.n_time_steps

        norm_adj_list = []

        if time_field and time_field in inter_feat:
            timestamps = np.array(inter_feat[time_field])
            t_min, t_max = timestamps.min(), timestamps.max()
            if t_max == t_min:
                boundaries = np.array([-np.inf] + [t_max] * T)
            else:
                boundaries    = np.linspace(t_min, t_max, T + 1)
                boundaries[0] = -np.inf
            for t in range(T):
                mask = timestamps <= boundaries[t + 1]
                u_t, v_t = users[mask], items[mask]
                norm_adj_list.append(self._build_norm_adj(u_t, v_t))
        else:
            adj = self._build_norm_adj(users, items)
            for _ in range(T):
                norm_adj_list.append(adj)

        return norm_adj_list

    def _build_norm_adj(self, users, items):
        """Symmetric-normalized bipartite adjacency: A_hat = D^{-1/2} A D^{-1/2}."""
        n   = self.n_users + self.n_items
        row = np.concatenate([users, items + self.n_users])
        col = np.concatenate([items + self.n_users, users])
        A   = sp.coo_matrix(
            (np.ones(len(row), dtype=np.float32), (row, col)), shape=(n, n)
        )
        deg_inv_sqrt = np.power(np.array(A.sum(axis=1)).flatten() + 1e-7, -0.5)
        D_inv_sqrt   = sp.diags(deg_inv_sqrt)
        L            = (D_inv_sqrt @ A @ D_inv_sqrt).tocoo()
        indices      = torch.LongTensor(np.stack([L.row, L.col]))
        values       = torch.FloatTensor(L.data)
        return torch.sparse_coo_tensor(indices, values, (n, n))

    # ------------------------------------------------------------------
    # GNN + LSTM forward pass
    # ------------------------------------------------------------------

    def _gnn_forward(self, norm_adj):
        """Single time-step GNN forward."""
        h = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        layer_outputs = []
        for _ in range(self.n_layers):
            z = torch.sparse.mm(norm_adj, h)
            h = self.gnn_mlp(torch.cat([z, h], dim=1))
            layer_outputs.append(h)
        return self.output_proj(torch.cat(layer_outputs, dim=1))

    def forward(self):
        """Full DyGCN forward pass.

        Returns:
            user_emb : (U, d)
            item_emb : (V, d)
        """
        self.lstm.flatten_parameters()

        time_step_embs = []
        for t in range(self.n_time_steps):
            z_t = self._gnn_forward(self.norm_adj_list[t])
            time_step_embs.append(z_t)

        seq = torch.stack(time_step_embs, dim=0)
        _, (h_n, _) = self.lstm(seq)
        d_star = self.lstm_proj(h_n.squeeze(0))

        return d_star[:self.n_users], d_star[self.n_users:]

    # ------------------------------------------------------------------
    # RecBole required methods
    # ------------------------------------------------------------------

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user     = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_emb, item_emb = self.forward()

        u_e   = user_emb[user]
        pos_e = item_emb[pos_item]
        neg_e = item_emb[neg_item]
        return self.mf_loss(
            (u_e * pos_e).sum(dim=-1),
            (u_e * neg_e).sum(dim=-1),
        )

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_emb, item_emb = self.forward()
        return (user_emb[user] * item_emb[item]).sum(dim=-1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        u_e = self.restore_user_e[user]
        return torch.matmul(u_e, self.restore_item_e.T).view(-1)
