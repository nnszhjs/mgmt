# -*- coding: utf-8 -*-
r"""
MECoDGNNLEG (Legacy)
################################################

Faithful reproduction of the original MgmtSci/code model logic within RecBole.

Key differences from MECoDGNN (new implementation):
1. GNN: Pure LightGCN propagation (no MLP update), layer-wise mean pooling.
2. Per-stage GNN depth: each time-step can have a different number of layers.
3. Residual connection: final embeddings = GNN+temporal output + raw embeddings.
4. L_mat: full-user Gumbel-Softmax edge prediction (tau=1, hard=True), single Linear(d,2).
5. L_reg: batch-only stage degree loss with sum reduction.
6. L2 reg: manual (1/2)*norm^2/batch_size (not RecBole's EmbLoss).

Temporal aggregation reuses the new code's configurable module (LSTM/RNN/Mean/Attn).
Graph construction reuses the new code's _build_time_snapshots (time-based partitioning).
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import fsolve

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType


class MECoDGNNLEG(GeneralRecommender):
    r"""MECoDGNNLEG: Legacy faithful reproduction of the original DGNN model.

    Pure LightGCN propagation per time-step with per-stage layer counts,
    RNN/LSTM temporal aggregation, residual embedding connection, and
    original-style L_mat / L_reg losses.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(MECoDGNNLEG, self).__init__(config, dataset)

        # ---------- load parameters ----------
        self.embedding_size = config["embedding_size"]
        self.n_layers_per_stage = config["n_layers_per_stage"]
        self.n_time_steps = config["n_time_steps"]
        self.lstm_hidden = config["lstm_hidden"]
        self.temporal_agg = (
            config["temporal_agg"] if "temporal_agg" in config else "rnn"
        )
        self.cumulative_graph = (
            config["cumulative_graph"] if "cumulative_graph" in config else False
        )
        self.n_groups = config["n_groups"]
        self.alpha = config["alpha"]          # L_reg weight (old stage_dl)
        self.beta = config["beta"]            # L_mat weight (old dl)
        self.gini_target = config["gini_target"]
        self.reg_weight = (
            config["reg_weight"] if "reg_weight" in config else 1e-3
        )
        self.mat_users = (
            config["mat_users"] if "mat_users" in config else 1024
        )
        self.mat_items = (
            config["mat_items"] if "mat_items" in config else 2048
        )
        self.use_compile = config["use_compile"]
        self.use_tf32 = config["use_tf32"]
        d = self.embedding_size

        # Pad n_layers_per_stage if shorter than n_time_steps
        if len(self.n_layers_per_stage) < self.n_time_steps:
            last = self.n_layers_per_stage[-1]
            self.n_layers_per_stage = list(self.n_layers_per_stage) + [last] * (
                self.n_time_steps - len(self.n_layers_per_stage)
            )

        if self.use_tf32:
            torch.set_float32_matmul_precision("high")

        # ---------- build temporal graph snapshots ----------
        self.norm_adj_list, self.item_degrees = self._build_time_snapshots(dataset)
        self.norm_adj_list = [adj.to(self.device) for adj in self.norm_adj_list]

        # Pre-cache log-degree: shape (T, V)
        self.register_buffer(
            "log_deg_all",
            torch.stack([
                torch.log(torch.tensor(deg, dtype=torch.float32) + 1.0)
                for deg in self.item_degrees
            ], dim=0),
        )

        # ---------- node embeddings (normal init, std=0.1, matching old code) ----------
        self.user_embedding = nn.Embedding(self.n_users, d)
        self.item_embedding = nn.Embedding(self.n_items, d)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

        # ---------- Temporal aggregation (reuse new code's design) ----------
        agg = self.temporal_agg.lower()
        if agg == "lstm":
            self.temporal_module = nn.LSTM(
                input_size=d, hidden_size=self.lstm_hidden, batch_first=False
            )
        elif agg == "rnn":
            self.temporal_module = nn.RNN(
                input_size=d, hidden_size=self.lstm_hidden,
                batch_first=False, nonlinearity="tanh",
            )
        elif agg == "mean":
            self.temporal_module = None
        elif agg == "attn":
            self.temporal_pos_emb = nn.Embedding(self.n_time_steps, d)
            n_heads = max(1, d // 16)
            self.temporal_attn = nn.MultiheadAttention(
                embed_dim=d, num_heads=n_heads, batch_first=False, dropout=0.0,
            )
            self.temporal_norm = nn.LayerNorm(d)
        else:
            raise ValueError(
                f"Unknown temporal_agg='{self.temporal_agg}'. "
                f"Choose from: lstm, rnn, mean, attn."
            )

        if agg in ("lstm", "rnn"):
            self.temporal_proj = (
                nn.Linear(self.lstm_hidden, d) if self.lstm_hidden != d else nn.Identity()
            )
        else:
            self.temporal_proj = nn.Identity()

        # ---------- L_mat: single Linear(d, 2) — old code's self.fl ----------
        self.fl = nn.Linear(d, 2)

        # ---------- L_reg: Linear(d, 1) — old code's self.fl2 ----------
        self.fl2 = nn.Linear(d, 1)

        # ---------- Lorenz-based item bins ----------
        self.kappa = self._solve_kappa(self.gini_target)
        if self.cumulative_graph:
            total_degrees = self.item_degrees[-1]
        else:
            total_degrees = np.sum(self.item_degrees, axis=0)
        item_bins = self._compute_item_bins(total_degrees)
        self.register_buffer("item_bins", torch.LongTensor(item_bins))
        self.register_buffer(
            "target_dist", torch.full((self.n_groups,), 1.0 / self.n_groups)
        )
        # Pre-build one-hot group indicator for matrix-multiply grouping
        # Shape: (n_items, n_groups) — used in _compute_l_mat
        one_hot = torch.zeros(self.n_items, self.n_groups)
        one_hot[torch.arange(self.n_items), item_bins] = 1.0
        self.register_buffer("group_indicator", one_hot)

        # ---------- evaluation cache ----------
        self.restore_user_e = None
        self.restore_item_e = None
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

        # ---------- torch.compile ----------
        if self.use_compile and hasattr(torch, "compile"):
            self.fl.forward = torch.compile(self.fl.forward)

    # ------------------------------------------------------------------
    # Temporal graph construction (reused from MECoDGNN)
    # ------------------------------------------------------------------

    def _build_time_snapshots(self, dataset):
        uid_field = dataset.uid_field
        iid_field = dataset.iid_field
        time_field = dataset.time_field
        inter_feat = dataset.inter_feat

        users = np.array(inter_feat[uid_field])
        items = np.array(inter_feat[iid_field])
        T = self.n_time_steps

        norm_adj_list = []
        item_degrees = []

        if time_field and time_field in inter_feat:
            timestamps = np.array(inter_feat[time_field])
            t_min, t_max = timestamps.min(), timestamps.max()
            if t_max == t_min:
                boundaries = np.array([-np.inf] + [t_max] * T)
            else:
                boundaries = np.linspace(t_min, t_max, T + 1)
                boundaries[0] = -np.inf
            for t in range(T):
                if self.cumulative_graph:
                    mask = timestamps <= boundaries[t + 1]
                else:
                    if t == 0:
                        mask = timestamps <= boundaries[1]
                    else:
                        mask = (timestamps > boundaries[t]) & (timestamps <= boundaries[t + 1])
                u_t, v_t = users[mask], items[mask]
                norm_adj_list.append(self._build_norm_adj(u_t, v_t))
                item_degrees.append(
                    np.bincount(v_t, minlength=self.n_items).astype(np.float32)
                )
        else:
            adj = self._build_norm_adj(users, items)
            deg = np.bincount(items, minlength=self.n_items).astype(np.float32)
            for _ in range(T):
                norm_adj_list.append(adj)
                item_degrees.append(deg)

        return norm_adj_list, item_degrees

    def _build_norm_adj(self, users, items):
        n = self.n_users + self.n_items
        row = np.concatenate([users, items + self.n_users])
        col = np.concatenate([items + self.n_users, users])
        A = sp.coo_matrix(
            (np.ones(len(row), dtype=np.float32), (row, col)), shape=(n, n)
        )
        deg_inv_sqrt = np.power(np.array(A.sum(axis=1)).flatten() + 1e-7, -0.5)
        D_inv_sqrt = sp.diags(deg_inv_sqrt)
        L = (D_inv_sqrt @ A @ D_inv_sqrt).tocoo()
        indices = torch.LongTensor(np.stack([L.row, L.col]))
        values = torch.FloatTensor(L.data)
        return torch.sparse_coo_tensor(indices, values, (n, n))

    # ------------------------------------------------------------------
    # Lorenz / Gini utilities (reused from MECoDGNN)
    # ------------------------------------------------------------------

    @staticmethod
    def _lorenz(p, kappa):
        return (np.exp(kappa * p) - 1.0) / (np.exp(kappa) - 1.0)

    @staticmethod
    def _gini_from_kappa(kappa):
        ek = np.exp(kappa)
        return ((kappa - 2) * ek + (kappa + 2)) / (kappa * (ek - 1))

    def _solve_kappa(self, gini_target):
        if gini_target <= 0.0:
            return 1e-6
        if gini_target >= 1.0:
            return 50.0
        def objective(k):
            return self._gini_from_kappa(k[0]) - gini_target
        try:
            k_est = float(fsolve(objective, x0=[2.0])[0])
            return max(k_est, 1e-6)
        except Exception:
            return 2.0

    def _compute_item_bins(self, item_degrees):
        Q = self.n_groups
        n = self.n_items
        rank_order = np.argsort(item_degrees)
        if self.kappa < 1e-5:
            p_boundaries = [q / Q for q in range(1, Q)]
        else:
            p_values = np.linspace(0, 1, 10001)
            lorenz_values = self._lorenz(p_values, self.kappa)
            targets = np.arange(1, Q) / Q
            idxs = np.searchsorted(lorenz_values, targets)
            idxs = np.clip(idxs, 0, len(p_values) - 1)
            p_boundaries = p_values[idxs].tolist()
        bin_boundaries = np.array(
            [0] + [int(round(p * n)) for p in p_boundaries] + [n]
        )
        rank_pos = np.argsort(rank_order)
        item_bins = np.searchsorted(
            bin_boundaries[1:-1], rank_pos, side="right"
        ).astype(np.int64)
        return item_bins

    # ------------------------------------------------------------------
    # GNN forward — pure LightGCN (no MLP), layer-wise mean pooling
    # ------------------------------------------------------------------

    def _gnn_forward(self, norm_adj, n_layers):
        """Single time-step GNN: pure LightGCN propagation.

        For each layer: all_emb = A_hat @ all_emb
        Output: mean of [h^0, h^1, ..., h^L]

        This matches old code model.py:computer(i).
        """
        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight
        all_emb = torch.cat([users_emb, items_emb], dim=0)
        embs = [all_emb]
        for _ in range(n_layers):
            all_emb = torch.sparse.mm(norm_adj, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        return light_out

    # ------------------------------------------------------------------
    # Temporal aggregation (reused from MECoDGNN)
    # ------------------------------------------------------------------

    def _temporal_aggregate(self, seq):
        agg = self.temporal_agg.lower()
        if agg in ("lstm", "rnn"):
            self.temporal_module.flatten_parameters()
            if agg == "lstm":
                _, (h_n, _) = self.temporal_module(seq)
            else:
                _, h_n = self.temporal_module(seq)
            d_star = self.temporal_proj(h_n.squeeze(0))
        elif agg == "mean":
            d_star = seq.mean(dim=0)
        elif agg == "attn":
            pos = self.temporal_pos_emb.weight.unsqueeze(1)
            seq_pos = seq + pos
            attn_out, _ = self.temporal_attn(seq_pos, seq_pos, seq_pos)
            attn_out = self.temporal_norm(attn_out + seq_pos)
            d_star = attn_out[-1]
        return d_star

    # ------------------------------------------------------------------
    # Full forward pass
    # ------------------------------------------------------------------

    def forward(self):
        """Full forward: GNN per time-step -> temporal aggregation.

        Returns:
            user_emb: (U, d) — GNN+temporal output (before residual)
            item_emb: (V, d) — GNN+temporal output (before residual)
            time_step_item_embs: list[T] of (V, d) — for L_reg
        """
        time_step_embs = []
        time_step_item_embs = []
        for t in range(self.n_time_steps):
            z_t = self._gnn_forward(
                self.norm_adj_list[t], self.n_layers_per_stage[t]
            )
            time_step_embs.append(z_t)
            time_step_item_embs.append(z_t[self.n_users:])

        seq = torch.stack(time_step_embs, dim=0)
        d_star = self._temporal_aggregate(seq)

        user_emb = d_star[:self.n_users]
        item_emb = d_star[self.n_users:]
        return user_emb, item_emb, time_step_item_embs

    # ------------------------------------------------------------------
    # Loss: L_reg — stage degree loss, mean reduction
    # ------------------------------------------------------------------

    def _compute_l_reg(self, time_step_item_embs, batch_items):
        """Stage Degree Regularization on batch items only.

        For each time-step t, predict degree of batch items via fl2,
        compare against log(degree+1) using MSE (mean reduction).
        """
        all_item_embs = torch.stack(
            [embs[batch_items] for embs in time_step_item_embs], dim=0
        )  # (T, B, d)
        k_hat = self.fl2(all_item_embs).squeeze(-1)           # (T, B)
        labels = self.log_deg_all[:, batch_items]              # (T, B)
        return F.mse_loss(k_hat, labels, reduction="mean")

    # ------------------------------------------------------------------
    # Loss: L_mat — full-user Gumbel-Softmax edge prediction
    # Matches old code model.py:item_degree_loss
    # ------------------------------------------------------------------

    def _compute_l_mat(self, user_emb, item_emb):
        """Matthew Effect Regularization with sampled users/items.

        Same logic as old code (Linear(d,2) + Gumbel-Softmax tau=1 hard=True +
        matrix-multiply grouping + KL), but samples mat_users users and
        mat_items items instead of using all users.

        Processes in chunks over sampled users to control memory.
        """
        n_users = user_emb.shape[0]
        n_items = item_emb.shape[0]
        n_sample_u = min(self.mat_users, n_users)
        n_sample_v = min(self.mat_items, n_items)

        u_idx = torch.randperm(n_users, device=self.device)[:n_sample_u]
        v_idx = torch.randperm(n_items, device=self.device)[:n_sample_v]

        u_emb = user_emb[u_idx]    # (n_sample_u, d)
        v_emb = item_emb[v_idx]    # (n_sample_v, d)

        chunk_size = 128
        item_d = torch.zeros(n_sample_v, device=self.device)

        for start in range(0, n_sample_u, chunk_size):
            u_chunk = u_emb[start: start + chunk_size]          # (C, d)
            # edge_emb: (C, V', d)
            edge_emb = u_chunk.unsqueeze(1) * v_emb.unsqueeze(0)
            edge_logits = self.fl(edge_emb)                     # (C, V', 2)
            edge = F.gumbel_softmax(edge_logits, tau=1, hard=True)
            item_d += torch.sum(edge[:, :, 0], dim=0)           # accumulate

        # Group aggregation via matrix multiply
        sampled_group_ind = self.group_indicator[v_idx]          # (V', Q)
        group_degree = torch.matmul(
            item_d.unsqueeze(0).float(), sampled_group_ind
        )  # (1, Q)

        group_degree = group_degree / (group_degree.sum() + 1e-8)

        # KL divergence with mean reduction for numerical stability
        S_safe = group_degree.clamp(min=1e-8)
        dl_loss = (S_safe * torch.log(S_safe * self.n_groups)).mean()
        return dl_loss

    # ------------------------------------------------------------------
    # RecBole required: calculate_loss
    # ------------------------------------------------------------------

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_emb, item_emb, time_step_item_embs = self.forward()

        # Residual connection: GNN+temporal output + raw embedding
        u_e = user_emb[user] + self.user_embedding(user)
        pos_e = item_emb[pos_item] + self.item_embedding(pos_item)
        neg_e = item_emb[neg_item] + self.item_embedding(neg_item)

        # BPR loss (softplus, matching old code)
        pos_scores = torch.sum(torch.mul(u_e, pos_e), dim=1)
        neg_scores = torch.sum(torch.mul(u_e, neg_e), dim=1)
        l_bpr = torch.mean(F.softplus(neg_scores - pos_scores))

        # L2 reg: manual (1/2)*norm^2/batch_size (matching old code)
        u_ego = self.user_embedding(user)
        pos_ego = self.item_embedding(pos_item)
        neg_ego = self.item_embedding(neg_item)
        l_emb = (1 / 2) * (
            u_ego.norm(2).pow(2)
            + pos_ego.norm(2).pow(2)
            + neg_ego.norm(2).pow(2)
        ) / float(len(user))

        # Unique items in this batch for L_reg and L_mat
        batch_items = torch.unique(torch.cat([pos_item, neg_item]))

        # L_reg: stage degree loss (batch-only, sum reduction)
        l_reg = self._compute_l_reg(time_step_item_embs, batch_items)

        # L_mat: matthew effect loss (sampled users/items, Gumbel-Softmax)
        all_user_emb = user_emb + self.user_embedding.weight
        all_item_emb = item_emb + self.item_embedding.weight
        l_mat = self._compute_l_mat(all_user_emb, all_item_emb)

        # Total loss: bpr + reg_weight*l2 + alpha*l_reg + beta*l_mat
        loss = l_bpr + self.reg_weight * l_emb + self.alpha * l_reg + self.beta * l_mat

        return {
            "loss": loss,
            "l_bpr": l_bpr.detach(),
            "l_reg": l_reg.detach(),
            "l_mat": l_mat.detach(),
            "l_emb": l_emb.detach(),
        }

    # ------------------------------------------------------------------
    # RecBole required: predict / full_sort_predict
    # ------------------------------------------------------------------

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_emb, item_emb, _ = self.forward()
        # Residual connection
        u_e = user_emb[user] + self.user_embedding(user)
        i_e = item_emb[item] + self.item_embedding(item)
        return torch.mul(u_e, i_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            user_emb, item_emb, _ = self.forward()
            # Residual for all users/items
            self.restore_user_e = user_emb + self.user_embedding.weight
            self.restore_item_e = item_emb + self.item_embedding.weight
        u_e = self.restore_user_e[user]
        return torch.matmul(u_e, self.restore_item_e.T).view(-1)
