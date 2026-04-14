# -*- coding: utf-8 -*-
r"""
MECo-DGNN
################################################

Reference:
    "Diversifying Recommendations on Digital Platforms: A Dynamic Graph Neural Network Approach"

    MECo-DGNN (Matthew Effect Control with Dynamic Graph Neural Networks) is a framework
    that models the dynamic evolution of the Matthew effect in user-item interactions and
    incorporates a control module to reduce exposure disparities across items.

    Key components:
    1. Dynamic GNN: Models evolving user-item graphs via T time-step snapshots + LSTM.
    2. Stage Degree Regularization (L_reg): Predicts item node degrees to encode graph topology.
    3. Matthew Effect Regularization (L_mat): Constrains item exposure to a target Gini coefficient.

    Overall loss: L = L_BPR + alpha * L_reg + beta * L_mat + reg_weight * L_emb
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import fsolve

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class MECoDGNN(GeneralRecommender):
    r"""MECo-DGNN: Matthew Effect Control with Dynamic Graph Neural Networks.

    Builds T graph snapshots (cumulative or incremental) from time-partitioned
    interactions, processes them through a shared GNN (LightGCN-style aggregation
    with MLP update), then feeds the temporal sequence through an LSTM to produce
    final embeddings. Two regularization modules penalize representations that fail
    to encode degree information (L_reg) or deviate from a predefined Gini-coefficient
    target (L_mat). An L2 embedding regularization (L_emb) prevents overfitting.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(MECoDGNN, self).__init__(config, dataset)

        # ---------- load parameters ----------
        self.embedding_size   = config["embedding_size"]        # d = 64
        self.n_layers         = config["n_layers"]              # L = 2
        self.n_time_steps     = config["n_time_steps"]          # T = 4
        self.lstm_hidden      = config["lstm_hidden"]           # hidden dim for RNN/LSTM
        self.temporal_agg     = (                               # temporal aggregation method
            config["temporal_agg"] if "temporal_agg" in config else "lstm"
        )  # Options: "lstm", "rnn", "mean", "attn"
        self.cumulative_graph = (                               # [Fix #7] read from config
            config["cumulative_graph"]
            if "cumulative_graph" in config
            else False
        )
        self.n_groups         = config["n_groups"]              # Q = 50
        self.alpha            = config["alpha"]                 # L_reg weight
        self.beta             = config["beta"]                  # L_mat weight
        self.gini_target      = config["gini_target"]           # target Gini coefficient
        self.mat_users        = (                               # sampled users for L_mat
            config["mat_users"] if "mat_users" in config else 1024
        )
        self.mat_items        = (                               # sampled items for L_mat
            config["mat_items"] if "mat_items" in config else 2048
        )
        self.mlp_hidden       = config["mlp_hidden"]            # MLP hidden dim = 128
        self.mat_freq         = (                               # [Fix #4] from config
            config["mat_freq"] if "mat_freq" in config else 1
        )
        self.reg_weight       = (                               # [Fix #3] L2 reg weight
            config["reg_weight"] if "reg_weight" in config else 1e-5
        )
        self.use_compile      = config["use_compile"]           # torch.compile
        self.use_tf32         = config["use_tf32"]              # TF32 matmul
        d = self.embedding_size

        # TF32 uses Tensor Core matmul with float32 range but reduced mantissa (10-bit vs 23-bit).
        # Safe for recommendation: BPR loss depends only on relative score ordering.
        # Ampere+ GPUs (A100/RTX30xx+) see 1.5-3x speedup on dense matmul; no-op on older GPUs.
        if self.use_tf32:
            torch.set_float32_matmul_precision("high")

        # ---------- build temporal graph snapshots ----------
        self.norm_adj_list, self.item_degrees = self._build_time_snapshots(dataset)
        # Move adjacency matrices to device once - avoids repeated CPU->GPU transfer per batch.
        self.norm_adj_list = [adj.to(self.device) for adj in self.norm_adj_list]

        # Pre-cache log-degree matrix as a buffer: shape (T, V).
        # Registered as a buffer so it moves with the model automatically.
        self.register_buffer(
            "log_deg_all",
            torch.stack([
                torch.log(torch.tensor(deg, dtype=torch.float32) + 1.0)
                for deg in self.item_degrees
            ], dim=0),   # (T, V)
        )

        # ---------- node embeddings ----------
        self.user_embedding = nn.Embedding(self.n_users, d)
        self.item_embedding = nn.Embedding(self.n_items, d)

        # ---------- GNN MLP update function f(z, h) -> h_next ----------
        # Paper footnote 7: 3-layer MLP, input=2d (concat z+h), output=d.
        # Shared across all GNN layers and time steps.
        self.gnn_mlp = nn.Sequential(
            nn.Linear(2 * d, self.mlp_hidden),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden, self.mlp_hidden),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden, d),
        )

        # ---------- Output projection o(): concat(h0..hL) -> d ----------
        # [Fix #6] Include initial embedding h0 in concatenation (L+1 components)
        self.output_proj = nn.Linear((self.n_layers + 1) * d, d)

        # ---------- Temporal aggregation: LSTM / RNN / Mean / Attention ----------
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
            self.temporal_module = None          # no learnable temporal module
        elif agg == "attn":
            # Lightweight self-attention over T time steps.
            # Learnable positional embedding for each snapshot.
            self.temporal_pos_emb = nn.Embedding(self.n_time_steps, d)
            n_heads = max(1, d // 16)            # e.g. d=64 -> 4 heads
            self.temporal_attn = nn.MultiheadAttention(
                embed_dim=d, num_heads=n_heads, batch_first=False, dropout=0.0,
            )
            self.temporal_norm = nn.LayerNorm(d)
        else:
            raise ValueError(
                f"Unknown temporal_agg='{self.temporal_agg}'. "
                f"Choose from: lstm, rnn, mean, attn."
            )

        # Projection from hidden_size back to d (only needed for lstm/rnn when dims differ)
        if agg in ("lstm", "rnn"):
            self.temporal_proj = (
                nn.Linear(self.lstm_hidden, d) if self.lstm_hidden != d else nn.Identity()
            )
        else:
            self.temporal_proj = nn.Identity()

        # ---------- Stage Degree Regularization FFNN (Eq. 8) ----------
        # Single linear layer: item embedding (d,) -> scalar log-degree prediction.
        self.degree_pred = nn.Linear(d, 1)

        # ---------- Matthew Effect FFNN (footnote 11) ----------
        # Two layers, input d (element-wise product u*v), output 2 (click/no-click).
        self.mat_ffnn = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 2),
        )

        # ---------- Lorenz-based item bins ----------
        self.kappa = self._solve_kappa(self.gini_target)
        # [Fix #7] Choose correct degree source based on graph mode
        if self.cumulative_graph:
            total_degrees = self.item_degrees[-1]          # last snapshot already cumulative
        else:
            total_degrees = np.sum(self.item_degrees, axis=0)  # sum incremental snapshots
        item_bins = self._compute_item_bins(total_degrees)
        self.register_buffer("item_bins", torch.LongTensor(item_bins))   # (V,)
        self.register_buffer(
            "target_dist", torch.full((self.n_groups,), 1.0 / self.n_groups)
        )

        # ---------- Loss functions ----------
        self.mf_loss = BPRLoss()
        self.emb_loss = EmbLoss()                          # [Fix #3] L2 embedding regularization

        # ---------- L_mat frequency cache ----------
        # _batch_count starts at -1 so that the first increment reaches 0,
        # and 0 % any mat_freq == 0 -> L_mat is always computed on batch 0.
        self._batch_count = -1
        # [Fix #5] Register as buffer so it follows .to(device) calls
        self.register_buffer("_l_mat_cache", torch.tensor(0.0))

        # ---------- evaluation cache ----------
        self.restore_user_e = None
        self.restore_item_e = None

        # ---------- initialize parameters ----------
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

        # ---------- torch.compile for compute-heavy sub-networks ----------
        # Compile the *forward method* rather than wrapping the module with torch.compile(module).
        # Wrapping with torch.compile(module) replaces the nn.Module with OptimizedModule,
        # whose __getattr__ delegates attribute lookup to _orig_mod. RecBole's get_flops
        # calls register_buffer("total_ops", ...) on every submodule; nn.Module.register_buffer
        # guards with `hasattr(self, name)` which hits the delegation and raises:
        #   KeyError: "attribute 'total_ops' already exists"
        # Compiling only the forward method leaves the module as a plain Sequential in the
        # tree, so register_buffer works correctly. The forward computation is still compiled.
        # LSTM is excluded because flatten_parameters() must remain accessible.
        # Requires PyTorch >= 2.0; silently skipped on older versions.
        if self.use_compile and hasattr(torch, "compile"):
            self.gnn_mlp.forward  = torch.compile(self.gnn_mlp.forward)
            self.mat_ffnn.forward = torch.compile(self.mat_ffnn.forward)

    # ------------------------------------------------------------------
    # Temporal graph construction
    # ------------------------------------------------------------------

    def _build_time_snapshots(self, dataset):
        """Partition interactions into T equal time windows and build
        normalized adjacency matrices for each snapshot.

        When cumulative_graph=True, each snapshot includes all interactions up to
        its boundary. When False (default), each snapshot only includes interactions
        within its own time window (incremental / independent snapshots).

        Returns:
            norm_adj_list : list[T] of sparse tensors (U+V, U+V)
            item_degrees  : list[T] of np.ndarray (V,) with per-snapshot degrees
        """
        uid_field  = dataset.uid_field
        iid_field  = dataset.iid_field
        time_field = dataset.time_field
        inter_feat = dataset.inter_feat

        users = np.array(inter_feat[uid_field])
        items = np.array(inter_feat[iid_field])
        T     = self.n_time_steps

        norm_adj_list = []
        item_degrees  = []

        if time_field and time_field in inter_feat:
            timestamps = np.array(inter_feat[time_field])
            t_min, t_max = timestamps.min(), timestamps.max()
            if t_max == t_min:
                boundaries = np.array([-np.inf] + [t_max] * T)
            else:
                boundaries    = np.linspace(t_min, t_max, T + 1)
                boundaries[0] = -np.inf   # first bin captures all up to t_max/T
            for t in range(T):
                if self.cumulative_graph:
                    mask = timestamps <= boundaries[t + 1]
                else:
                    # Incremental: only interactions within this time window
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
            # No timestamp: replicate the single full-graph snapshot T times.
            adj = self._build_norm_adj(users, items)
            deg = np.bincount(items, minlength=self.n_items).astype(np.float32)
            for _ in range(T):
                norm_adj_list.append(adj)
                item_degrees.append(deg)

        return norm_adj_list, item_degrees

    def _build_norm_adj(self, users, items):
        """Symmetric-normalized bipartite adjacency: A_hat = D^{-1/2} A D^{-1/2}.

        A = [[0, R], [R^T, 0]]  shape (U+V, U+V)

        Returns:
            torch.sparse_coo_tensor of shape (n_users+n_items, n_users+n_items)
        """
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
    # Lorenz / Gini utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _lorenz(p, kappa):
        """L(p; kappa) = (exp(kappa*p) - 1) / (exp(kappa) - 1)"""
        return (np.exp(kappa * p) - 1.0) / (np.exp(kappa) - 1.0)

    @staticmethod
    def _gini_from_kappa(kappa):
        """Gini = ((kappa-2)*e^kappa + (kappa+2)) / (kappa*(e^kappa-1))"""
        ek = np.exp(kappa)
        return ((kappa - 2) * ek + (kappa + 2)) / (kappa * (ek - 1))

    def _solve_kappa(self, gini_target):
        """Solve for kappa given a target Gini using scipy nonlinear solver."""
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
        """Assign each item to one of Q bins using vectorised NumPy ops.

        Items are ranked by historical popularity. The inverse Lorenz curve
        under kappa defines bin boundaries such that each bin covers an equal
        share of total clicks under the target Gini distribution.

        Returns:
            np.ndarray of shape (n_items,), values in [0, Q-1]
        """
        Q = self.n_groups
        n = self.n_items

        # rank_order[k] = item index at popularity rank k  (0 = least popular)
        rank_order = np.argsort(item_degrees)

        # Lorenz-curve bin boundaries (proportion of items at each boundary)
        if self.kappa < 1e-5:
            p_boundaries = [q / Q for q in range(1, Q)]
        else:
            p_values      = np.linspace(0, 1, 10001)
            lorenz_values = self._lorenz(p_values, self.kappa)
            targets       = np.arange(1, Q) / Q                    # (Q-1,)
            idxs          = np.searchsorted(lorenz_values, targets) # vectorised
            idxs          = np.clip(idxs, 0, len(p_values) - 1)
            p_boundaries  = p_values[idxs].tolist()

        # Convert proportions -> integer item counts -> bin assignment
        bin_boundaries = np.array(
            [0] + [int(round(p * n)) for p in p_boundaries] + [n]
        )
        # rank_pos[item] = rank of that item (inverse permutation of rank_order)
        rank_pos  = np.argsort(rank_order)
        # np.searchsorted(bin_boundaries[1:-1], rank_pos, side='right') maps each rank
        # to the index of the first boundary it meets or exceeds -> bin index.
        item_bins = np.searchsorted(
            bin_boundaries[1:-1], rank_pos, side="right"
        ).astype(np.int64)
        return item_bins

    # ------------------------------------------------------------------
    # GNN + LSTM forward pass
    # ------------------------------------------------------------------

    def _gnn_forward(self, norm_adj):
        """Single time-step GNN forward.

        L layers of:
          z^l  = A_hat @ h^l           (symmetric-normalised aggregation, Eq. 2-3)
          h^{l+1} = MLP(concat[z^l, h^l])   (MLP update, Eq. 4)

        Output: o(h^0, h^1, ..., h^L) = Linear(concat[h^0..h^L])   (Eq. 5)
        [Fix #6] h^0 (initial embedding) is included in the concatenation.

        Returns: (U+V, d)
        """
        h = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        layer_outputs = [h]                      # [Fix #6] include h^0
        for _ in range(self.n_layers):
            z = torch.sparse.mm(norm_adj, h)
            h = self.gnn_mlp(torch.cat([z, h], dim=1))
            layer_outputs.append(h)
        return self.output_proj(torch.cat(layer_outputs, dim=1))   # (U+V, d)

    def _temporal_aggregate(self, seq):
        """Aggregate T time-step embeddings into a single representation.

        Args:
            seq: (T, N, d) tensor of per-snapshot node embeddings.

        Returns:
            d_star: (N, d) final node embeddings.
        """
        agg = self.temporal_agg.lower()

        if agg in ("lstm", "rnn"):
            # flatten_parameters() makes weights contiguous for cuDNN.
            self.temporal_module.flatten_parameters()
            if agg == "lstm":
                _, (h_n, _) = self.temporal_module(seq)    # h_n: (1, N, hidden)
            else:
                _, h_n = self.temporal_module(seq)          # h_n: (1, N, hidden)
            d_star = self.temporal_proj(h_n.squeeze(0))    # (N, d)

        elif agg == "mean":
            d_star = seq.mean(dim=0)                       # (N, d)

        elif agg == "attn":
            # Add learnable positional embeddings: (T, 1, d) broadcast to (T, N, d)
            pos = self.temporal_pos_emb.weight.unsqueeze(1)  # (T, 1, d)
            seq_pos = seq + pos                              # (T, N, d)
            # Self-attention: query = key = value = seq_pos
            attn_out, _ = self.temporal_attn(seq_pos, seq_pos, seq_pos)  # (T, N, d)
            # Residual + LayerNorm, then take the LAST time step as final representation
            attn_out = self.temporal_norm(attn_out + seq_pos)
            d_star = attn_out[-1]                            # (N, d)

        return d_star

    def forward(self):
        """Full MECo-DGNN forward pass.

        Returns:
            user_emb           : (U, d)
            item_emb           : (V, d)
            time_step_item_embs: list[T] of (V, d)  -- for L_reg
        """
        time_step_embs      = []
        time_step_item_embs = []
        for t in range(self.n_time_steps):
            z_t = self._gnn_forward(self.norm_adj_list[t])   # (U+V, d)
            time_step_embs.append(z_t)
            time_step_item_embs.append(z_t[self.n_users:])   # (V, d)

        # Temporal aggregation: (T, U+V, d) -> (U+V, d)
        seq = torch.stack(time_step_embs, dim=0)             # (T, U+V, d)
        d_star = self._temporal_aggregate(seq)                # (U+V, d)

        return d_star[:self.n_users], d_star[self.n_users:], time_step_item_embs

    # ------------------------------------------------------------------
    # Loss components
    # ------------------------------------------------------------------

    def _compute_l_reg(self, time_step_item_embs):
        """Stage Degree Regularization (Eq. 8-10) -- fully vectorised over T.

        Stacks item embeddings from all T time steps into a (T, V, d) tensor,
        applies the degree-prediction head in one pass, and computes MSE against
        the pre-cached log-degree matrix (T, V).

        L_reg = mean over (t, v) of (log(d_v^t + 1) - k_hat_v^t)^2
        """
        # (T, V, d)  ->  degree_pred: Linear(d,1) broadcasts over leading dims
        all_item_embs = torch.stack(time_step_item_embs, dim=0)  # (T, V, d)
        k_hat = self.degree_pred(all_item_embs).squeeze(-1)       # (T, V)
        return F.mse_loss(k_hat, self.log_deg_all, reduction="mean")

    def _compute_l_mat(self, user_emb, item_emb):
        """Matthew Effect Regularization (Eq. 11-15) -- binning fully vectorised.

        1. Sample mat_users users and mat_items items for memory efficiency.
        2. Compute element-wise product u*v in chunks (chunk_size users at a time).
        3. FFNN + Gumbel-Softmax -> click probability; sum over sampled users -> d_v*.
        4. scatter_add_ to accumulate d_v* into Q bins.
        5. KL divergence from S to uniform S_0.
        [Fix #2] Uses .mean() instead of .sum() for numerical stability.
        """
        n_users    = user_emb.shape[0]
        n_items    = item_emb.shape[0]
        n_sample_u = min(self.mat_users, n_users)
        n_sample_v = min(self.mat_items, n_items)

        # Random permutations for sampling
        u_idx = torch.randperm(n_users, device=self.device)[:n_sample_u]
        v_idx = torch.randperm(n_items, device=self.device)[:n_sample_v]

        u_emb = user_emb[u_idx]    # (n_sample_u, d)
        v_emb = item_emb[v_idx]    # (n_sample_v, d)

        # --- Chunked FFNN to avoid OOM ---
        chunk_size = 128
        d_star = torch.zeros(n_sample_v, device=self.device)
        for start in range(0, n_sample_u, chunk_size):
            u_chunk  = u_emb[start: start + chunk_size]                    # (C, d)
            pw       = u_chunk.unsqueeze(1) * v_emb.unsqueeze(0)           # (C, V', d)
            logits   = self.mat_ffnn(pw.reshape(-1, pw.shape[-1]))         # (C*V', 2)
            probs    = F.gumbel_softmax(logits, tau=2.0, hard=False)       # (C*V', 2)
            click    = probs[:, 1].reshape(u_chunk.shape[0], n_sample_v)   # (C, V')
            d_star  += click.sum(dim=0)                                    # accumulate

        # --- Vectorised binning with scatter_add ---
        sampled_bins = self.item_bins[v_idx]                    # (V',)
        S = torch.zeros(self.n_groups, device=self.device)
        S.scatter_add_(0, sampled_bins, d_star)

        # Normalize to form probability distribution
        S = S / (S.sum() + 1e-8)

        # KL(S || uniform S_0) = sum_q S_q * log(S_q / S_{0,q})
        # Since S_0 is uniform (1/Q), it becomes sum_q S_q * log(S_q * Q)
        S_safe = S.clamp(min=1e-8)
        # [Fix #2] Use .mean() for numerical stability, consistent with L_bpr and L_reg
        return (S_safe * torch.log(S_safe * self.n_groups)).mean()

    # ------------------------------------------------------------------
    # RecBole required methods
    # ------------------------------------------------------------------

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user     = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_emb, item_emb, time_step_item_embs = self.forward()

        # BPR loss (Eq. 16)
        u_e       = user_emb[user]
        pos_e     = item_emb[pos_item]
        neg_e     = item_emb[neg_item]
        l_bpr     = self.mf_loss(
            (u_e * pos_e).sum(dim=-1),
            (u_e * neg_e).sum(dim=-1),
        )

        # [Fix #3] L2 embedding regularization on raw (pre-GNN) embeddings
        u_ego   = self.user_embedding(user)
        pos_ego = self.item_embedding(pos_item)
        neg_ego = self.item_embedding(neg_item)
        l_emb   = self.emb_loss(u_ego, pos_ego, neg_ego)

        # Stage Degree Regularization (Eq. 8-10) -- every batch
        l_reg = self._compute_l_reg(time_step_item_embs)

        # Matthew Effect Regularization (Eq. 11-15) -- every mat_freq batches
        self._batch_count += 1
        if self._batch_count % self.mat_freq == 0:
            l_mat = self._compute_l_mat(user_emb, item_emb)
            self._l_mat_cache = l_mat.detach()
        else:
            l_mat = self._l_mat_cache

        # [Fix #1] Return dict for structured logging.
        # Trainer uses losses["loss"] for backward; other keys are logged as components.
        loss = l_bpr + self.alpha * l_reg + self.beta * l_mat + self.reg_weight * l_emb
        return {
            "loss": loss,
            "l_bpr": l_bpr.detach(),
            "l_reg": l_reg.detach(),
            "l_mat": l_mat.detach() if torch.is_tensor(l_mat) else l_mat,
            "l_emb": l_emb.detach(),
        }

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_emb, item_emb, _ = self.forward()
        return (user_emb[user] * item_emb[item]).sum(dim=-1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e, _ = self.forward()
        u_e = self.restore_user_e[user]
        return torch.matmul(u_e, self.restore_item_e.T).view(-1)
