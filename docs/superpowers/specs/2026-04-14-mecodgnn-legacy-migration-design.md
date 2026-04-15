# MECoDGNNLEG: Legacy Model Migration Design

## Goal

Migrate the original MgmtSci/code model logic into RecBole as `MECoDGNNLEG`, preserving the old computation graph exactly so benchmark results can be reproduced and compared against the new `MECoDGNN`.

## Scope

- **Preserve from old code**: GNN propagation, per-stage layer counts, residual connection, L_mat (Gumbel-Softmax edge prediction), L_reg (stage degree loss), loss combination.
- **Adopt from new code**: RecBole interface (`GeneralRecommender`), temporal graph construction (`_build_time_snapshots`), temporal aggregation module (LSTM/RNN/Mean/Attn), parameter registration style (yaml config).
- **New file**: `recbole/model/general_recommender/mecodgnn_legacy.py`
- **New config**: `recbole/properties/model/MECoDGNNLEG.yaml`
- **Registration**: add import to `general_recommender/__init__.py`

## Detailed Differences (Old vs New, what Legacy preserves)

### 1. GNN Propagation — Pure LightGCN (no MLP)

Old code `model.py:computer(i)`:
```python
for layer in range(self.n_layers_se[i]):
    all_emb = torch.sparse.mm(g_droped, all_emb)
    embs.append(all_emb)
embs = torch.stack(embs, dim=1)
light_out = torch.mean(embs, dim=1)
```

New code `mecodgnn.py:_gnn_forward`: each layer does `z = A_hat @ h` then `h = MLP(concat[z, h])`, and final output is `Linear(concat[h0..hL])`.

**Legacy keeps**: pure sparse matmul per layer, layer-wise mean pooling, no MLP, no output projection.

### 2. Per-Stage GNN Layer Counts

Old code: `self.n_layers_se = [3,3,3,3]` — each time step can have a different GNN depth.

New code: single `self.n_layers` shared across all time steps.

**Legacy keeps**: a list `n_layers_per_stage` read from config (default `[3,3,3,3]`). If the list length < T, the last value is repeated.

### 3. Temporal Aggregation — Reuse New Code

Old code: fixed `nn.RNN(d, d, 1)`.

New code: configurable LSTM/RNN/Mean/Attn via `_temporal_aggregate`.

**Legacy adopts new code's temporal aggregation module** (user confirmed). Default config will set `temporal_agg: rnn` and `lstm_hidden` equal to `embedding_size` to match old behavior when desired.

### 4. Residual Connection on Final Embeddings

Old code `model.py:getUsersRating` and `getEmbedding`:
```python
users_emb = all_users[users] + self.embedding_user(users)
pos_emb = all_items[pos_items] + self.embedding_item(pos_items)
```

New code: no residual — uses GNN+temporal output directly.

**Legacy keeps**: explicit addition of raw embedding to GNN+temporal output for scoring and loss computation.

### 5. L_mat — Full-User Gumbel-Softmax Edge Prediction

Old code `model.py:item_degree_loss`:
- Computes `edge_emb = all_user * item_emb.unsqueeze(1)` for ALL users × batch items
- Single `Linear(d, 2)` layer (`self.fl`)
- `F.gumbel_softmax(tau=1, hard=True)`
- Groups via matrix multiply with one-hot group indicator
- KL divergence against uniform distribution

New code `mecodgnn.py:_compute_l_mat`:
- Samples `mat_users` users and `mat_items` items
- Two-layer MLP `Linear(d,d)->ReLU->Linear(d,2)`
- `F.gumbel_softmax(tau=2.0, hard=False)`
- Groups via `scatter_add_`

**Legacy keeps**: full-user computation, single Linear(d,2), tau=1 hard=True, matrix-multiply grouping, KL loss. The group indicator matrix is built from `item_bins` (Lorenz-curve binning reused from new code's `_compute_item_bins`).

### 6. L_reg — Batch-Only Stage Degree Loss with Sum Reduction

Old code `model.py:item_stage_degree_loss`:
- Only computes for batch items (not all items)
- Loops over T stages, applies `Linear(d, 1)` per stage
- Label: `log(degree + 1)` from `item_pop_list`
- Loss: `torch.sum((label - pred)^2)` — sum reduction

New code `mecodgnn.py:_compute_l_reg`:
- Computes for ALL items, vectorized over T
- `F.mse_loss(reduction="mean")`

**Legacy keeps**: batch-only computation, loop over stages, sum reduction. Degree labels come from `item_degrees` (per-snapshot item degrees from `_build_time_snapshots`), stored as `log_deg_all` buffer same as new code.

### 7. Loss Combination

Old code `utils.py:BPRLoss.stageOne`:
```python
reg_loss = (1/2)*(u.norm(2)^2 + pos.norm(2)^2 + neg.norm(2)^2) / batch_size
loss1 = bpr_loss + weight_decay * reg_loss
loss = loss1 + dl * degree_loss + stage_dl * stage_degree_loss
```

**Legacy keeps**: manual L2 reg computation (not RecBole's EmbLoss), same formula. Config params:
- `reg_weight` → old `weight_decay` (L2 on raw embeddings)
- `beta` → old `dl` (L_mat weight)
- `alpha` → old `stage_dl` (L_reg weight)

## Architecture

```
MECoDGNNLEG(GeneralRecommender)
├── __init__(config, dataset)
│   ├── _build_time_snapshots(dataset)          # reuse from MECoDGNN
│   ├── user_embedding, item_embedding          # nn.Embedding
│   ├── fl: Linear(d, 2)                        # L_mat edge predictor
│   ├── fl2: Linear(d, 1)                       # L_reg degree predictor
│   ├── temporal_module (LSTM/RNN/Mean/Attn)    # reuse from MECoDGNN
│   ├── _compute_item_bins()                    # Lorenz-curve grouping
│   └── register buffers (log_deg_all, item_bins, target_dist)
│
├── _gnn_forward(norm_adj, n_layers)            # pure LightGCN per stage
│   └── for each layer: all_emb = A_hat @ all_emb
│       return mean(all layer outputs)
│
├── forward()
│   ├── for t in T: _gnn_forward(adj[t], n_layers_per_stage[t])
│   ├── _temporal_aggregate(stacked outputs)    # reuse from MECoDGNN
│   └── return user_emb, item_emb, time_step_item_embs
│
├── calculate_loss(interaction)
│   ├── forward() → user_emb, item_emb, time_step_item_embs
│   ├── residual: u_e = user_emb[u] + embedding_user(u)
│   ├── BPR loss (softplus)
│   ├── L2 reg: manual (1/2)*norm^2/batch_size
│   ├── L_reg: _compute_l_reg(time_step_item_embs, batch_items)
│   ├── L_mat: _compute_l_mat(user_emb, item_emb, batch_items)
│   └── loss = bpr + reg_weight*l2 + alpha*l_reg + beta*l_mat
│
├── predict(interaction)
├── full_sort_predict(interaction)
```

## Config (MECoDGNNLEG.yaml)

```yaml
embedding_size: 64
n_layers_per_stage: [3,3,3,3]   # per-stage GNN depth
n_time_steps: 4
lstm_hidden: 64
temporal_agg: rnn               # default to RNN to match old code
cumulative_graph: False
n_groups: 50                    # old default group_num
alpha: 1e-5                     # old stage_dl
beta: 2                         # old dl
gini_target: 0.75
reg_weight: 1e-3                # old decay
use_compile: False
use_tf32: False
```

## Implementation Plan

1. Create `recbole/properties/model/MECoDGNNLEG.yaml`
2. Create `recbole/model/general_recommender/mecodgnn_legacy.py` with class `MECoDGNNLEG`
3. Register in `recbole/model/general_recommender/__init__.py`
