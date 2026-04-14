#!/bin/bash
# ==========================================================================
#  run_ddp.sh — RecBole 统一训练启动脚本（基于 torchrun）
# ==========================================================================
#
# 用法:
#   ./run_ddp.sh <GPU数> <模型名> <数据集> [配置文件...]
#
# 示例:
#   ./run_ddp.sh 1 BPR book                    # 单卡
#   ./run_ddp.sh 2 MECoDGNN book               # 双卡
#   ./run_ddp.sh 4 LightGCN amazon             # 四卡
#   ./run_ddp.sh 2 SASRec book my_config.yaml  # 带配置文件
#
# 说明:
#   所有运行统一通过 torchrun 启动，单卡时 nproc_per_node=1。
#   gpu_id 在 overall.yaml 或 config 文件中设置（如 gpu_id: 0,1）。
#   master_port 会自动随机分配以避免多任务冲突。
#
# --------------------------------------------------------------------------
#  可用模型（--model 参数，名称需与类名一致）
# --------------------------------------------------------------------------
#
#  RecBole 内置:
#    BPR, LightGCN, NGCF, GCMC, NeuMF, Pop, SpectralCF, SGL, NCL,
#    DGCF, ItemKNN, FISM, NAIS, ConvNCF, DMF, EASE, CDAE, MultiVAE,
#    MultiDAE, MacridVAE, RecVAE, RaCT, ENMF, LINE, SimpleX, DiffRec,
#    LDiffRec, NCEPLRec, ADMMSLIM, SLIMElastic, NNCF, Random, AsymKNN,
#    SASRec, GRU4Rec, BERT4Rec, Caser, STAMP, SRGNN, NARM, CORE,
#    FDSA, FOSSIL, FPMC, GCSAN, HGN, HRM, KSR, LightSANs, NextItNet,
#    NPE, RepeatNet, S3Rec, SHAN, SINE, TransRec, FEARec, DIN, DIEN, ...
#
#  新增模型（本项目实现）:
#  ┌────────────────┬─────────┬──────────────────────────────────────────┐
#  │ 模型名          │ 类型     │ 说明                                    │
#  ├────────────────┼─────────┼──────────────────────────────────────────┤
#  │ MECoDGNN       │ General │ 动态图 + Matthew 效应控制（核心模型）     │
#  │ DyGCN          │ General │ MECo-DGNN 骨架（仅 BPR，无正则化）       │
#  │ DDPG           │ General │ Actor-Critic 强化学习推荐                │
#  │ YouTubeDNN     │ General │ 双塔候选生成（均值池化 + MLP）           │
#  │ ComiRec        │ Sequent │ 多兴趣胶囊网络 Dynamic Routing           │
#  ├────────────────┼─────────┼──────────────────────────────────────────┤
#  │ BPR_PC         │ General │ BPR + 流行度补偿 (推理时减 log_pop)      │
#  │ BPR_PDA        │ General │ BPR + 流行度去偏 (训练加 pop loss)       │
#  │ LightGCN_PC    │ General │ LightGCN + 流行度补偿                   │
#  │ LightGCN_PDA   │ General │ LightGCN + 流行度去偏                   │
#  │ SASRec_PC      │ Sequent │ SASRec + 流行度补偿                     │
#  │ SASRec_PDA     │ Sequent │ SASRec + 流行度去偏 (脚注13: 只用最近item)│
#  └────────────────┴─────────┴──────────────────────────────────────────┘
#
# --------------------------------------------------------------------------
#  可用数据集（--dataset 参数，需在 dataset/ 目录下有对应文件夹）
# --------------------------------------------------------------------------
#
#    book          Book crossing
#    amazon        Amazon reviews
#    movielen      MovieLens (大规模)
#    ml-100k       MovieLens 100K (内置示例)
#
# ==========================================================================

set -euo pipefail

GPUS=${1:-2}
MODEL=${2:-MECoDGNN}
DATASET=${3:-book}
shift 3 2>/dev/null || true
CONFIG_FILES="$*"

# Auto-assign a free master_port to avoid conflicts when running multiple jobs
MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

torchrun \
    --nproc_per_node="$GPUS" \
    --master_port="$MASTER_PORT" \
    run_recbole.py \
    --model="$MODEL" \
    --dataset="$DATASET" \
    ${CONFIG_FILES:+--config_files=$CONFIG_FILES}
