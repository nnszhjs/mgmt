#!/bin/bash
# ==========================================================================
#  run_ddp.sh — RecBole 多卡分布式训练启动脚本
# ==========================================================================
#
# 用法:
#   ./run_ddp.sh <GPU数> <模型名> <数据集>
#   ./run_ddp.sh 4 MECoDGNN ml-100k
#   ./run_ddp.sh 2 BPR_PC ml-1m
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
#  PC（Popularity Compensation）:
#    继承自 base model，训练不变，推理时 score' = score - pc_lambda * log(pop+1)
#    超参: pc_lambda (默认 1.0)
#
#  PDA（Popularity-bias Deconfounding and Adjusting）:
#    继承自 base model，训练时额外加 popularity BPR loss，推理时只用原始分数
#    超参: pda_weight (默认 1.0)
#    注: SASRec_PDA 遵循论文脚注13，只对序列最近一个 item 做 causal learning
#
#  运行示例:
#    ./run_ddp.sh 2 BPR_PC ml-100k
#    ./run_ddp.sh 2 LightGCN_PDA ml-100k
#    ./run_ddp.sh 2 SASRec_PC ml-100k
#
# --------------------------------------------------------------------------
#  可用数据集（--dataset 参数，需在 dataset/ 目录下有对应文件夹）
# --------------------------------------------------------------------------
#
#    ml-100k       MovieLens 100K
#    ml-1m         MovieLens 1M（需自行下载放入 dataset/ml-1m/）
#    amazon-books  Amazon Books（需自行下载）
#    yelp2022      Yelp 2022（需自行下载）
#    gowalla       Gowalla check-in（需自行下载）
#
# ==========================================================================

GPUS=${1:-2}
MODEL=${2:-MECoDGNN}
DATASET=${3:-book}
# CONFIG_FILES=${4:-}

torchrun \
    --nproc_per_node=$GPUS \
    --master_port=29500 \
    run_recbole.py \
    --model=$MODEL \
    --dataset=$DATASET
    # \
    # ${CONFIG_FILES:+--config_files=$CONFIG_FILES}
