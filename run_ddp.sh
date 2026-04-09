#!/bin/bash
# Multi-GPU training using torchrun

GPUS=${1:-2}
MODEL=${2:-MECoDGNN}
DATASET=${3:-movielen}
CONFIG_FILES=${4:-}

torchrun \
    --nproc_per_node=$GPUS \
    --master_port=29500 \
    run_recbole.py \
    --model=$MODEL \
    --dataset=$DATASET \
    ${CONFIG_FILES:+--config_files=$CONFIG_FILES}
