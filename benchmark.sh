# 基本用法：2模型 × 1数据集 × 2轮 × 3种子                     
python3 run_benchmark.py \
    --models BPR LightGCN \
    --datasets book \
    --rounds 2 \
    --train_splits 3 \
    --seeds 2022 2023 2024 2025 2026
                                            