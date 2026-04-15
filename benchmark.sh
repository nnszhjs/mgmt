#!/bin/bash
# ==========================================================================
#  benchmark.sh - Benchmark框架快捷启动脚本
# ==========================================================================
#
# 使用新的滑动窗口benchmark框架 (window_size + rounds)
#
# 用法:
#   ./benchmark.sh                    # 使用默认配置 benchmark_config.yaml
#   ./benchmark.sh my_config.yaml     # 使用自定义配置
#   ./benchmark.sh --test             # 运行快速测试
#   ./benchmark.sh --clean            # 清理旧数据
#   ./benchmark.sh --report           # 只生成报告
#   ./benchmark.sh --overwrite        # 强制重跑所有实验
#   ./benchmark.sh --force-resplit    # 强制重新生成窗口
#
# ==========================================================================

set -euo pipefail

# 默认配置文件
DEFAULT_CONFIG="benchmark_config.yaml"

# 解析参数
CONFIG_FILE="$DEFAULT_CONFIG"
MODE="run"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test)
            echo "运行快速测试..."
            exec ./test_benchmark.sh
            ;;
        --clean)
            echo "清理旧数据..."
            exec ./clean_benchmark.sh
            ;;
        --report)
            MODE="report"
            shift
            ;;
        --overwrite)
            EXTRA_ARGS="$EXTRA_ARGS --overwrite"
            shift
            ;;
        --force-resplit)
            EXTRA_ARGS="$EXTRA_ARGS --force-resplit"
            shift
            ;;
        *)
            CONFIG_FILE="$1"
            shift
            ;;
    esac
done

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    echo ""
    echo "提示: 从示例配置创建:"
    echo "  cp benchmark_config.yaml.example benchmark_config.yaml"
    exit 1
fi

echo "=========================================="
echo "Benchmark框架"
echo "=========================================="
echo "配置文件: $CONFIG_FILE"
echo "模式: $MODE"
echo "额外参数: $EXTRA_ARGS"
echo ""

# 运行benchmark
if [ "$MODE" = "report" ]; then
    python3 run_benchmark.py --config "$CONFIG_FILE" --report_only $EXTRA_ARGS
else
    python3 run_benchmark.py --config "$CONFIG_FILE" --continue_on_error $EXTRA_ARGS
fi
