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
#
# ==========================================================================

set -euo pipefail

# 默认配置文件
DEFAULT_CONFIG="benchmark_config.yaml"

# 解析参数
if [ $# -eq 0 ]; then
    CONFIG_FILE="$DEFAULT_CONFIG"
    MODE="run"
elif [ "$1" = "--test" ]; then
    echo "运行快速测试..."
    exec ./test_benchmark.sh
elif [ "$1" = "--clean" ]; then
    echo "清理旧数据..."
    exec ./clean_benchmark.sh
elif [ "$1" = "--report" ]; then
    CONFIG_FILE="${2:-$DEFAULT_CONFIG}"
    MODE="report"
else
    CONFIG_FILE="$1"
    MODE="run"
fi

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
echo ""

# 运行benchmark
if [ "$MODE" = "report" ]; then
    python3 run_benchmark.py --config "$CONFIG_FILE" --report_only
else
    python3 run_benchmark.py --config "$CONFIG_FILE" --continue_on_error
fi
