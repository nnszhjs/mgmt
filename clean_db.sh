#!/bin/bash
# Clean benchmark database to rebuild with new schema

echo "Cleaning benchmark database..."

if [ -f "benchmark_results/benchmark.db" ]; then
    echo "Removing benchmark_results/benchmark.db"
    rm -f benchmark_results/benchmark.db
    rm -f benchmark_results/benchmark.db-shm
    rm -f benchmark_results/benchmark.db-wal
    echo "✓ Database removed"
else
    echo "No database found, nothing to clean"
fi

echo ""
echo "Database will be recreated with new schema on next run."
echo "New unique key: (dataset, model, window_size, rounds, window_idx, seed)"
