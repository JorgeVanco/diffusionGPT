#!/bin/bash

# 1. Configuration
NUM_GPUS=4
TRIALS_PER_GPU=1
SCRIPT_NAME="sweep.py"

# Calculate total trials
TOTAL_TRIALS=$((NUM_GPUS * TRIALS_PER_GPU))

# 2. Cleanup Function
# This runs if you press Ctrl+C, killing all child worker processes
cleanup() {
    echo ""
    echo "🛑 Caught signal! Killing all workers..."
    kill $(jobs -p) 2>/dev/null
    exit 1
}

# Register the cleanup function for SIGINT (Ctrl+C) and SIGTERM
trap cleanup SIGINT SIGTERM

echo "🚀 Launching $NUM_GPUS parallel Optuna workers..."
echo "📊 Target: $TRIALS_PER_GPU trials per GPU (Total: $TOTAL_TRIALS trials)"
echo "---------------------------------------------------"

# 3. Create a logs directory
mkdir -p logs

# 4. Launch Loop
for ((i=0; i<NUM_GPUS; i++)); do
    echo "  -> Starting worker $i on GPU $i (Log: logs/worker_$i.log)"

    CUDA_VISIBLE_DEVICES=$i uv run $SCRIPT_NAME --n_trials $TRIALS_PER_GPU > logs/worker_$i.log 2>&1 &
done

echo "---------------------------------------------------"
echo "✅ All workers launched."
echo "   - View progress: tail -f logs/worker_0.log"
echo "   - Stop execution: Press Ctrl+C"
echo "   - Monitor optuna dashboard: optuna-dashboard sqlite:///db.sqlite3"

# 5. Wait for all background jobs to finish
wait