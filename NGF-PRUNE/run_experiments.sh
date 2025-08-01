#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Config ---
DATASET="fashion_mnist"
MODELS=("lenet")
SEEDS=(42 123)
MAX_EPOCHS=50 # Max limit
PATIENCE=4 # Early stopping
LR=0.0005 # Learning rate
GPU_ID=0 # None for CPU
# NGF specific params
NGF_CRITICAL_STARTS=(1 5) # Epochs to start critical period
NGF_CRITICAL_DURATIONS=(3 5) # Durations (e.g., 5 means epochs 5,6,7,8,9)
NGF_THRESHOLDS=(0.95 0.90 0.85 0.80) # Fraction to keep

# Magnitude specific params
MAGNITUDE_PRUNE_EPOCHS=(5 10) # Epochs where magnitude pruning is applied
MAGNITUDE_THRESHOLDS=(0.05 0.10 0.15 0.20) # Fraction to prune

# Base Command
BASE_CMD="python main.py --dataset $DATASET --epochs $MAX_EPOCHS --early-stopping-patience $PATIENCE --lr $LR"

# Add GPU if specified
if [ ! -z "$GPU_ID" ]; then
    BASE_CMD="$BASE_CMD --gpu $GPU_ID"
fi

# Experiment Loop
TOTAL_RUNS=0

for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        # No Pruning (Baseline)
        echo "Running: $model, Seed: $seed, Pruning: none"
        CMD="$BASE_CMD --model $model --seed $seed --pruning-method none"
        echo "Executing: $CMD"
        $CMD
        TOTAL_RUNS=$((TOTAL_RUNS + 1))
        echo "---------------------------------"

        # NGF Pruning
        for start_epoch in "${NGF_CRITICAL_STARTS[@]}"; do
            for duration in "${NGF_CRITICAL_DURATIONS[@]}"; do
                for keep_fraction in "${NGF_THRESHOLDS[@]}"; do
                    pruning_apply_epoch=$((start_epoch + duration))
                    echo "Running: $model, Seed: $seed, Pruning: ngf, StartEpoch: $start_epoch, Duration: $duration (Apply @ $pruning_apply_epoch), KeepFraction: $keep_fraction"
                    CMD="$BASE_CMD --model $model --seed $seed --pruning-method ngf --critical-start-epoch $start_epoch --critical-duration $duration --pruning-threshold $keep_fraction"
                    echo "Executing: $CMD"
                    $CMD
                    TOTAL_RUNS=$((TOTAL_RUNS + 1))
                    echo "---------------------------------"
                done
            done
        done

        # Magnitude Pruning
        for prune_epoch in "${MAGNITUDE_PRUNE_EPOCHS[@]}"; do
            for threshold in "${MAGNITUDE_THRESHOLDS[@]}"; do
                echo "Running: $model, Seed: $seed, Pruning: magnitude, PruneEpoch: $prune_epoch, Threshold: $threshold"
                CMD="$BASE_CMD --model $model --seed $seed --pruning-method magnitude --pruning-apply-epoch $prune_epoch --pruning-threshold $threshold"
                echo "Executing: $CMD"
                $CMD
                TOTAL_RUNS=$((TOTAL_RUNS + 1))
                echo "---------------------------------"
            done
        done
    done
done

echo "Finished running $TOTAL_RUNS experiments."

# Trigger Plotting
echo "Running plotting script..."
python plot_results.py --results-dir results --output-dir results/comparison_plots
echo "Plotting complete. Check the 'results/comparison_plots' directory." 