#!/bin/bash

# Complete Analysis Pipeline Script
# Based on training analysis techniques from various open-source projects
# SFT and evaluation methods adapted from: https://github.com/simplescaling/s1

# UPDATE THESE PATHS FOR YOUR SETUP
BASE_MODEL="./models/your_base_model"  # Replace with your base model path
CHECKPOINT_DIR="./checkpoints/your_experiment/checkpoints"  # Replace with your checkpoint directory
DATASET_PATH="./data/your_dataset.parquet"  # Replace with your dataset path
OUTPUT_DIR="./results/your_experiment"  # Replace with your output directory
EPOCHS="1,2,3,4,5,6"  # Modify as needed for your experiment

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Step 1: Run the analysis
echo "Starting training dynamics analysis..."
python ./analysis/analyze_training_dynamics.py \
    --base_model "$BASE_MODEL" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS"

# Step 2: Run the visualization
echo "Starting visualization of training dynamics..."
python ./visualization/visualize_training_dynamics.py \
    --results_dir "$OUTPUT_DIR"

echo "Complete analysis finished! Results and visualizations are saved in $OUTPUT_DIR"
