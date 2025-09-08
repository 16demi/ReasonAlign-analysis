#!/bin/bash

# DPO Model Analysis Script
# Based on analysis techniques from various open-source projects
# SFT and evaluation methods adapted from: https://github.com/simplescaling/s1

# UPDATE THESE PATHS FOR YOUR SETUP
MODEL_PATH="./models/your_dpo_model"  # Replace with your DPO model path
DATASET_PATH="./data/your_dataset.parquet"  # Replace with your dataset path
OUTPUT_DIR="./results/dpo_analysis"  # Replace with your output directory

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Make the analysis script executable
chmod +x ./analysis/analyze_dpo_model.py

# Run the analysis on each GPU in parallel (adjust GPU count as needed)
echo "Starting DPO model analysis across multiple GPUs..."

for gpu_id in {0..6}; do
    echo "Starting analysis on GPU $gpu_id..."
    python ./analysis/analyze_dpo_model.py \
        --model_path "$MODEL_PATH" \
        --dataset_path "$DATASET_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --gpu_id $gpu_id > "$OUTPUT_DIR/gpu_${gpu_id}.log" 2>&1 &
    
    # Add a small delay to avoid potential resource conflicts
    sleep 5
done

# Wait for all background processes to finish
echo "Waiting for all GPU analyses to complete..."
wait

# Combine results into a single ordered CSV file
echo "Combining results into a single ordered CSV file..."
python - << 'EOF'
import os
import pandas as pd

# Output directory
output_dir = "./results/dpo_analysis"

# List to store all results
all_results = []

# Read each GPU's results
for gpu_id in range(7):
    csv_path = os.path.join(output_dir, f"gpu_{gpu_id}.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        all_results.append(df)
    else:
        print(f"Warning: Results for GPU {gpu_id} not found at {csv_path}")

# Combine all results
if all_results:
    combined_df = pd.concat(all_results)
    
    # Sort by epoch (which is GPU ID + 1)
    combined_df = combined_df.sort_values('epoch')
    
    # Save to CSV
    combined_path = os.path.join(output_dir, "combined_results.csv")
    combined_df.to_csv(combined_path, index=False)
    print(f"Combined results saved to {combined_path}")
else:
    print("No results found to combine")
EOF

echo "Analysis complete! Results are saved in $OUTPUT_DIR"
echo "Combined results are available in $OUTPUT_DIR/combined_results.csv"
