#!/bin/bash

# DPO Training Script for Multiple Base Models
# Based on training techniques from the simplescaling/s1 repository
# SFT and evaluation methods referenced from: https://github.com/simplescaling/s1

# Script to run DPO training on different base models with specific data subsets
# Each model will be trained with a different subset of the dataset

# Define base models and their names (UPDATE THESE PATHS FOR YOUR SETUP)
declare -a MODEL_PATHS=(
    "./models/your_base_model"  # Replace with your actual model path
)

declare -a MODEL_NAMES=(
    "your_base_model"  # Replace with your model name
)

# Define the dataset path (UPDATE THIS PATH FOR YOUR SETUP)
DATASET_PATH="./data/your_dataset.parquet"  # Replace with your dataset path

# Set common parameters
MAX_STEPS=200
CONFIG_FILE="./configs/ds_config_zero3_10k.json"
SCRIPT_PATH="./dpo/combined_dpo.py"

# Use 4 GPUs for training (adjust based on your hardware)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# NCCL optimization settings
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BUFFSIZE=2097152
export NCCL_NSOCKS_PERTHREAD=4

# Disable bitsandbytes library
export BNB_CUDA_VERSION=0
export BNB_DISABLE_CUDA_KERNEL=1
export BITSANDBYTES_NOWELCOME=1
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# CUDA settings
export CUDA_DEVICE_MAX_CONNECTIONS=1

# NCCL debug and optimization settings
export NCCL_DEBUG=INFO
export NCCL_MIN_NCHANNELS=8
export NCCL_TIMEOUT=300
export CUDNN_DETERMINISTIC=1
export NCCL_ALGO=Tree

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

# Hugging Face settings
export HF_TRUST_REMOTE_CODE=1
export TOKENIZERS_PARALLELISM=false
export SAFETENSORS_FAST_GPU=1

# LoRA parameters
lora_r=256
lora_alpha=512

# Create temporary data files for different subsets
echo "Creating temporary data files for different subsets..."
PROCESSED_DIR="./processed_data"
mkdir -p $PROCESSED_DIR

# Create a log directory for the entire training process
LOG_DIR="./logs/multi_model_dpo_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR
echo "Starting multi-model DPO training at $(date)" | tee -a $LOG_DIR/combined_runs.log

# Create the dataset subsets
python3 -c "
import pandas as pd
import os

# Load the dataset (UPDATE THE PATH VARIABLE AS NEEDED)
df = pd.read_parquet('$DATASET_PATH')
print(f'Full dataset size: {len(df)} rows')

# Create subsets (modify as needed for your use case)
df_subset = df.copy()  # Use full dataset or create your own subsets

# Save the processed dataset
df_subset.to_parquet('$PROCESSED_DIR/processed_dataset.parquet')

print(f'Created processed dataset with {len(df_subset)} rows')
" | tee -a $LOG_DIR/data_preparation.log

# Define dataset paths for each model
declare -a DATA_PATHS=(
    "$PROCESSED_DIR/processed_dataset.parquet"
)

# Train each model with its corresponding dataset
for i in {0..0}; do  # Adjust range based on number of models
    MODEL_PATH=${MODEL_PATHS[$i]}
    MODEL_NAME=${MODEL_NAMES[$i]}
    DATA_PATH=${DATA_PATHS[$i]}
    
    echo "========================================================" | tee -a $LOG_DIR/combined_runs.log
    echo "Starting training for model $MODEL_NAME ($(($i+1))/1)" | tee -a $LOG_DIR/combined_runs.log
    echo "Base model: $MODEL_PATH" | tee -a $LOG_DIR/combined_runs.log
    echo "Dataset: $DATA_PATH" | tee -a $LOG_DIR/combined_runs.log
    echo "Started at: $(date)" | tee -a $LOG_DIR/combined_runs.log
    
    # Create output directory for this model (UPDATE OUTPUT PATH AS NEEDED)
    OUTPUT_DIR="./outputs/dpo_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p $OUTPUT_DIR
    
    # Verify model path exists
    if [ ! -d "$MODEL_PATH" ]; then
        echo "❌ Model path $MODEL_PATH does not exist, skipping!" | tee -a $LOG_DIR/combined_runs.log
        continue
    fi
    
    # Verify dataset path exists
    if [ ! -f "$DATA_PATH" ]; then
        echo "❌ Dataset file $DATA_PATH does not exist, skipping!" | tee -a $LOG_DIR/combined_runs.log
        continue
    fi
    
    # Set random port to avoid conflicts
    export MASTER_ADDR="localhost"
    export MASTER_PORT="$((RANDOM % 10000 + 50000))"
    echo "Using port: $MASTER_PORT" | tee -a $LOG_DIR/combined_runs.log
    
    # Run the training
    torchrun --nproc_per_node=4 \
        --master_port=$MASTER_PORT \
        $SCRIPT_PATH \
        --model_name $MODEL_PATH \
        --train_file_path $DATA_PATH \
        --output_dir $OUTPUT_DIR \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --learning_rate 5e-7 \
        --lr_scheduler_type cosine \
        --max_prompt_length 600 \
        --max_length 20000 \
        --max_steps $MAX_STEPS \
        --bf16 \
        --use_flash_attention \
        --gradient_checkpointing \
        --lora_r $lora_r \
        --lora_alpha $lora_alpha \
        --lora_dropout 0.05 \
        --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj" \
        --merge_weights \
        --prompt_field "question" \
        --chosen_field "text" \
        --rejected_field "worse-text" \
        --deepspeed=${CONFIG_FILE} \
        --logging_steps 10 \
        --eval_steps 100 2>&1 | tee -a $LOG_DIR/model_${MODEL_NAME}_training.log
    
    TRAINING_STATUS=$?
    if [ $TRAINING_STATUS -eq 0 ]; then
        echo "✅ Training completed successfully for model $MODEL_NAME at $(date)" | tee -a $LOG_DIR/combined_runs.log
    else
        echo "❌ Training failed for model $MODEL_NAME with exit code $TRAINING_STATUS at $(date)" | tee -a $LOG_DIR/combined_runs.log
    fi
    
    echo "Finished at: $(date)" | tee -a $LOG_DIR/combined_runs.log
    echo "========================================================" | tee -a $LOG_DIR/combined_runs.log
    echo "" | tee -a $LOG_DIR/combined_runs.log
done

echo "All training runs completed at $(date)" | tee -a $LOG_DIR/combined_runs.log
echo "Summary of training runs:" | tee -a $LOG_DIR/combined_runs.log
echo "- Model 1: ${MODEL_NAMES[0]} with processed dataset" | tee -a $LOG_DIR/combined_runs.log
echo "Logs saved to $LOG_DIR" | tee -a $LOG_DIR/combined_runs.log
