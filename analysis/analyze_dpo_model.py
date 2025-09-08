#!/usr/bin/env python3
"""
DPO Model Analysis for Forward vs. Reverse Reasoning

This script analyzes Direct Preference Optimization (DPO) models by computing
log probabilities for forward and reverse reasoning completions. It provides
detailed analysis of model preferences and probability distributions.

Based on analysis techniques from the simplescaling/s1 repository.
SFT and evaluation code referenced from: https://github.com/simplescaling/s1

Features:
- Log probability calculation for model completions
- Support for Qwen model formatting and tokenization
- Batch processing of datasets with error handling
- Comprehensive logging and progress tracking
- GPU memory optimization with garbage collection
"""
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import gc
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def calculate_log_probs(model, tokenizer, prompt, completion, max_tokens=1000):
    """
    Calculate average log probability for the completion given the prompt.
    Only consider the first max_tokens tokens of the completion.
    """
    if not isinstance(completion, str) or not completion.strip():
        logger.warning(f"Empty or invalid completion received: {completion}")
        return float('-inf'), float('-inf'), []
    
    try:
        device = next(model.parameters()).device
        
        # Encode prompt and completion
        prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        completion_ids = tokenizer.encode(completion, return_tensors='pt').to(device)
        
        # Limit maximum number of tokens for completion
        if completion_ids.shape[1] > max_tokens:
            completion_ids = completion_ids[:, :max_tokens]
        
        # Complete sequence is prompt+completion (skip first token of completion, usually BOS)
        if completion_ids.shape[1] > 1:
            full_ids = torch.cat([prompt_ids, completion_ids[:, 1:]], dim=1)
        else:
            # If completion is too short, only use prompt
            full_ids = prompt_ids
            logger.warning(f"Completion too short: {completion}")
            return float('-inf'), float('-inf'), []
            
        # Get logits for the entire sequence
        with torch.no_grad():
            outputs = model(full_ids)
            logits = outputs.logits
        
        # Only keep logits for predicted completion tokens
        shift_logits = logits[:, prompt_ids.shape[1]-1:-1, :]
        shift_labels = full_ids[:, prompt_ids.shape[1]:]
        
        # Calculate log probabilities
        log_probs = []
        for i in range(shift_labels.shape[1]):
            logit = shift_logits[:, i, :]
            label = shift_labels[:, i]
            log_prob = torch.log_softmax(logit, dim=-1).gather(1, label.unsqueeze(-1)).item()
            log_probs.append(log_prob)
        
        # Calculate average log probability and peak probability
        if log_probs:
            avg_log_prob = sum(log_probs) / len(log_probs)
            peak_log_prob = max(log_probs)
        else:
            avg_log_prob = float('-inf')
            peak_log_prob = float('-inf')
            
        return avg_log_prob, peak_log_prob, log_probs
        
    except Exception as e:
        logger.error(f"Error calculating log probs: {str(e)}")
        return float('-inf'), float('-inf'), []

def format_prompt(row):
    """Format prompt from the dataset row."""
    # Use Qwen model formatting
    try:
        prompt = f"<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>user\n{row['question']}<|im_end|>\n<|im_start|>assistant\n"
        return prompt
    except KeyError:
        logger.error(f"Missing 'question' field in row: {row}")
        return "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>user\nSolve this problem.\n<|im_end|>\n<|im_start|>assistant\n"

def analyze_dataset(model, tokenizer, dataset, gpu_id, output_dir):
    """Analyze all examples in the dataset and save results."""
    logger.info(f"Analyzing dataset on GPU {gpu_id}")
    
    results = []
    tokens_analyzed = []
    valid_count = 0
    
    for i, example in enumerate(tqdm(dataset, desc=f"Processing on GPU {gpu_id}")):
        try:
            prompt = format_prompt(example)
            
            # Check required fields
            if 'text' not in example or 'worse-text' not in example:
                logger.warning(f"Missing required fields in example {i}")
                continue
            
            # Calculate log probability for text field
            forward_avg_log_prob, forward_peak_log_prob, _ = calculate_log_probs(
                model, tokenizer, prompt, example['text']
            )
            
            # Calculate log probability for worse-text field
            reverse_avg_log_prob, reverse_peak_log_prob, _ = calculate_log_probs(
                model, tokenizer, prompt, example['worse-text']
            )
            
            # Calculate difference
            log_prob_diff = forward_avg_log_prob - reverse_avg_log_prob
            peak_diff = forward_peak_log_prob - reverse_peak_log_prob
            
            # Calculate token count
            forward_tokens = len(tokenizer.encode(example['text']))
            reverse_tokens = len(tokenizer.encode(example['worse-text']))
            avg_tokens = (forward_tokens + reverse_tokens) / 2
            tokens_analyzed.append(avg_tokens)
            
            valid_count += 1
            
        except Exception as e:
            logger.error(f"Error processing example {i}: {str(e)}")
            continue
    
    if valid_count > 0:
        # Calculate averages
        avg_results = {
            'epoch': gpu_id + 1,  # Use GPU ID + 1 as epoch number
            'text_avg_log_prob': forward_avg_log_prob,
            'text_peak_log_prob': forward_peak_log_prob,
            'worse_avg_log_prob': reverse_avg_log_prob,
            'worse_peak_log_prob': reverse_peak_log_prob,
            'avg_diff': log_prob_diff,
            'peak_diff': peak_diff,
            'tokens_analyzed_avg': sum(tokens_analyzed) / len(tokens_analyzed) if tokens_analyzed else 0,
            'valid_samples': valid_count
        }
        
        # Save averages to CSV
        avg_df = pd.DataFrame([avg_results])
        csv_path = os.path.join(output_dir, f"gpu_{gpu_id}.csv")
        avg_df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to {csv_path}")
        
        return avg_df
    else:
        logger.warning(f"No valid results for GPU {gpu_id}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Analyze DPO model dynamics")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the DPO model")
    parser.add_argument("--dataset_path", type=str, required=True, 
                        help="Path to the dataset parquet file")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save results")
    parser.add_argument("--gpu_id", type=int, required=True, 
                        help="GPU ID to use for this analysis")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device(f"cuda:0")
    
    # Load dataset
    try:
        logger.info(f"Loading dataset from {args.dataset_path}")
        df = pd.read_parquet(args.dataset_path)
        logger.info(f"Loaded dataset with {len(df)} samples")
        
        # Check required columns
        required_columns = ['question', 'text', 'worse-text']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Dataset missing required columns: {missing_columns}")
            return
            
        # Use all data
        all_samples = df.to_dict('records')
        logger.info(f"Using all {len(all_samples)} samples")
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return
    
    try:
        # Load model
        logger.info(f"Loading model from {args.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        
        # Analyze dataset
        logger.info(f"Starting analysis on GPU {args.gpu_id}")
        analyze_dataset(model, tokenizer, all_samples, args.gpu_id, args.output_dir)
        
        # Free memory
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
    
    logger.info(f"Analysis on GPU {args.gpu_id} complete!")

if __name__ == "__main__":
    main()
