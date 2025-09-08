#!/usr/bin/env python3
"""
Training Dynamics Analysis for SFT Models

This script analyzes training dynamics by computing log probabilities for forward 
and reverse reasoning completions across different epochs of supervised fine-tuning.
It tracks how model preferences evolve during the training process.

Based on analysis techniques from various open-source projects.
Training infrastructure and evaluation methods adapted from: https://github.com/simplescaling/s1

Features:
- Log probability analysis across training epochs
- Support for LoRA adapter loading with PEFT
- Batch processing with comprehensive error handling
- CSV output with epoch-by-epoch tracking
- Memory optimization for multi-epoch analysis
"""
# Analysis of model probability distributions for forward and reverse reasoning during training
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
import gc
import logging

# Configure logging
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
        

        prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        completion_ids = tokenizer.encode(completion, return_tensors='pt').to(device)
        
        if completion_ids.shape[1] > max_tokens:
            completion_ids = completion_ids[:, :max_tokens]
        if completion_ids.shape[1] > 1:
            full_ids = torch.cat([prompt_ids, completion_ids[:, 1:]], dim=1)
        else:
            full_ids = prompt_ids
            logger.warning(f"Completion too short: {completion}")
            return float('-inf'), float('-inf'), []
            
        with torch.no_grad():
            outputs = model(full_ids)
            logits = outputs.logits
        
        shift_logits = logits[:, prompt_ids.shape[1]-1:-1, :]
        shift_labels = full_ids[:, prompt_ids.shape[1]:]
        
        log_probs = []
        for i in range(shift_labels.shape[1]):
            logit = shift_logits[:, i, :]
            label = shift_labels[:, i]
            log_prob = torch.log_softmax(logit, dim=-1).gather(1, label.unsqueeze(-1)).item()
            log_probs.append(log_prob)
        
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

def load_model_and_tokenizer(base_model_path, adapter_path=None):
    """Load the model and tokenizer with optional adapter."""
    logger.info(f"Loading base model from {base_model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True
        )
        if adapter_path:
            logger.info(f"Loading adapter from {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
        
        model.eval()
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def format_prompt(row):
    """Format prompt from the dataset row."""
    try:
        prompt = f"<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>user\n{row['question']}<|im_end|>\n<|im_start|>assistant\n"
        return prompt
    except KeyError:
        logger.error(f"Missing 'question' field in row: {row}")
        return "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>user\nSolve this problem.\n<|im_end|>\n<|im_start|>assistant\n"

def analyze_dataset(model, tokenizer, dataset, subset_name, output_path, epoch):
    """Analyze all examples in the dataset and save results."""
    results = []
    
    for i, row in tqdm(enumerate(dataset), total=len(dataset), desc=f"Processing {subset_name}"):
        try:
            prompt = format_prompt(row)
            if 'text' not in row or 'worse-text' not in row:
                logger.warning(f"Missing required fields in row {i}: {row.keys()}")
                continue
            
            text_avg_log_prob, text_peak_log_prob, text_log_probs = calculate_log_probs(
                model, tokenizer, prompt, row['text'])
            worse_avg_log_prob, worse_peak_log_prob, worse_log_probs = calculate_log_probs(
                model, tokenizer, prompt, row['worse-text'])
            
            avg_diff = text_avg_log_prob - worse_avg_log_prob
            peak_diff = text_peak_log_prob - worse_peak_log_prob
            if text_avg_log_prob != float('-inf') and worse_avg_log_prob != float('-inf'):
                results.append({
                    'example_id': i,
                    'epoch': epoch,
                    'text_avg_log_prob': text_avg_log_prob,
                    'worse_avg_log_prob': worse_avg_log_prob,
                    'text_peak_log_prob': text_peak_log_prob,
                    'worse_peak_log_prob': worse_peak_log_prob,
                    'avg_diff': avg_diff,
                    'peak_diff': peak_diff,
                    'tokens_analyzed': min(len(text_log_probs), len(worse_log_probs))
                })
            else:
                logger.warning(f"Skipping row {i} due to invalid probability values")
                
        except Exception as e:
            logger.error(f"Error processing row {i}: {str(e)}")
    
    if results:
        df = pd.DataFrame(results)
        
        csv_path = os.path.join(output_path, f"{subset_name}.csv")
        if os.path.exists(csv_path):
            try:
                existing_df = pd.read_csv(csv_path)
                df = df[~df.apply(lambda x: ((existing_df['epoch'] == x['epoch']) & 
                                             (existing_df['example_id'] == x['example_id'])).any(), axis=1)]
                if not df.empty:
                    combined_df = pd.concat([existing_df, df])
                    combined_df.to_csv(csv_path, index=False)
                    logger.info(f"Added {len(df)} new entries to {csv_path}")
                else:
                    logger.info(f"No new entries to add to {csv_path}")
            except Exception as e:
                logger.error(f"Error when appending to existing file: {str(e)}")
                os.rename(csv_path, f"{csv_path}.bak")
                df.to_csv(csv_path, index=False)
                logger.info(f"Created new file {csv_path} (old file backed up)")
        else:
            df.to_csv(csv_path, index=False)
            logger.info(f"Results saved to {csv_path}")
        
        return df
    else:
        logger.warning(f"No valid results for {subset_name}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Analyze training dynamics across epochs")
    parser.add_argument("--base_model", type=str, required=True, 
                        help="Path to the base model")
    parser.add_argument("--checkpoint_dir", type=str, required=True, 
                        help="Directory containing epoch checkpoints")
    parser.add_argument("--dataset_path", type=str, required=True, 
                        help="Path to the dataset parquet file")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save results")
    parser.add_argument("--epochs", type=str, default="1,2,3,4,5,6", 
                        help="Comma-separated list of epochs to analyze")
    parser.add_argument("--log_file", type=str, default="", 
                        help="Path to log file (if not provided, logs to stdout)")
    
    args = parser.parse_args()

    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        logger.info(f"Loading dataset from {args.dataset_path}")
        df = pd.read_parquet(args.dataset_path)
        logger.info(f"Loaded dataset with {len(df)} samples")
        required_columns = ['question', 'text', 'worse-text']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Dataset missing required columns: {missing_columns}")
            return
        first_50 = df.iloc[:50].to_dict('records')
        last_50 = df.iloc[50:100].to_dict('records') if len(df) >= 100 else df.iloc[50:].to_dict('records')
        all_samples = df.to_dict('records')
        
        logger.info(f"Created subsets: first_50 ({len(first_50)} samples), last_50 ({len(last_50)} samples), all ({len(all_samples)} samples)")
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return
    try:
        epochs = [int(e) for e in args.epochs.split(',')]
        logger.info(f"Will analyze epochs: {epochs}")
    except Exception as e:
        logger.error(f"Error parsing epochs: {str(e)}")
        return
    
    for epoch in epochs:
        try:
            checkpoint_path = os.path.join(args.checkpoint_dir, str(epoch))
            
            if not os.path.exists(checkpoint_path):
                logger.error(f"Checkpoint path does not exist: {checkpoint_path}")
                continue
                
            model, tokenizer = load_model_and_tokenizer(args.base_model, checkpoint_path)
            logger.info(f"Starting analysis for epoch {epoch}")
            analyze_dataset(model, tokenizer, first_50, "first50", args.output_dir, epoch)
            analyze_dataset(model, tokenizer, last_50, "last50", args.output_dir, epoch)
            analyze_dataset(model, tokenizer, all_samples, "all", args.output_dir, epoch)
            
            del model
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error analyzing epoch {epoch}: {str(e)}")
    
    logger.info("Analysis complete!")

if __name__ == "__main__":
    main()
