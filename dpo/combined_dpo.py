#!/usr/bin/env python3
"""
Combined DPO (Direct Preference Optimization) Training Script

This script implements DPO training for language models with support for:
- Multi-GPU distributed training with DeepSpeed ZeRO-3
- LoRA (Low-Rank Adaptation) fine-tuning
- Flash Attention 2 optimization
- Custom data formatting and preprocessing
- Comprehensive logging and checkpointing

Based on DPO implementation techniques from various open-source projects.
Training infrastructure and evaluation methods adapted from: https://github.com/simplescaling/s1

Usage:
    torchrun --nproc_per_node=4 combined_dpo.py --model_name /path/to/model --train_file_path /path/to/data.parquet

For detailed usage instructions and parameter explanations, see the README.md file.
"""

import os
import sys
import logging
import torch
import json
import argparse
from datasets import load_dataset, Dataset
import pandas as pd
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
import trl
from trl import DPOTrainer
from trl.trainer import DPOConfig
from trl.trainer.utils import DPODataCollatorWithPadding
from typing import Dict, Optional, List, Any
from datetime import datetime
import safetensors.torch
import deepspeed



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="DPO training with TRL, combined with direct model loading")
    
    # Model and data parameters
    parser.add_argument("--model_name", type=str, required=True, 
                        help="Path to SFT fine-tuned model")
    parser.add_argument("--train_file_path", type=str, required=True, 
                        help="Path to training data parquet file")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save model")
    
    # Training parameters
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="Training batch size per GPU")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1,
                        help="Evaluation batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        help="Learning rate scheduler type")
    parser.add_argument("--warmup_steps", type=int, default=20,
                        help="Learning rate warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.05,
                        help="Weight decay")
    parser.add_argument("--max_steps", type=int, default=50,
                        help="Maximum training steps")
    parser.add_argument("--max_prompt_length", type=int, default=1024,
                        help="Maximum prompt length")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="Beta parameter in DPO loss")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--save_steps", type=int, default=7,
                        help="Save checkpoint every X steps")
    parser.add_argument("--save_total_limit", type=int, default=7,
                        help="Limit total number of checkpoints")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log every X steps")
    parser.add_argument("--eval_steps", type=int, default=200,
                        help="Evaluate every X steps")
    
    # Mixed precision and optimization parameters
    parser.add_argument("--bf16", action="store_true", default=False, 
                        help="Whether to use bf16 mixed precision")
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="Whether to use fp16 mixed precision")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False,
                        help="Whether to use gradient checkpointing")
    
    # LoRA parameters
    parser.add_argument("--use_lora", action="store_true", default=False,
                        help="Whether to use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=64,
                        help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=128,
                        help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout probability")
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj", 
                        help="LoRA target modules")
    
    # Flash Attention parameters
    parser.add_argument("--use_flash_attention", action="store_true", default=False,
                        help="Whether to use Flash Attention 2")
    
    # DeepSpeed configuration
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="Path to DeepSpeed configuration file")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training")
    
    # Merge weights parameter
    parser.add_argument("--merge_weights", action="store_true", default=True,
                        help="Whether to save merged complete model")
                        
    # Dataset field mapping
    parser.add_argument("--prompt_field", type=str, default="question",
                        help="Field name for input prompts")
    parser.add_argument("--chosen_field", type=str, default="text",
                        help="Field name for preferred answers") 
    parser.add_argument("--rejected_field", type=str, default="worse-text",
                        help="Field name for less preferred answers")
    
    return parser.parse_args()

def load_model_and_tokenizer(args, is_reference_model=False):
    """
    Load model and tokenizer, ensuring consistency with the approach used in s1 training
    """
    logger.info(f"Loading {('reference ' if is_reference_model else '')}model from {args.model_name}")
    logger.info("Using the same method as s1 to load the model")
    
    # Check if it's a local path
    if not os.path.exists(args.model_name):
        logger.info(f"Model path '{args.model_name}' doesn't exist, trying to load from Hugging Face Hub")
        local_files_only = False
    else:
        logger.info(f"Loading model from local path: {args.model_name}")
        local_files_only = True
    
    # Distributed training compatible model loading method
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    logger.info(f"Using distributed training compatible model loading method, local_rank={local_rank}")
    
    # Use Flash Attention 2
    logger.info("Using Flash Attention 2")
    attn_implementation = "flash_attention_2"
    
    # Use gradient checkpointing
    logger.info("Using gradient checkpointing")
    
    try:
        # Load model
        logger.info(f"Starting to load model, parameters: torch_dtype=auto, trust_remote_code=True, local_files_only={local_files_only}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype="auto",
            trust_remote_code=True,
            local_files_only=local_files_only,
            use_auth_token=False,
            revision=None,
            attn_implementation=attn_implementation,
            device_map={"": local_rank} if local_rank != -1 else None,
        )
        logger.info("Model loaded successfully")
        
        # Enable gradient checkpointing to save GPU memory
        if not is_reference_model:
            logger.info("Enabling gradient checkpointing")
            model.gradient_checkpointing_enable()
        
        # Load tokenizer
        logger.info("Starting to load tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, 
            trust_remote_code=True,
            use_fast=False,
            padding_side="left"
        )
        tokenizer.padding_side = 'left' 
        logger.info("Tokenizer loaded successfully")
        
        # Set pad token
        if "Llama" in args.model_name:
            logger.info("Detected Llama model, setting pad_token to <|reserved_special_token_5|>")
            tokenizer.pad_token = "<|reserved_special_token_5|>"
        elif "Qwen" in args.model_name or "qwen" in args.model_name:
            logger.info("Detected Qwen model, setting pad_token to <|fim_pad|>")
            tokenizer.pad_token = "<|fim_pad|>"
        else:
            logger.info("No specific model detected, defaulting pad_token to <|fim_pad|>")
            tokenizer.pad_token = "<|fim_pad|>"
        
        logger.info(f"Tokenizer configuration complete, pad_token set to: {tokenizer.pad_token}")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model or tokenizer: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def preprocess_dataset(data_path, prompt_field="question", chosen_field="text", rejected_field="worse-text", 
                       max_prompt_length=None, max_length=None):
    """
    Preprocess dataset from parquet file to TRL DPO format
    """
    logger.info(f"Loading data from {data_path}")
    try:
        # Load parquet file
        if data_path.endswith('.parquet'):
            logger.info("Detected parquet file, using parquet loader")
            try:
                dataset = load_dataset('parquet', data_files=data_path)
            except Exception as e:
                logger.error(f"Failed to use parquet loader: {e}")
                df = pd.read_parquet(data_path)
                logger.info(f"Successfully loaded {len(df)} rows of data using pandas")
                dataset = Dataset.from_pandas(df)
                dataset = {'train': dataset}
        else:
            # For other formats, try using load_dataset directly
            logger.info("Attempting to use generic dataset loader")
            try:
                dataset = load_dataset(data_path)
            except Exception as e:
                logger.error(f"Generic loader failed: {e}")
                raise
        
        # Ensure dataset has a train split
        if 'train' not in dataset:
            logger.info("Dataset doesn't have a train split, reorganizing")
            tmp_dataset = dataset
            dataset = {}
            dataset['train'] = tmp_dataset
        
        logger.info(f"Successfully loaded dataset with {len(dataset['train'])} samples")
        logger.info(f"Dataset fields: {dataset['train'].column_names}")
        
        # Check sample types
        if 'sample_type' in dataset['train'].column_names:
            sample_types = dataset['train']['sample_type']
            counter = {}
            for st in sample_types:
                if st in counter:
                    counter[st] += 1
                else:
                    counter[st] = 1
            logger.info(f"Found 'sample_type' column, distribution: {counter}")
        
        # Convert to pandas DataFrame to avoid Arrow type issues
        train_df = dataset['train'].to_pandas()
        
        # Apply sequence length limits
        if max_prompt_length is not None:
            logger.info(f"Limiting prompt length to {max_prompt_length}")
            train_df[prompt_field] = train_df[prompt_field].astype(str).str[:max_prompt_length]
        
        if max_length is not None:
            logger.info(f"Limiting total length to {max_length}")
            # Calculate the length of each prompt
            prompt_lens = train_df[prompt_field].astype(str).str.len()
            
            # Limit chosen and rejected lengths to ensure prompt+response doesn't exceed max_length
            for i, pl in enumerate(prompt_lens):
                max_resp_len = max(1, max_length - pl)
                train_df.at[i, chosen_field] = str(train_df.at[i, chosen_field])[:max_resp_len]
                train_df.at[i, rejected_field] = str(train_df.at[i, rejected_field])[:max_resp_len]
        
        # Create formatted dataset
        formatted_data = {
            "prompt": train_df[prompt_field].tolist(),
            "chosen": train_df[chosen_field].tolist(),
            "rejected": train_df[rejected_field].tolist(),
        }
        
        # Convert to Dataset
        formatted_dataset = Dataset.from_dict(formatted_data)
        
        # Create train/eval splits
        splits = formatted_dataset.train_test_split(test_size=50, seed=42)
        dpo_dataset = {
            'train': splits['train'],
            'eval': splits['test']
        }
            
        logger.info(f"Created DPO dataset with {len(dpo_dataset['train'])} training samples and {len(dpo_dataset['eval'])} evaluation samples")
        return dpo_dataset
        
    except Exception as e:
        logger.error(f"Error during data preprocessing: {e}")
        raise

class CustomDPOTrainer(DPOTrainer):
    """
    Custom DPOTrainer class for better compatibility with directly loaded Qwen2 models
    """
    def __init__(self, *args, **kwargs):
        # Remove parameters not accepted by DPOTrainer from kwargs
        prompt_template = kwargs.pop('prompt_template', None)
        response_template = kwargs.pop('response_template', None)
        tokenizer = kwargs.pop('tokenizer', None)
        max_prompt_length = kwargs.pop('max_prompt_length', None)
        max_length = kwargs.pop('max_length', None)
        
        # Ensure tokenizer is correctly set as processing_class
        if tokenizer is not None:
            kwargs['processing_class'] = tokenizer
            logger.info(f"Setting tokenizer as processing_class: {tokenizer.__class__.__name__}")
        
        logger.info("Initializing custom DPOTrainer, removed unsupported parameters")
        super().__init__(*args, **kwargs)
        
        # Save templates
        self.prompt_template = prompt_template
        self.response_template = response_template
        self.tokenizer = tokenizer
        tokenizer.padding_side = 'left'  # Ensure tokenizer's padding_side is left
        self.max_prompt_length = max_prompt_length
        self.max_length = max_length
        
        logger.info(f"Custom DPOTrainer initialization complete, templates: {self.prompt_template}, {self.response_template}")
    
    def concatenated_inputs(self, batch, padding_value):
        """
        Custom concatenated_inputs method for Qwen2.5 models, handling missing attention_mask
        """
        # Check if prompt_attention_mask is missing, create it if needed
        if "prompt_attention_mask" not in batch:
            logger.info("prompt_attention_mask not found, creating automatically")
            # Create attention_mask based on prompt_input_ids (all non-padding positions are 1)
            if "prompt_input_ids" in batch:
                batch["prompt_attention_mask"] = (batch["prompt_input_ids"] != padding_value).long()
            else:
                raise ValueError("batch is missing prompt_input_ids, cannot create attention_mask")
        
        # Check if chosen_attention_mask and rejected_attention_mask are missing
        if "chosen_attention_mask" not in batch and "chosen_input_ids" in batch:
            batch["chosen_attention_mask"] = (batch["chosen_input_ids"] != padding_value).long()
        
        if "rejected_attention_mask" not in batch and "rejected_input_ids" in batch:
            batch["rejected_attention_mask"] = (batch["rejected_input_ids"] != padding_value).long()
        
        # Call parent method for processing
        return super().concatenated_inputs(batch, padding_value)
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Special handling for directly loaded models
        """
        try:
            # Try using standard DPOTrainer loss calculation
            loss_output = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
            
            # Extract loss values and log them
            if isinstance(loss_output, tuple):
                loss, outputs = loss_output
                
                # Only log from main process to avoid duplicate outputs
                if self.args.local_rank == 0:
                    # Log detailed loss information
                    if hasattr(outputs, "policy_loss") and hasattr(outputs, "reference_loss"):
                        policy_loss = outputs.policy_loss.mean().item() if hasattr(outputs.policy_loss, "mean") else outputs.policy_loss
                        reference_loss = outputs.reference_loss.mean().item() if hasattr(outputs.reference_loss, "mean") else outputs.reference_loss
                        
                        # Save detailed component losses for analysis
                        if not hasattr(self, 'detailed_loss_log'):
                            self.detailed_loss_log = []
                            
                        self.detailed_loss_log.append({
                            'step': self.state.global_step,
                            'total_loss': loss.item() if hasattr(loss, 'item') else float(loss),
                            'policy_loss': float(policy_loss),
                            'reference_loss': float(reference_loss),
                            'learning_rate': self.optimizer.param_groups[0]['lr']
                        })
                        
                        # Periodically save detailed loss metrics to CSV
                        if self.state.global_step % 50 == 0 or self.state.global_step == self.args.max_steps:
                            import pandas as pd
                            import os
                            
                            loss_df = pd.DataFrame(self.detailed_loss_log)
                            os.makedirs(os.path.join(self.args.output_dir, 'logs'), exist_ok=True)
                            loss_df.to_csv(os.path.join(self.args.output_dir, 'logs', 'detailed_loss.csv'), index=False)
            
            if return_outputs:
                return loss, outputs if 'outputs' in locals() else None
            return loss
            
        except Exception as e:
            logger.warning(f"Standard loss calculation failed: {e}, trying custom implementation")
            
            # Here we could implement custom loss calculation logic
            # if needed for specific model architectures
            
            # Fallback to parent class implementation
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch=num_items_in_batch)
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Override training step to add more detailed logging
        """
        # Call parent's training_step
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        # Get current learning rate and step info
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # Only log from main process to avoid duplicate outputs
        if self.args.local_rank == 0:
            # Log to both console and tensorboard
            log_info = {
                'loss': loss.item(),
                'learning_rate': current_lr,
                'step': self.state.global_step,
                'progress': f"{self.state.global_step}/{self.args.max_steps}"
            }
            
            # Print training information
            print(f"[TRAINING INFO] Step: {self.state.global_step}/{self.args.max_steps}, Loss: {loss.item():.6f}, LR: {current_lr:.8f}")
            
            # Log to tensorboard
            for k, v in log_info.items():
                if isinstance(v, (int, float)):
                    self.log({k: v})
            
            # Save detailed metrics to CSV for later analysis
            if not hasattr(self, 'metrics_log'):
                self.metrics_log = []
            
            self.metrics_log.append(log_info)
            
            # Periodically save metrics to CSV
            if self.state.global_step % 50 == 0 or self.state.global_step == self.args.max_steps:
                import pandas as pd
                import os
                
                log_df = pd.DataFrame(self.metrics_log)
                os.makedirs(os.path.join(self.args.output_dir, 'logs'), exist_ok=True)
                log_df.to_csv(os.path.join(self.args.output_dir, 'logs', 'training_metrics.csv'), index=False)
            
            # If evaluation metrics exist, also log them
            if hasattr(self, 'eval_metrics') and self.eval_metrics:
                print(f"[EVAL METRICS] {self.eval_metrics}")
        
        return loss

def main():
    """
    Main function
    """
    try:
        logger.info("Starting execution of DPO training main function")
        
        # Parse command line arguments
        args = parse_args()
        logger.info(f"Command line arguments: {args}")
        
        # Set random seed
        torch.manual_seed(args.seed)
        logger.info(f"Setting random seed: {args.seed}")
        
        # Set output directory
        if args.output_dir is None:
            args.output_dir = os.path.join("outputs", f"combined_dpo_10k_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Output directory: {args.output_dir}")
        
        # Save training logs
        with open(os.path.join(args.output_dir, "training.log"), "w") as f:
            f.write(f"Starting Combined DPO training (8 GPU, 10K sequence length) {datetime.now().strftime('%a %b %d %I:%M:%S %p %Z %Y')}\n")
            f.write(f"Model path: {args.model_name}\n")
            f.write(f"Data path: {args.train_file_path}\n")
            f.write(f"Output directory: {args.output_dir}\n")
            f.write(f"Using 8 GPUs, ZeRO-3, Flash Attention 2, gradient checkpointing and LoRA\n")
        
        # Set log format
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
        )
        
        # Directly load model and tokenizer using our custom function
        model, tokenizer = load_model_and_tokenizer(args)
        tokenizer.padding_side = 'left' 
        
        # Create a reference model copy for DPO training
        logger.info("Loading reference model")
        ref_model, _ = load_model_and_tokenizer(args, is_reference_model=True)
        
        # Configure LoRA (if enabled)
        if args.use_lora:
            logger.info("Using LoRA for parameter-efficient fine-tuning")
            logger.info(f"Using target modules: {args.lora_target_modules}")
            
            # Parse target modules string to list
            target_modules_list = args.lora_target_modules.split(",") if args.lora_target_modules else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            
            # Create LoRA configuration
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules_list,
            )
            
            # If using LoRA, a separate reference model is not needed
            if args.use_lora:
                logger.info("Using LoRA training, not loading separate reference model")
                ref_model = None
        else:
            lora_config = None
        
        # Load and preprocess dataset
        datasets = preprocess_dataset(
            data_path=args.train_file_path, 
            prompt_field=args.prompt_field, 
            chosen_field=args.chosen_field, 
            rejected_field=args.rejected_field,
            max_prompt_length=args.max_prompt_length,
            max_length=args.max_length
        )
        train_dataset = datasets['train']
        eval_dataset = datasets['eval']
        
        # Create training parameters
        training_args = DPOConfig(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_steps=20,
            max_steps=50,
            bf16=args.bf16,
            fp16=args.fp16,
            logging_dir=os.path.join(args.output_dir, "logs"),
            logging_first_step=True,
            optim="adamw_torch",
            ddp_find_unused_parameters=False,
            ddp_backend="nccl",
            seed=args.seed,
            dataloader_num_workers=4,
            report_to="tensorboard",
            deepspeed=args.deepspeed,
            beta=args.beta,  # Set beta in DPOConfig
            padding_value=tokenizer.pad_token_id,  # Add padding_value parameter
            save_steps=args.save_steps,  # Add save_steps parameter
            save_total_limit=args.save_total_limit  # Limit the total number of saved checkpoints
        )
        
        # Start creating custom DPOTrainer
        logger.info("Starting to create custom DPOTrainer")
        
        # Create custom DPOTrainer
        dpo_trainer = CustomDPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            peft_config=lora_config,
            prompt_template="system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant. \nuser\n{prompt}",
            response_template="assistant\n{response}",
            max_prompt_length=args.max_prompt_length,
            max_length=args.max_length,
            # Use DPODataCollatorWithPadding, only pass pad_token_id parameter
            data_collator=DPODataCollatorWithPadding(pad_token_id=tokenizer.pad_token_id),
        )
        
        logger.info(f"Custom DPOTrainer creation complete, parameters: {training_args}")
        
        # Start training
        logger.info("Starting DPO training")
        try:
            dpo_trainer.train()
            logger.info("Training completed, preparing to save model")
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
        
        # Save final model
        logger.info(f"Saving final model to {args.output_dir}")
        dpo_trainer.save_model(args.output_dir)
        
        # If using LoRA and want to merge weights
        if args.use_lora and args.merge_weights:
            logger.info("Merging LoRA weights into base model")
            if hasattr(dpo_trainer.model, "merge_and_unload"):
                try:
                    # Save merged model in safetensors format
                    merged_model = dpo_trainer.model.merge_and_unload()
                    merged_output_dir = os.path.join(args.output_dir, "merged")
                    merged_model.save_pretrained(
                        merged_output_dir,
                        safe_serialization=True,  # Use safetensors format
                        save_function=safetensors.torch.save_file  # Explicitly use safetensors
                    )
                    logger.info(f"Successfully saved merged model to {merged_output_dir} in safetensors format")
                    
                    # Save training metrics summary
                    if hasattr(dpo_trainer, 'metrics_log') and dpo_trainer.metrics_log:
                        import pandas as pd
                        metrics_df = pd.DataFrame(dpo_trainer.metrics_log)
                        metrics_summary = {
                            'final_loss': float(metrics_df['loss'].iloc[-1]),
                            'min_loss': float(metrics_df['loss'].min()),
                            'max_loss': float(metrics_df['loss'].max()),
                            'mean_loss': float(metrics_df['loss'].mean()),
                            'final_lr': float(metrics_df['learning_rate'].iloc[-1]),
                            'training_steps': int(metrics_df['step'].max())
                        }
                        
                        # Save metrics summary as JSON
                        import json
                        with open(os.path.join(merged_output_dir, 'training_summary.json'), 'w') as f:
                            json.dump(metrics_summary, f, indent=2)
                        
                        logger.info(f"Training summary: Final loss: {metrics_summary['final_loss']:.6f}, Min loss: {metrics_summary['min_loss']:.6f}")
                    
                    # Also save tokenizer
                    tokenizer.save_pretrained(merged_output_dir)
                except Exception as e:
                    logger.error(f"Error merging LoRA weights: {e}")
                    logger.info("Still saving unmerged model")
            else:
                logger.warning("Model does not support merge_and_unload method, cannot merge LoRA weights")
        
        logger.info("DPO training completed")
        
    except Exception as e:
        logger.error(f"Error during main function execution: {e}")
        raise

if __name__ == "__main__":
    main()
