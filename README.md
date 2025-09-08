# Forward and Reverse Reasoning Training Dynamics

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

This repository contains the implementation and analysis code for studying training dynamics in Direct Preference Optimization (DPO) with forward and reverse reasoning tasks. The research explores how language models learn to distinguish between correct forward reasoning and incorrect reverse reasoning during preference optimization.

## 🔄 Path Configuration Guide

**IMPORTANT**: Before using this code, you must update the paths in the configuration files to match your system setup.

### Required Path Updates

1. **Model Paths** - Update in all scripts:
   ```bash
   # Example: Replace placeholder paths
   ./models/your_base_model → /path/to/your/actual/model
   ```

2. **Dataset Paths** - Update in training scripts:
   ```bash
   # Example: Replace placeholder paths  
   ./data/your_dataset.parquet → /path/to/your/actual/dataset.parquet
   ```

3. **Output Paths** - Update in shell scripts:
   ```bash
   # Example: Replace placeholder paths
   ./outputs/ → /your/preferred/output/directory/
   ```

### Quick Path Replacement Commands

```bash
# Replace model paths in all Python files
find . -name "*.py" -exec sed -i 's|./models/your_base_model|/path/to/your/actual/model|g' {} +

# Replace dataset paths in all shell scripts  
find . -name "*.sh" -exec sed -i 's|./data/your_dataset.parquet|/path/to/your/actual/dataset.parquet|g' {} +

# Replace output paths
find . -name "*.sh" -exec sed -i 's|./outputs/|/your/preferred/output/directory/|g' {} +
```

## 📋 Code Attribution

This repository contains implementations based on various open-source projects:

- **SFT and Evaluation Methods**: Adapted from [simplescaling/s1](https://github.com/simplescaling/s1)
- **DPO Training Infrastructure**: Based on community DPO implementations
- **Analysis and Visualization**: Custom implementations with techniques from various sources

We respect all original copyrights and licenses. This code is provided for research purposes under the MIT License.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU(s)
- 16GB+ GPU memory recommended

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd paper

# Install dependencies
pip install -r requirements.txt
```

### Dataset Format

Your dataset should be a Parquet file with these columns:
- `question`: The reasoning problem/question
- `text`: The preferred response (forward reasoning)
- `worse-text`: The rejected response (reverse reasoning)

### Basic Usage

1. **Update paths** in the configuration files (see Path Configuration Guide above)

2. **Run DPO Training**:
```bash
# Update paths in the script first!
bash dpo/7b-train_multiple_base_models.sh
```

3. **Analyze Results**:
```bash
# Update model and dataset paths first!
python analysis/analyze_dpo_model.py \
    --model_path /path/to/your/trained/model \
    --dataset_path /path/to/your/dataset.parquet \
    --output_dir ./results \
    --gpu_id 0
```

4. **Generate Visualizations**:
```bash
python visualization/plot_with_wider_middle.py
```

## 📁 Repository Structure

```
├── README.md                           # This file
├── LICENSE                            # MIT License  
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
├── r1k.parquet                       # Sample dataset (~1k examples)
├── data-100samples.parquet           # Small sample for testing
├── configs/
│   └── ds_config_zero3_10k.json     # DeepSpeed ZeRO-3 configuration
├── dpo/                              # DPO training scripts
│   ├── combined_dpo.py               # Main DPO training implementation
│   └── 7b-train_multiple_base_models.sh  # Multi-model training script
├── analysis/                         # Training analysis tools
│   ├── analyze_dpo_model.py          # Model output analysis
│   ├── analyze_training_dynamics.py  # Training dynamics analysis  
│   └── *.sh                         # Analysis shell scripts
└── visualization/                    # Plotting and visualization
    ├── plot_with_wider_middle.py     # Custom visualization with non-linear scaling
    └── visualize_training_dynamics.py # Training dynamics plots
```

## 🔧 Technical Specifications

### Hardware Requirements
- **Minimum**: 1x GPU with 16GB+ VRAM
- **Recommended**: 4x GPUs with 24GB+ VRAM each
- **Memory**: 32GB+ system RAM
- **Storage**: 100GB+ free space for models and outputs

### Software Dependencies
- **Python**: 3.8 or higher
- **PyTorch**: 2.0+ with CUDA support  
- **Transformers**: Latest version with flash attention support
- **DeepSpeed**: For distributed training optimization

### Training Configuration
- **Optimizer**: AdamW with cosine learning rate scheduling
- **Precision**: Mixed precision training (bfloat16)
- **LoRA Parameters**: r=256, alpha=512, dropout=0.05
- **Batch Size**: 1 per device with gradient accumulation
- **Max Length**: 20,000 tokens for long-form reasoning

## 🎯 Experiment Pipeline

### 1. Data Preparation
```bash
# Your dataset should contain forward/reverse reasoning pairs
# Format: question, text (preferred), worse-text (rejected)
```

### 2. DPO Training  
```bash
# Multi-GPU distributed training with DeepSpeed
torchrun --nproc_per_node=4 dpo/combined_dpo.py [args]
```

### 3. Analysis
```bash
# Compute log probabilities and preference metrics
python analysis/analyze_dpo_model.py [args]
```

### 4. Visualization
```bash
# Generate publication-quality plots
python visualization/plot_with_wider_middle.py
```

## 📊 Key Components

- **DPO Training**: Implementation of Direct Preference Optimization with LoRA fine-tuning
- **Training Dynamics Analysis**: Tools to analyze how models learn preferences over time  
- **Log Probability Calculation**: Compute model confidence in forward vs. reverse reasoning
- **Custom Visualization**: Non-linear scaling for probability range visualization
- **Multi-GPU Support**: Distributed training with DeepSpeed ZeRO-3

## 🎨 Visualization Features

- **Non-linear axis scaling** emphasizing critical probability ranges
- **LaTeX-compatible** output for publications
- **Multiple export formats**: PNG, PDF, SVG
- **Customizable styling** and color schemes
- **Multi-series support** for comparison plots

## 📖 Usage Examples

### Training a Model
```bash
# Update paths in the script first!
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 dpo/combined_dpo.py \
    --model_name /path/to/base/model \
    --train_file_path /path/to/dataset.parquet \
    --output_dir ./outputs/my_dpo_model \
    --max_steps 200 \
    --learning_rate 5e-7 \
    --bf16 \
    --use_flash_attention
```

### Analyzing Results
```bash
python analysis/analyze_dpo_model.py \
    --model_path ./outputs/my_dpo_model \
    --dataset_path /path/to/test_dataset.parquet \
    --output_dir ./analysis_results \
    --gpu_id 0
```

## 🏗️ Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch  
3. Update paths in your changes to be generic
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 References and Attribution

- SFT and evaluation methods adapted from: https://github.com/simplescaling/s1
- DPO implementation based on community best practices
- Custom analysis and visualization implementations

### Academic Citations

If you use this codebase or methodology in your research, please consider citing the following papers:

```bibtex
@article{ren2024learning,
  title={Learning dynamics of llm finetuning},
  author={Ren, Yi and Sutherland, Danica J},
  journal={arXiv preprint arXiv:2407.10490},
  year={2024}
}

@article{muennighoff2025s1,
  title={s1: Simple test-time scaling},
  author={Muennighoff, Niklas and Yang, Zitong and Shi, Weijia and Li, Xiang Lisa and Fei-Fei, Li and Hajishirzi, Hannaneh and Zettlemoyer, Luke and Liang, Percy and Cand{\`e}s, Emmanuel and Hashimoto, Tatsunori},
  journal={arXiv preprint arXiv:2501.19393},
  year={2025}
}

@article{rafailov2023direct,
  title={Direct preference optimization: Your language model is secretly a reward model},
  author={Rafailov, Rafael and Sharma, Archit and Mitchell, Eric and Manning, Christopher D and Ermon, Stefano and Finn, Chelsea},
  journal={Advances in neural information processing systems},
  volume={36},
  pages={53728--53741},
  year={2023}
}
```


