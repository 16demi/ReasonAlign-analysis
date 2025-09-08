#!/usr/bin/env python3
"""
Training Dynamics Visualization Script

This script generates publication-quality plots showing probability changes over 
training epochs for forward and reverse reasoning tasks. Creates line plots,
heatmaps, and multi-subplot visualizations.

Based on visualization techniques from various open-source projects.
SFT methods adapted from: https://github.com/simplescaling/s1

Features:
- Multi-subplot probability evolution plots
- Example-by-example heatmap visualizations
- High-resolution output for publications
- Seaborn and matplotlib styling
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def plot_probabilities(df, output_dir, subset_name):
    """Generate plots showing probability changes over epochs."""
    # Group by epoch and calculate mean for each metric
    epoch_metrics = df.groupby('epoch').agg({
        'text_avg_log_prob': 'mean',
        'worse_avg_log_prob': 'mean',
        'text_peak_log_prob': 'mean',
        'worse_peak_log_prob': 'mean',
        'avg_diff': 'mean',
        'peak_diff': 'mean'
    }).reset_index()
    
    # Set the style
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 20))
    
    # 1. Average log probabilities plot
    plt.subplot(3, 1, 1)
    plt.plot(epoch_metrics['epoch'], epoch_metrics['text_avg_log_prob'], 'b-', marker='o', label='Text (Forward Reasoning)')
    plt.plot(epoch_metrics['epoch'], epoch_metrics['worse_avg_log_prob'], 'r-', marker='x', label='Worse-Text (Backward Reasoning)')
    plt.xlabel('Epoch')
    plt.ylabel('Average Log Probability per Token')
    plt.title(f'Average Log Probabilities Over Training - {subset_name}')
    plt.legend()
    plt.grid(True)
    
    # 2. Peak log probabilities plot
    plt.subplot(3, 1, 2)
    plt.plot(epoch_metrics['epoch'], epoch_metrics['text_peak_log_prob'], 'b-', marker='o', label='Text (Forward Reasoning)')
    plt.plot(epoch_metrics['epoch'], epoch_metrics['worse_peak_log_prob'], 'r-', marker='x', label='Worse-Text (Backward Reasoning)')
    plt.xlabel('Epoch')
    plt.ylabel('Peak Log Probability')
    plt.title(f'Peak Log Probabilities Over Training - {subset_name}')
    plt.legend()
    plt.grid(True)
    
    # 3. Probability difference plot
    plt.subplot(3, 1, 3)
    plt.plot(epoch_metrics['epoch'], epoch_metrics['avg_diff'], 'g-', marker='s', label='Average Probability Difference')
    plt.plot(epoch_metrics['epoch'], epoch_metrics['peak_diff'], 'm-', marker='d', label='Peak Probability Difference')
    plt.xlabel('Epoch')
    plt.ylabel('Log Probability Difference (Text - Worse-Text)')
    plt.title(f'Probability Difference Over Training - {subset_name}')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the figure
    plot_path = os.path.join(output_dir, f"{subset_name}_probability_plots.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Plots saved to {plot_path}")

def create_heatmap(df, output_dir, subset_name):
    """Create a heatmap showing example-by-example changes over epochs."""
    # Pivot the dataframe for heatmap
    avg_diff_pivot = df.pivot(index='example_id', columns='epoch', values='avg_diff')
    
    plt.figure(figsize=(12, max(8, len(avg_diff_pivot) / 4)))
    sns.heatmap(
        avg_diff_pivot, 
        cmap='RdBu_r', 
        center=0,
        annot=False,
        cbar_kws={'label': 'Log Probability Difference (Text - Worse-Text)'}
    )
    plt.title(f'Example-by-Example Probability Difference Over Epochs - {subset_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Example ID')
    
    # Save the figure
    heatmap_path = os.path.join(output_dir, f"{subset_name}_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    
    print(f"Heatmap saved to {heatmap_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize training dynamics across epochs")
    parser.add_argument("--results_dir", type=str, required=True, 
                        help="Directory containing CSV results files")
    
    args = parser.parse_args()
    
    # List of subset names
    subset_names = ["first50", "last50", "all"]
    
    for subset_name in subset_names:
        csv_path = os.path.join(args.results_dir, f"{subset_name}.csv")
        
        if os.path.exists(csv_path):
            print(f"Processing {csv_path}...")
            df = pd.read_csv(csv_path)
            
            # Generate plots
            plot_probabilities(df, args.results_dir, subset_name)
            create_heatmap(df, args.results_dir, subset_name)
        else:
            print(f"Warning: {csv_path} does not exist")
    
    print("Visualization complete!")

if __name__ == "__main__":
    main()
