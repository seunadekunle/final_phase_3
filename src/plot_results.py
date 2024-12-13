"""
Plot training metrics for all variants.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def load_metrics(variant_dir):
    """Load training metrics from a variant directory."""
    metrics_path = Path(variant_dir) / 'logs' / 'training_metrics.json'
    if not metrics_path.exists():
        return None
    
    with open(metrics_path, 'r') as f:
        return json.load(f)

def plot_metrics(variants_dir, save_dir=None):
    """Plot training metrics for all variants."""
    # Set the style
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Find all variants
    variants_dir = Path(variants_dir)
    variant_dirs = [d for d in variants_dir.iterdir() if d.is_dir() and d.name.startswith('resnet_variant')]
    variant_dirs.sort()
    
    # Load metrics for all variants
    variant_metrics = {}
    for var_dir in variant_dirs:
        metrics = load_metrics(var_dir)
        if metrics:
            variant_metrics[var_dir.name] = metrics
    
    if not variant_metrics:
        print("No metrics found!")
        return
    
    # Create plots
    metrics_to_plot = [
        ('loss', ('train_loss', 'val_loss'), 'Loss'),
        ('accuracy', ('train_acc', 'val_acc'), 'Accuracy (%)'),
        ('learning_rate', ('lr',), 'Learning Rate')
    ]
    
    for plot_name, metrics_keys, ylabel in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        
        for variant_name, metrics in variant_metrics.items():
            epochs = range(1, len(metrics['train_loss']) + 1)
            
            for key in metrics_keys:
                if key == 'lr':
                    label = f"{variant_name} LR"
                    linestyle = '--'
                else:
                    label = f"{variant_name} {'Train' if 'train' in key else 'Val'}"
                    linestyle = '-'
                
                plt.plot(epochs, metrics[key], label=label, linestyle=linestyle)
        
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.title(f'Training {plot_name.title()} vs. Epoch')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(save_dir / f'{plot_name}_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Plots saved in {save_dir}")

def main():
    parser = argparse.ArgumentParser(description='Plot training metrics for all variants')
    parser.add_argument('--variants-dir', type=str, default='experiments/multi_comparison',
                       help='Directory containing variant subdirectories')
    parser.add_argument('--save-dir', type=str, default='plots',
                       help='Directory to save plots')
    args = parser.parse_args()
    
    plot_metrics(args.variants_dir, args.save_dir)

if __name__ == '__main__':
    main() 