"""Utility for generating training reports."""

import os
import json
from datetime import datetime
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class ReportGenerator:
    def __init__(self, model, config, save_dir='reports'):
        self.model = model
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = {
            'epoch_metrics': [],
            'train_losses': [],
            'train_accs': [],
            'val_losses': [],
            'val_accs': [],
            'learning_rates': []
        }
        
    def update_metrics(self, epoch_metrics):
        """Update metrics after each epoch."""
        self.metrics['epoch_metrics'].append(epoch_metrics)
        self.metrics['train_losses'].append(epoch_metrics['train_loss'])
        self.metrics['train_accs'].append(epoch_metrics['train_acc'])
        self.metrics['val_losses'].append(epoch_metrics['val_loss'])
        self.metrics['val_accs'].append(epoch_metrics['val_acc'])
        self.metrics['learning_rates'].append(epoch_metrics['learning_rate'])
    
    def plot_training_curves(self):
        """Generate training curves."""
        # Set seaborn style
        sns.set_theme(style="darkgrid")
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot accuracy
        epochs = range(1, len(self.metrics['train_accs']) + 1)
        sns.lineplot(x=epochs, y=self.metrics['train_accs'], label='Train Acc', ax=ax1)
        sns.lineplot(x=epochs, y=self.metrics['val_accs'], label='Val Acc', ax=ax1)
        ax1.set_title('Training and Validation Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.legend()
        
        # Plot loss
        sns.lineplot(x=epochs, y=self.metrics['train_losses'], label='Train Loss', ax=ax2)
        sns.lineplot(x=epochs, y=self.metrics['val_losses'], label='Val Loss', ax=ax2)
        ax2.set_title('Training and Validation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png')
        plt.close()
    
    def generate_model_summary(self):
        """Generate model architecture summary."""
        summary = []
        total_params = 0
        trainable_params = 0
        
        for name, param in self.model.named_parameters():
            param_count = param.numel()
            total_params += param_count
            if param.requires_grad:
                trainable_params += param_count
            layer_info = f"{name}: {list(param.shape)}, Parameters: {param_count:,}"
            summary.append(layer_info)
        
        return {
            'layer_summary': summary,
            'total_params': total_params,
            'trainable_params': trainable_params
        }
    
    def generate_baseline_report(self, final_metrics):
        """Generate the baseline report markdown file."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_path = self.save_dir / f'baseline_report_{timestamp}.md'
        
        model_summary = self.generate_model_summary()
        best_val_acc = max(self.metrics['val_accs'])
        best_epoch = self.metrics['val_accs'].index(best_val_acc) + 1
        
        with open(report_path, 'w') as f:
            # Title
            f.write("# ResNet-18 CIFAR-10 Baseline Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Model Architecture
            f.write("## Model Architecture\n\n")
            f.write("### Overview\n")
            f.write("- Base Architecture: ResNet-18\n")
            f.write("- Input Size: 32x32x3 (CIFAR-10 images)\n")
            f.write(f"- Total Parameters: {model_summary['total_params']:,}\n")
            f.write(f"- Trainable Parameters: {model_summary['trainable_params']:,}\n\n")
            
            # Hyperparameters
            f.write("## Training Hyperparameters\n\n")
            f.write("```python\n")
            for key, value in vars(self.config).items():
                if not key.startswith('__'):
                    f.write(f"{key} = {value}\n")
            f.write("```\n\n")
            
            # Performance Metrics
            f.write("## Performance Metrics\n\n")
            f.write("### Final Results\n")
            f.write(f"- Test Accuracy: {final_metrics['test_acc']:.2f}%\n")
            f.write(f"- Test Loss: {final_metrics['test_loss']:.4f}\n")
            f.write(f"- Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})\n")
            f.write(f"- Training Time: {final_metrics['training_time']:.2f} hours\n\n")
            
            # Training Curves
            f.write("### Training Curves\n")
            f.write("![Training Curves](training_curves.png)\n\n")
            
            # Comparison to Original ResNet-18
            f.write("## Comparison to Original ResNet-18\n\n")
            f.write("### Architectural Differences\n")
            f.write("- Base implementation follows the original ResNet-18 architecture\n")
            f.write("- Adapted for CIFAR-10 with appropriate input size handling\n")
            f.write("- Uses batch normalization and ReLU activation as in original paper\n\n")
            
            # Notes and Observations
            f.write("## Notes and Observations\n\n")
            f.write("### Training Process\n")
            f.write(f"- Model trained for {len(self.metrics['epoch_metrics'])} epochs\n")
            f.write(f"- Used {self.config.device} for training\n")
            f.write("- Implemented early stopping with patience\n\n")
            
            f.write("### Challenges and Solutions\n")
            f.write("1. Initial Implementation:\n")
            f.write("   - Challenge: Adapting ResNet-18 for CIFAR-10\n")
            f.write("   - Solution: Modified initial convolution and pooling layers\n\n")
            
            f.write("2. Training Stability:\n")
            f.write("   - Challenge: Learning rate tuning\n")
            f.write("   - Solution: Implemented warmup and cosine annealing\n\n")
            
            f.write("3. Overfitting:\n")
            f.write("   - Challenge: Gap between train and validation accuracy\n")
            f.write("   - Solution: Added data augmentation and regularization\n\n")
            
        return report_path 