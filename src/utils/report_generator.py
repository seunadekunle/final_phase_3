"""Utility for generating training reports."""

import os
import json
from datetime import datetime
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

class ReportGenerator:
    """Generate training reports and visualizations."""
    
    def __init__(self, model, config, save_dir):
        """Initialize report generator.
        
        Args:
            model (nn.Module): The model being trained
            config (dict): Training configuration
            save_dir (Path): Directory to save reports and plots
        """
        self.model = model
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics tracking
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.learning_rates = []
        
    def update(self, epoch, train_loss, train_acc, val_loss, val_acc):
        """Update metrics after each epoch.
        
        Args:
            epoch (int): Current epoch number
            train_loss (float): Training loss
            train_acc (float): Training accuracy
            val_loss (float): Validation loss
            val_acc (float): Validation accuracy
        """
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)
        
        # Print progress
        print(f'\nEpoch: {epoch+1}')
        print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.2f}%')
        print('='*70)
    
    def finalize(self, test_loss, test_acc, training_time):
        """Generate final report and plots.
        
        Args:
            test_loss (float): Final test loss
            test_acc (float): Final test accuracy
            training_time (float): Total training time in seconds
        """
        # Save metrics
        metrics = {
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'training_time': training_time
        }
        
        # Plot training curves
        self._plot_training_curves()
        
        # Generate report
        self._generate_report(metrics)
    
    def _plot_training_curves(self):
        """Plot training and validation curves."""
        plt.figure(figsize=(12, 4))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train')
        plt.plot(self.val_losses, label='Validation')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='Train')
        plt.plot(self.val_accs, label='Validation')
        plt.title('Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png')
        plt.close()
    
    def _generate_report(self, metrics):
        """Generate training report.
        
        Args:
            metrics (dict): Training metrics
        """
        report = [
            "# Training Report\n",
            f"## Configuration\n```yaml\n{yaml.dump(self.config, default_flow_style=False)}```\n",
            "## Results\n",
            f"- Test Loss: {metrics['test_loss']:.4f}",
            f"- Test Accuracy: {metrics['test_acc']:.2f}%",
            f"- Training Time: {metrics['training_time']/3600:.2f} hours",
            f"- Best Validation Accuracy: {max(self.val_accs):.2f}%",
            "\n## Training Curves\n",
            "![Training Curves](training_curves.png)\n"
        ]
        
        with open(self.save_dir / 'report.md', 'w') as f:
            f.write('\n'.join(report))