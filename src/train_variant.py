"""Train a specific ResNet18 variant with enhanced logging and progress tracking."""

import os
import sys
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import logging
import argparse
import json
import csv

from torch.optim.swa_utils import AveragedModel, SWALR
from data.dataset import CIFAR10Data
from models.resnet import ResNet18
from utils.early_stopping import EarlyStopping
from utils.report_generator import ReportGenerator

def setup_logging(log_dir, name):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(name)

def load_config(config_path):
    """Load and validate configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert numeric values to proper types
    config['training']['batch_size'] = int(config['training']['batch_size'])
    config['training']['seed'] = int(config['training']['seed'])
    config['training']['num_workers'] = int(config['training']['num_workers'])
    config['training']['num_epochs'] = int(config['training']['num_epochs'])
    
    config['optimization']['learning_rate'] = float(config['optimization']['learning_rate'])
    config['optimization']['momentum'] = float(config['optimization']['momentum'])
    config['optimization']['weight_decay'] = float(config['optimization']['weight_decay'])
    config['optimization']['nesterov'] = bool(config['optimization']['nesterov'])
    
    config['lr_schedule']['warmup_epochs'] = int(config['lr_schedule']['warmup_epochs'])
    config['lr_schedule']['min_lr'] = float(config['lr_schedule']['min_lr'])
    config['lr_schedule']['warmup_start_lr'] = float(config['lr_schedule']['warmup_start_lr'])
    if 'cosine_decay_end' in config['lr_schedule']:
        config['lr_schedule']['cosine_decay_end'] = float(config['lr_schedule']['cosine_decay_end'])
    if 'milestones' in config['lr_schedule']:
        config['lr_schedule']['milestones'] = [int(x) for x in config['lr_schedule']['milestones']]
    if 'gamma' in config['lr_schedule']:
        config['lr_schedule']['gamma'] = float(config['lr_schedule']['gamma'])
    
    config['advanced_training']['stochastic_depth_prob'] = float(config['advanced_training']['stochastic_depth_prob'])
    config['advanced_training']['label_smoothing'] = float(config['advanced_training']['label_smoothing'])
    config['advanced_training']['mixup_alpha'] = float(config['advanced_training']['mixup_alpha'])
    config['advanced_training']['cutmix_alpha'] = float(config['advanced_training']['cutmix_alpha'])
    if 'mixup_prob' in config['advanced_training']:
        config['advanced_training']['mixup_prob'] = float(config['advanced_training']['mixup_prob'])
    if 'cutmix_prob' in config['advanced_training']:
        config['advanced_training']['cutmix_prob'] = float(config['advanced_training']['cutmix_prob'])
    if 'alternate_augment' in config['advanced_training']:
        config['advanced_training']['alternate_augment'] = bool(config['advanced_training']['alternate_augment'])
    
    config['data_augmentation']['randaugment'] = bool(config['data_augmentation']['randaugment'])
    config['data_augmentation']['randaugment_n'] = int(config['data_augmentation']['randaugment_n'])
    config['data_augmentation']['randaugment_m'] = int(config['data_augmentation']['randaugment_m'])
    config['data_augmentation']['random_erasing_prob'] = float(config['data_augmentation']['random_erasing_prob'])
    config['data_augmentation']['trivialaugment'] = bool(config['data_augmentation']['trivialaugment'])
    
    config['regularization']['drop_path_rate'] = float(config['regularization']['drop_path_rate'])
    if 'dropout_rate' in config['regularization']:
        config['regularization']['dropout_rate'] = float(config['regularization']['dropout_rate'])
    
    config['early_stopping']['patience'] = int(config['early_stopping']['patience'])
    config['early_stopping']['min_delta'] = float(config['early_stopping']['min_delta'])
    
    config['model']['num_classes'] = int(config['model']['num_classes'])
    config['model']['use_se_blocks'] = bool(config['model']['use_se_blocks'])
    if 'use_fc_dropout' in config['model']:
        config['model']['use_fc_dropout'] = bool(config['model']['use_fc_dropout'])
    
    # Handle SWA config if present
    if 'swa' in config:
        config['swa']['enabled'] = bool(config['swa']['enabled'])
        if config['swa']['enabled']:
            config['swa']['start_epoch'] = int(config['swa']['start_epoch'])
            config['swa']['lr'] = float(config['swa']['lr'])
            config['swa']['anneal_epochs'] = int(config['swa']['anneal_epochs'])
    
    return config

class MetricTracker:
    """Track and log training metrics."""
    def __init__(self):
        self.metrics = {}
        self.reset()
    
    def reset(self):
        self.metrics = {
            'loss': 0.0,
            'acc': 0.0,
            'lr': 0.0,
            'batch_count': 0
        }
    
    def update(self, loss, acc, lr):
        self.metrics['loss'] += loss
        self.metrics['acc'] += acc
        self.metrics['lr'] = lr
        self.metrics['batch_count'] += 1
    
    def get_averages(self):
        count = self.metrics['batch_count']
        if count == 0:
            return self.metrics
        
        return {
            'loss': self.metrics['loss'] / count,
            'acc': self.metrics['acc'] / count,
            'lr': self.metrics['lr'],
            'batch_count': count
        }

def get_lr_scheduler(optimizer, config):
    """Get learning rate scheduler based on config.
    
    Args:
        optimizer: PyTorch optimizer
        config: Configuration dictionary
        
    Returns:
        PyTorch learning rate scheduler
    """
    if config['lr_schedule']['type'] == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs'] - config['lr_schedule']['warmup_epochs'],
            eta_min=config['lr_schedule']['min_lr']
        )
    elif config['lr_schedule']['type'] == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config['lr_schedule']['milestones'],
            gamma=config['lr_schedule']['gamma']
        )
    else:
        raise ValueError(f"Unknown scheduler type: {config['lr_schedule']['type']}")

def apply_mixup(x, y, alpha, device):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def apply_cutmix(x, y, alpha, device):
    """Apply CutMix augmentation.
    
    Args:
        x (torch.Tensor): Input images
        y (torch.Tensor): Labels
        alpha (float): Beta distribution parameter
        device (torch.device): Device to use
        
    Returns:
        tuple: Mixed images, target A, target B, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    # Get random box dimensions
    W = x.size()[2]
    H = x.size()[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Get random box position
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # Apply cutmix
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def train_epoch(epoch, model, train_loader, criterion, optimizer, scheduler, device, tracker, logger, args, config):
    """Train for one epoch with progress bar and metrics."""
    model.train()
    tracker.reset()
    
    desc = f'Epoch {epoch}/{config["training"]["num_epochs"]}'
    progress_bar = tqdm(train_loader, desc=desc, leave=True) if args.progress else train_loader
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        acc = correct / len(target)
        
        # Update metrics
        tracker.update(loss.item(), acc, optimizer.param_groups[0]['lr'])
        
        # Update progress bar
        if args.progress and batch_idx % args.log_interval == 0:
            metrics = tracker.get_averages()
            progress_bar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'acc': f"{metrics['acc']:.4f}",
                'lr': f"{metrics['lr']:.6f}"
            })
    
    # Log epoch metrics
    metrics = tracker.get_averages()
    logger.info(f"Epoch {epoch} - Train loss: {metrics['loss']:.4f}, acc: {metrics['acc']:.4f}, lr: {metrics['lr']:.6f}")
    
    return metrics

def validate(model, val_loader, criterion, device, logger, epoch, config, desc='Val'):
    """Validate the model on the validation set.

    Args:
        model: The model to validate
        val_loader: The validation data loader
        criterion: The loss function
        device: The device to use
        logger: The logger to use
        epoch: The current epoch
        config: The configuration dictionary
        desc: Description for the progress bar

    Returns:
        tuple: The validation loss and accuracy
    """
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f'{desc} Epoch {epoch}/{config["training"]["num_epochs"]}', leave=True)
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            acc = 100. * correct / total
            progress_bar.set_postfix({'loss': f"{val_loss/len(val_loader):.4f}", 'acc': f"{acc:.2f}%"})
    
    val_loss /= len(val_loader)
    acc = 100. * correct / total
    
    logger.info(f"{desc} - Average loss: {val_loss:.4f}, Accuracy: {acc:.2f}%")
    return val_loss, acc

def save_metrics_csv(metrics, variant_dir):
    """Save training metrics to a CSV file.
    
    Args:
        metrics (dict): Dictionary containing training metrics
        variant_dir (Path): Directory to save the CSV file
    """
    csv_path = variant_dir / 'logs' / 'training_metrics.csv'
    
    # Prepare data as rows
    rows = []
    for i in range(len(metrics['train_loss'])):
        rows.append({
            'epoch': i + 1,
            'train_loss': metrics['train_loss'][i],
            'train_acc': metrics['train_acc'][i],
            'val_loss': metrics['val_loss'][i],
            'val_acc': metrics['val_acc'][i],
            'learning_rate': metrics['lr'][i]
        })
    
    # Write to CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'learning_rate'])
        writer.writeheader()
        writer.writerows(rows)

def train_variant(config_path, args):
    """Train a specific variant with enhanced logging and progress tracking."""
    # Load configuration
    config = load_config(config_path)
    variant_dir = Path(config_path).parent.parent
    
    # Setup logging
    logger = setup_logging(variant_dir / 'logs', variant_dir.name)
    logger.info(f"Starting training for {variant_dir.name}")
    logger.info(f"Configuration: {config_path}")
    
    # Initialize metrics tracking
    metrics = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    # Set random seed
    torch.manual_seed(config['training']['seed'])
    
    # Setup device
    device = torch.device(config['training']['device'])
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    data = CIFAR10Data(
        data_dir=config['paths']['data_dir'],
        train_batch_size=config['training']['batch_size'],
        eval_batch_size=100,
        num_workers=config['training']['num_workers'],
        autoaugment=config['data_augmentation']['randaugment'],
        cutout_size=16 if config['data_augmentation']['random_erasing_prob'] > 0 else 0,
        cutout_prob=config['data_augmentation']['random_erasing_prob']
    )
    
    # Create model
    model = ResNet18(
        num_classes=config['model']['num_classes'],
        use_se_blocks=config['model']['use_se_blocks']
    ).to(device)
    
    # Print model architecture
    logger.info("\nResNet-18 Architecture:")
    logger.info("-" * 50)
    logger.info("Initial Conv Layer: 3 -> 64 channels")
    logger.info("Layer1: 64 -> 64 channels, 2 blocks with SE")
    logger.info("Layer2: 64 -> 128 channels, 2 blocks with SE")
    logger.info("Layer3: 128 -> 256 channels, 2 blocks with SE")
    logger.info("Layer4: 256 -> 512 channels, 2 blocks with SE")
    logger.info("Global Average Pooling")
    logger.info("Fully Connected: 512 -> 10 classes")
    logger.info("-" * 50)
    
    # Setup training components
    criterion = nn.CrossEntropyLoss(
        label_smoothing=config['advanced_training']['label_smoothing']
    )
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['optimization']['learning_rate'],
        momentum=config['optimization']['momentum'],
        weight_decay=config['optimization']['weight_decay'],
        nesterov=config['optimization']['nesterov']
    )
    scheduler = get_lr_scheduler(optimizer, config)
    early_stopping = EarlyStopping(
        patience=config['early_stopping']['patience'],
        min_delta=config['early_stopping']['min_delta']
    )
    
    # Initialize metric tracker
    tracker = MetricTracker()
    
    # Training loop
    best_acc = 0
    start_time = time.time()
    logger.info("Starting training loop")
    
    for epoch in range(1, config['training']['num_epochs'] + 1):
        # Train epoch
        train_metrics = train_epoch(epoch, model, data.train_dataloader(), criterion, 
                                  optimizer, scheduler, device, tracker, logger, args, config)
        
        # Validate
        val_loss, val_acc = validate(model, data.val_dataloader(), criterion, device, logger, epoch, config)
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # Save metrics
        metrics['train_loss'].append(train_metrics['loss'])
        metrics['train_acc'].append(train_metrics['acc'])
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)
        metrics['lr'].append(current_lr)
        
        # Save metrics to file after each epoch
        metrics_path = variant_dir / 'logs' / 'training_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Also save as CSV for easy data analysis
        save_metrics_csv(metrics, variant_dir)
        
        # Save checkpoint if improved
        if val_acc > best_acc and args.save_freq > 0 and epoch % args.save_freq == 0:
            best_acc = val_acc
            checkpoint_path = variant_dir / 'checkpoints' / f'best_model_epoch_{epoch}.pt'
            os.makedirs(checkpoint_path.parent, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint with accuracy {val_acc:.2f}%")
        
        # Early stopping check
        if early_stopping(val_loss):
            logger.info("Early stopping triggered")
            break
    
    # Final evaluation
    test_loss, test_acc = validate(model, data.test_dataloader(), criterion, device, logger, epoch, config, "Test")
    
    # Save final results
    training_time = time.time() - start_time
    results = {
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'best_val_accuracy': best_acc,
        'training_time': training_time,
        'epochs_completed': epoch,
        'final_train_loss': metrics['train_loss'][-1],
        'final_train_acc': metrics['train_acc'][-1],
        'final_val_loss': metrics['val_loss'][-1],
        'final_val_acc': metrics['val_acc'][-1]
    }
    
    # Save final results
    results_path = variant_dir / 'logs' / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info("\nTraining completed!")
    logger.info(f"Total training time: {training_time/3600:.2f} hours")
    logger.info(f"Final test accuracy: {test_acc:.2f}%")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Train ResNet18 variant')
    parser.add_argument('config_path', type=str, help='Path to variant config file')
    parser.add_argument('--progress', action='store_true', help='Show progress bars')
    parser.add_argument('--log-interval', type=int, default=10, help='Logging interval')
    parser.add_argument('--save-freq', type=int, default=1, help='Checkpoint save frequency')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    args = parser.parse_args()
    
    results = train_variant(args.config_path, args)
    
    # Save results
    variant_dir = Path(args.config_path).parent.parent
    results_file = variant_dir / 'logs' / 'results.txt'
    with open(results_file, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")

if __name__ == '__main__':
    main() 