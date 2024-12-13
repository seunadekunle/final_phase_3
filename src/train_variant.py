"""
Train a specific ResNet18 variant.
"""

import os
import sys
import time
import math
import yaml
import torch
import numpy as np
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
import argparse
from tqdm import tqdm

from data.dataset import CIFAR10Data
from models.resnet import ResNet18
from utils.report_generator import ReportGenerator
from utils.early_stopping import EarlyStopping

def load_config(config_path):
    """Load and validate configuration from YAML file.
    
    Args:
        config_path (str): Path to YAML config file
        
    Returns:
        dict: Configuration with proper type conversion
    """
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

def train_variant(config_path):
    # Load configuration with type conversion
    config = load_config(config_path)
    variant_dir = Path(config_path).parent.parent
    
    # Set random seed
    torch.manual_seed(config['training']['seed'])
    
    # Setup device
    device = torch.device(config['training']['device'])
    
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
    )
    if config['model'].get('use_fc_dropout', False):
        model.fc = nn.Sequential(
            nn.Dropout(config['regularization']['dropout_rate']),
            model.fc
        )
    model = model.to(device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss(
        label_smoothing=config['advanced_training']['label_smoothing']
    )
    
    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['optimization']['learning_rate'],
        momentum=config['optimization']['momentum'],
        weight_decay=config['optimization']['weight_decay'],
        nesterov=config['optimization']['nesterov']
    )
    
    # Learning rate scheduler
    scheduler = get_lr_scheduler(optimizer, config)
    
    # SWA setup if enabled
    if config.get('swa', {}).get('enabled', False):
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(
            optimizer,
            swa_lr=config['swa']['lr'],
            anneal_epochs=config['swa']['anneal_epochs'],
            anneal_strategy=config['swa']['anneal_strategy']
        )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['early_stopping']['patience'],
        min_delta=config['early_stopping']['min_delta']
    )
    
    # Initialize report generator
    report_gen = ReportGenerator(model, config, save_dir=variant_dir / 'logs')
    
    # Training loop
    start_time = time.time()
    for epoch in range(config['training']['num_epochs']):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, targets) in enumerate(data.train_dataloader()):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Apply augmentation (MixUp or CutMix)
            if config['advanced_training'].get('alternate_augment', False):
                # Alternate between MixUp and CutMix
                if batch_idx % 2 == 0 and float(config['advanced_training']['mixup_alpha']) > 0:
                    inputs, targets_a, targets_b, lam = apply_mixup(
                        inputs, targets,
                        float(config['advanced_training']['mixup_alpha']),
                        device
                    )
                    mixed = True
                elif float(config['advanced_training']['cutmix_alpha']) > 0:
                    inputs, targets_a, targets_b, lam = apply_cutmix(
                        inputs, targets,
                        float(config['advanced_training']['cutmix_alpha']),
                        device
                    )
                    mixed = True
                else:
                    mixed = False
            else:
                # Use configured probabilities
                if (config['advanced_training'].get('mixup_prob', 0) > 0 and 
                    np.random.random() < float(config['advanced_training']['mixup_prob'])):
                    inputs, targets_a, targets_b, lam = apply_mixup(
                        inputs, targets,
                        float(config['advanced_training']['mixup_alpha']),
                        device
                    )
                    mixed = True
                elif (config['advanced_training'].get('cutmix_prob', 0) > 0 and 
                      np.random.random() < float(config['advanced_training']['cutmix_prob'])):
                    inputs, targets_a, targets_b, lam = apply_cutmix(
                        inputs, targets,
                        float(config['advanced_training']['cutmix_alpha']),
                        device
                    )
                    mixed = True
                else:
                    mixed = False
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            if mixed:
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                loss = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            if mixed:
                train_correct += (lam * predicted.eq(targets_a).sum().item()
                                + (1 - lam) * predicted.eq(targets_b).sum().item())
            else:
                train_correct += predicted.eq(targets).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in data.val_dataloader():
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        # Calculate metrics
        train_loss = train_loss / len(data.train_dataloader())
        train_acc = 100. * train_correct / train_total
        val_loss = val_loss / len(data.val_dataloader())
        val_acc = 100. * val_correct / val_total
        
        # Update learning rate
        if config.get('swa', {}).get('enabled', False) and epoch >= int(config['swa']['start_epoch']):
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()
        
        # Update report
        report_gen.update(epoch, train_loss, train_acc, val_loss, val_acc)
        
        # Early stopping check
        if early_stopping(val_loss):
            print(f"Early stopping triggered at epoch {epoch}")
            break
    
    # Final evaluation
    if config.get('swa', {}).get('enabled', False):
        # Use SWA model for final evaluation
        model = swa_model
        torch.optim.swa_utils.update_bn(data.train_dataloader(), swa_model, device=device)
    
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for inputs, targets in data.test_dataloader():
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
    
    test_loss = test_loss / len(data.test_dataloader())
    test_acc = 100. * test_correct / test_total
    
    # Generate final report
    report_gen.finalize(test_loss, test_acc, time.time() - start_time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a ResNet18 variant')
    parser.add_argument('config_path', type=str, help='Path to the configuration file')
    args = parser.parse_args()
    
    train_variant(args.config_path) 