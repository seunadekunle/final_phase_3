"""
Main training script for ResNet18 on CIFAR-10.

This script handles the training loop, evaluation, and model checkpointing.
Features:
- Cross-Entropy Loss
- SGD with momentum and weight decay
- Cosine learning rate scheduler with warmup
- Early stopping
- Model checkpointing
- Training report generation
"""

import os
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import math

from models.resnet import ResNet18
from data.dataset import CIFAR10Data
from config import Config
from utils.report_generator import ReportGenerator

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=7, min_delta=0):
        """
        args:
            patience (int): how many epochs to wait before stopping when loss is
                          not improving
            min_delta (float): minimum change in monitored value to qualify as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def save_checkpoint(state, is_best, checkpoint_dir):
    """
    save model checkpoint
    
    args:
        state: dictionary containing model state and metadata
        is_best: whether this is the best model so far
        checkpoint_dir: directory to save checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # save latest checkpoint
    torch.save(state, checkpoint_dir / 'latest.pth')
    
    # save best checkpoint
    if is_best:
        torch.save(state, checkpoint_dir / 'best.pth')

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer):
    """
    train for one epoch
    
    args:
        model: the neural network model
        train_loader: dataloader for training data
        criterion: loss function
        optimizer: optimization algorithm
        device: device to train on
        epoch: current epoch number
        writer: tensorboard writer
    
    returns:
        tuple: (average loss, accuracy)
    """
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # backward pass
        loss.backward()
        optimizer.step()
        
        # statistics
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # update progress bar
        progress_bar.set_postfix({
            'Loss': f'{train_loss/(batch_idx+1):.3f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
        
        # log to tensorboard
        if batch_idx % Config.log_interval == 0:
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('train/loss', loss.item(), step)
            writer.add_scalar('train/accuracy', 100.*correct/total, step)
            writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], step)
    
    return train_loss/len(train_loader), 100.*correct/total

def evaluate(model, data_loader, criterion, device, prefix='val'):
    """
    evaluate the model
    
    args:
        model: the neural network model
        data_loader: dataloader for evaluation
        criterion: loss function
        device: device to evaluate on
        prefix: prefix for logging ('val' or 'test')
    
    returns:
        tuple: (average loss, accuracy)
    """
    model.eval()
    loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, targets).item()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return loss/len(data_loader), 100.*correct/total

def setup_device():
    """
    set up the appropriate device for M1 Mac
    
    returns:
        torch.device: appropriate device for training
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def get_lr_scheduler(optimizer, num_epochs, warmup_epochs):
    """Creates a learning rate scheduler with warmup and cosine annealing.
    
    Args:
        optimizer: the optimizer to schedule
        num_epochs: total number of epochs
        warmup_epochs: number of warmup epochs
    
    Returns:
        scheduler: CosineAnnealingLR with linear warmup
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # linear warmup
            return float(epoch) / float(max(1, warmup_epochs))
        # cosine decay
        progress = float(epoch - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_model():
    """Main training function."""
    # Set random seed for reproducibility
    torch.manual_seed(Config.seed)
    
    # Setup device
    device = setup_device()
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.seed)
    
    # Create data loaders with augmentation
    data = CIFAR10Data(
        data_dir=Config.data_dir,
        train_batch_size=Config.batch_size,
        eval_batch_size=100,
        num_workers=Config.num_workers,
        autoaugment=Config.randaugment,
        cutout_size=16 if Config.random_erasing_prob > 0 else 0,
        cutout_prob=Config.random_erasing_prob
    )
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    test_loader = data.test_dataloader()
    
    # Create model
    model = ResNet18(num_classes=Config.num_classes)
    model = model.to(device)
    
    # Initialize report generator
    report_gen = ReportGenerator(model, Config)
    
    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.label_smoothing)
    
    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=Config.learning_rate,
        momentum=Config.momentum,
        weight_decay=Config.weight_decay,
        nesterov=Config.nesterov
    )
    
    # Learning rate scheduler
    def lr_lambda(epoch):
        if epoch < Config.warmup_epochs:
            # Linear warmup from warmup_start_lr to learning_rate
            alpha = epoch / Config.warmup_epochs
            return Config.warmup_start_lr / Config.learning_rate * (1 - alpha) + alpha
        # Cosine decay
        progress = float(epoch - Config.warmup_epochs) / float(max(1, Config.num_epochs - Config.warmup_epochs))
        return max(Config.min_lr / Config.learning_rate, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Initialize variables
    best_val_acc = 0.0
    patience_counter = 0
    start_time = time.time()
    
    # Training loop
    for epoch in range(Config.num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Apply mixup
            if Config.mixup_alpha > 0:
                lam = np.random.beta(Config.mixup_alpha, Config.mixup_alpha)
                rand_index = torch.randperm(inputs.size()[0]).to(device)
                targets_a = targets
                targets_b = targets[rand_index]
                inputs = lam * inputs + (1 - lam) * inputs[rand_index]
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            if Config.mixup_alpha > 0:
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
            if Config.mixup_alpha > 0:
                train_correct += (lam * predicted.eq(targets_a).sum().item()
                                + (1 - lam) * predicted.eq(targets_b).sum().item())
            else:
                train_correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{train_loss/(batch_idx+1):.3f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        # Calculate metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update report generator
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': current_lr
        }
        report_gen.update_metrics(epoch_metrics)
        
        # Print epoch results
        print(f'\nEpoch: {epoch+1}')
        print(f'Learning Rate: {current_lr:.6f}')
        print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.2f}%')
        print(f'Best Val Acc: {best_val_acc:.2f}%')
        print('='*70)
        
        # Early stopping check
        if val_acc > best_val_acc + Config.min_delta:
            print(f'Validation accuracy improved from {best_val_acc:.2f} to {val_acc:.2f}')
            best_val_acc = val_acc
            patience_counter = 0
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_acc': best_val_acc
            }, os.path.join(Config.checkpoint_dir, 'best.pth'))
        else:
            patience_counter += 1
            if patience_counter >= Config.patience:
                print('Early stopping triggered')
                break
        
        # Update learning rate
        scheduler.step()
    
    # Load best model and evaluate on test set
    checkpoint = torch.load(os.path.join(Config.checkpoint_dir, 'best.pth'))
    model.load_state_dict(checkpoint['model'])
    
    # Test evaluation
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
    
    test_loss = test_loss / len(test_loader)
    test_acc = 100. * test_correct / test_total
    
    # Calculate total training time
    total_time = time.time() - start_time
    
    # Generate final report
    final_metrics = {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'training_time': total_time / 3600
    }
    
    # Generate plots and report
    report_gen.plot_training_curves()
    report_path = report_gen.generate_baseline_report(final_metrics)
    
    # Print final results
    print('\nFinal Test Results:')
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%')
    print(f'\nTotal training time: {total_time/3600:.2f} hours')
    print(f'\nBaseline report generated at: {report_path}')

def cutmix_data(x, y, alpha=1.0):
    """Performs CutMix augmentation."""
    batch_size = x.size()[0]
    
    # Generate mixed sample
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(batch_size).to(x.device)
    
    # Get random box dimensions
    W = x.size()[2]
    H = x.size()[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    
    # Get random box position
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    # Get box coordinates
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Perform cutmix
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    return x, y, y[rand_index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Criterion for mixed samples."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

if __name__ == '__main__':
    train_model() 