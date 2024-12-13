"""
CIFAR-10 dataset loading and preprocessing module.

This module handles the loading and preprocessing of the CIFAR-10 dataset,
including data augmentation for training.
"""

import torch
import torchvision
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from torchvision.transforms import RandAugment, RandomErasing

class TransformDataset(Dataset):
    """Dataset wrapper that applies transforms."""
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if isinstance(img, Image.Image):
            # If already PIL Image, apply transform directly
            if self.transform:
                img = self.transform(img)
        else:
            # If numpy array, convert to PIL Image first
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            if self.transform:
                img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.dataset)

class CIFAR10Data:
    """CIFAR-10 data loading and preprocessing class."""
    
    def __init__(self, data_dir='./data', train_batch_size=128, eval_batch_size=100, 
                 num_workers=4, autoaugment=True, cutout_size=16, cutout_prob=1.0):
        """Initialize the data loading pipeline.
        
        Args:
            data_dir (str): directory to store the dataset
            train_batch_size (int): batch size for training
            eval_batch_size (int): batch size for validation and testing
            num_workers (int): number of workers for data loading
            autoaugment (bool): whether to use RandAugment
            cutout_size (int): size of cutout patches
            cutout_prob (float): probability of applying cutout
        """
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        
        # CIFAR-10 mean and std
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2023, 0.1994, 0.2010]
        
        # Build augmentation pipeline
        train_transforms = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ]
        
        if autoaugment:
            train_transforms.append(RandAugment(num_ops=2, magnitude=14))
        
        train_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        if cutout_prob > 0:
            train_transforms.append(RandomErasing(p=cutout_prob, scale=(0.02, 0.2)))
        
        self.train_transform = transforms.Compose(train_transforms)
        
        # Test transform
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        self.prepare_data()
    
    def prepare_data(self):
        """Download and split the data."""
        # Download
        trainset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=None  # No transform here
        )
        
        testset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=None  # No transform here
        )
        
        # Create validation split
        train_size = int(0.9 * len(trainset))
        val_size = len(trainset) - train_size
        
        # Use fixed generator for reproducibility
        generator = torch.Generator().manual_seed(42)
        
        # Create transformed datasets
        self.trainset = TransformDataset(trainset, self.train_transform)
        self.testset = TransformDataset(testset, self.test_transform)
        
        # Split train into train and val
        indices = list(range(len(self.trainset)))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        self.trainset = Subset(self.trainset, train_indices)
        self.valset = Subset(TransformDataset(trainset, self.test_transform), val_indices)
    
    def train_dataloader(self):
        """Create training dataloader."""
        return DataLoader(
            self.trainset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
    
    def val_dataloader(self):
        """Create validation dataloader."""
        return DataLoader(
            self.valset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
    
    def test_dataloader(self):
        """Create test dataloader."""
        return DataLoader(
            self.testset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        ) 