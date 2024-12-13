"""
Visualization utilities for CIFAR-10 dataset.

This module provides functions to visualize samples from the CIFAR-10 dataset.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

# CIFAR-10 classes
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

def denormalize_image(image):
    """
    denormalize image from cifar10 normalization
    
    args:
        image (torch.Tensor): normalized image tensor
    
    returns:
        torch.Tensor: denormalized image tensor
    """
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2023, 0.1994, 0.2010])
    
    # add batch dimension if not present
    if len(image.shape) == 3:
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
    
    return image * std + mean

def show_samples(data_loader, num_samples=4):
    """
    display random samples from the dataset
    
    args:
        data_loader: pytorch dataloader for cifar10
        num_samples (int): number of samples to display
    """
    # get random samples
    images, labels = next(iter(data_loader))
    
    # create subplot grid
    fig, axes = plt.subplots(1, num_samples, figsize=(12, 3))
    
    for i in range(num_samples):
        # get image and denormalize
        img = denormalize_image(images[i])
        label = CLASSES[labels[i]]
        
        # convert to numpy and transpose
        img = img.permute(1, 2, 0).numpy()
        # clip values to valid range
        img = np.clip(img, 0, 1)
        
        # display image
        axes[i].imshow(img)
        axes[i].set_title(label)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # example usage
    from data.dataset import CIFAR10Data
    
    # create data loaders
    data = CIFAR10Data()
    train_loader, _, _ = data.get_data_loaders()
    
    # show samples
    show_samples(train_loader) 