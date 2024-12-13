"""
Script to verify CIFAR-10 dataset integrity and display statistics.
"""

from data.dataset import CIFAR10Data
from utils.visualize import show_samples

def main():
    """main function to verify dataset"""
    print("Loading CIFAR-10 dataset...")
    data = CIFAR10Data()
    train_loader, val_loader, test_loader = data.get_data_loaders()
    
    # print dataset statistics
    print("\nDataset Statistics:")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # verify batch sizes
    print("\nBatch Sizes:")
    print(f"Training batch size: {train_loader.batch_size}")
    print(f"Validation batch size: {val_loader.batch_size}")
    print(f"Test batch size: {test_loader.batch_size}")
    
    # show sample images
    print("\nDisplaying sample images from training set...")
    show_samples(train_loader)
    
    print("\nData verification complete!")

if __name__ == '__main__':
    main() 