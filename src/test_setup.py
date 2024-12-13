"""
Quick test script to validate the entire setup works correctly.
Tests data loading, model creation, and a minimal training loop.
"""

import os
import torch
import yaml
from pathlib import Path

def test_imports():
    """Test all required imports."""
    try:
        import torch
        import torchvision
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import yaml
        print("✓ All required packages imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {str(e)}")
        return False

def test_gpu():
    """Test GPU/MPS availability."""
    if torch.backends.mps.is_available():
        print("✓ MPS (Apple Silicon) is available")
        return "mps"
    elif torch.cuda.is_available():
        print("✓ CUDA (NVIDIA GPU) is available")
        return "cuda"
    else:
        print("! No GPU available, falling back to CPU")
        return "cpu"

def test_data_loading():
    """Test data loading functionality."""
    try:
        from torchvision.datasets import CIFAR10
        from torchvision.transforms import ToTensor
        
        # Try to load a small subset of data
        dataset = CIFAR10(root='./data', train=True, download=True, transform=ToTensor())
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
        
        # Try to load one batch
        images, labels = next(iter(dataloader))
        print(f"✓ Data loading successful. Image shape: {images.shape}, Labels shape: {labels.shape}")
        return True
    except Exception as e:
        print(f"✗ Data loading error: {str(e)}")
        return False

def test_model():
    """Test model creation and forward pass."""
    try:
        from models.resnet import ResNet18
        device = test_gpu()
        
        # Create model
        model = ResNet18(num_classes=10)
        model = model.to(device)
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 32, 32).to(device)
        output = model(dummy_input)
        
        print(f"✓ Model creation and forward pass successful. Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"✗ Model error: {str(e)}")
        return False

def test_training_loop():
    """Test a minimal training loop."""
    try:
        from models.resnet import ResNet18
        from torch.optim import SGD
        from torch.nn import CrossEntropyLoss
        
        device = test_gpu()
        
        # Load minimal dataset
        from torchvision.datasets import CIFAR10
        from torchvision.transforms import ToTensor
        dataset = CIFAR10(root='./data', train=True, download=True, transform=ToTensor())
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
        
        # Setup model and training components
        model = ResNet18(num_classes=10).to(device)
        optimizer = SGD(model.parameters(), lr=0.01)
        criterion = CrossEntropyLoss()
        
        # Run one mini-batch
        images, labels = next(iter(dataloader))
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        print(f"✓ Training loop successful. Loss: {loss.item():.4f}")
        return True
    except Exception as e:
        print(f"✗ Training loop error: {str(e)}")
        return False

def test_variant_configs():
    """Test loading and validation of variant configurations."""
    try:
        variant_paths = [
            "experiments/multi_comparison/resnet_variant_1/config/test_hyperparams.yaml",
            "experiments/multi_comparison/resnet_variant_1/config/hyperparams.yaml",
            "experiments/multi_comparison/resnet_variant_2/config/hyperparams.yaml",
            "experiments/multi_comparison/resnet_variant_3/config/hyperparams.yaml"
        ]
        
        for path in variant_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    config = yaml.safe_load(f)
                print(f"✓ Successfully loaded config: {path}")
            else:
                print(f"! Config not found: {path}")
        return True
    except Exception as e:
        print(f"✗ Config loading error: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("\n=== Running Setup Tests ===\n")
    
    tests = [
        ("Import Test", test_imports),
        ("GPU/MPS Test", test_gpu),
        ("Data Loading Test", test_data_loading),
        ("Model Test", test_model),
        ("Training Loop Test", test_training_loop),
        ("Config Test", test_variant_configs)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with error: {str(e)}")
            results.append(False)
    
    # Summary
    print("\n=== Test Summary ===")
    success_count = sum(1 for r in results if r)
    print(f"Passed: {success_count}/{len(tests)} tests")
    
    if all(results):
        print("\n✓ All systems ready for training!")
    else:
        print("\n! Some tests failed. Please check the output above.")

if __name__ == "__main__":
    main() 