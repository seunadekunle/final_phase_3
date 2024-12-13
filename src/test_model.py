"""
Script to test ResNet-18 model implementation.
"""

import torch
from models.resnet import ResNet18

def test_model():
    """test resnet18 implementation"""
    # create model
    model = ResNet18()
    
    # test with random input
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)  # CIFAR-10 image size
    
    # forward pass
    out = model(x)
    
    # print shapes
    print("\nTesting forward pass:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    # verify output dimensions
    assert out.shape == (batch_size, 10), f"Expected output shape (4, 10), got {out.shape}"
    print("\nAll tests passed!")

if __name__ == '__main__':
    test_model() 