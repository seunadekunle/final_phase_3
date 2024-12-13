"""
ResNet18 implementation for CIFAR-10 classification.

Architecture Details:
- Initial Conv Layer: 3x3 conv, 64 filters
- Layer1: 2 basic blocks, 64 channels
- Layer2: 2 basic blocks, 128 channels
- Layer3: 2 basic blocks, 256 channels
- Layer4: 2 basic blocks, 512 channels
- Global Average Pooling
- Fully Connected Layer (512 -> num_classes)

Optional SE (Squeeze-and-Excitation) blocks can be added to each residual block.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    
    def __init__(self, channel, reduction=16):
        """
        Initialize SE block.
        
        Args:
            channel (int): Number of input channels
            reduction (int): Reduction ratio for the bottleneck
        """
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet18.
    
    Architecture:
    input -> conv1 -> bn1 -> relu -> conv2 -> bn2 -> (+input) -> relu -> output
    
    If input channels != output channels or stride != 1:
        input -> 1x1 conv -> bn -> (+) -> output
        
    Optional SE block can be added after conv2.
    """
    
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, use_se=False):
        """
        Initialize basic block.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            stride (int): Stride for convolution (used for downsampling)
            use_se (bool): Whether to use SE block
        """
        super(BasicBlock, self).__init__()
        
        # First convolution block
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, 
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolution block
        self.conv2 = nn.Conv2d(
            out_channels, out_channels * self.expansion, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)
        
        # SE block
        self.se = SEBlock(out_channels * self.expansion) if use_se else None
        
        # Shortcut connection (identity mapping or 1x1 conv)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, self.expansion * out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * out_channels)
            )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        Forward pass of the basic block.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor after residual connection
        """
        identity = x
        
        # First conv block
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        
        # Second conv block
        out = self.bn2(self.conv2(out))
        
        # Apply SE if enabled
        if self.se is not None:
            out = self.se(out)
        
        # Residual connection
        out += self.shortcut(identity)
        out = F.relu(out)
        
        return out

class ResNet18(nn.Module):
    """
    ResNet18 model for CIFAR-10 classification.
    
    Architecture modifications for CIFAR-10:
    1. Initial conv layer has stride=1 instead of stride=2
    2. No max pooling after initial conv
    3. Ends with adaptive average pooling to handle varying input sizes
    4. Optional SE blocks in each residual block
    """
    
    def __init__(self, num_classes=10, dropout_rate=0.0, use_se_blocks=False):
        """
        Initialize ResNet18.
        
        Args:
            num_classes (int): Number of output classes
            dropout_rate (float): Dropout rate for regularization
            use_se_blocks (bool): Whether to use SE blocks in residual blocks
        """
        super().__init__()
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual layers
        self.layer1 = self.make_layer(64, 64, 2, stride=1, use_se=use_se_blocks)
        self.layer2 = self.make_layer(64, 128, 2, stride=2, use_se=use_se_blocks)
        self.layer3 = self.make_layer(128, 256, 2, stride=2, use_se=use_se_blocks)
        self.layer4 = self.make_layer(256, 512, 2, stride=2, use_se=use_se_blocks)
        
        # Global average pooling and final fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
        # Print model summary
        self._print_model_summary(use_se_blocks)

    def make_layer(self, in_channels, out_channels, num_blocks, stride, use_se=False):
        """
        Create a layer of basic blocks.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            num_blocks (int): Number of basic blocks in the layer
            stride (int): Stride for first block (for downsampling)
            use_se (bool): Whether to use SE blocks
            
        Returns:
            nn.Sequential: Sequence of basic blocks
        """
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, use_se))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, use_se=use_se))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Initial conv block
        out = F.relu(self.bn1(self.conv1(x)))
        
        # Residual blocks
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # Global average pooling
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        
        # Final classification
        out = self.dropout(out)
        out = self.fc(out)
        
        return out
    
    def _initialize_weights(self):
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _print_model_summary(self, use_se):
        """Print model architecture summary."""
        print("\nResNet-18 Architecture:")
        print("-" * 50)
        print("Initial Conv Layer: 3 -> 64 channels")
        print(f"Layer1: 64 -> 64 channels, 2 blocks {'with SE' if use_se else ''}")
        print(f"Layer2: 64 -> 128 channels, 2 blocks {'with SE' if use_se else ''}")
        print(f"Layer3: 128 -> 256 channels, 2 blocks {'with SE' if use_se else ''}")
        print(f"Layer4: 256 -> 512 channels, 2 blocks {'with SE' if use_se else ''}")
        print("Global Average Pooling")
        print(f"Fully Connected: 512 -> {self.fc.out_features} classes")
        print("-" * 50)

def count_parameters(model):
    """
    Count number of trainable parameters in the model.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    # Create model and print summary
    model = ResNet18(use_se_blocks=True)
    print(f"\nTotal trainable parameters: {count_parameters(model):,}")