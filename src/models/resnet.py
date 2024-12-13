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

Total number of layers: 18 (1 + 16 conv layers in basic blocks + 1 FC layer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet18.
    
    Architecture:
    input -> conv1 -> bn1 -> relu -> conv2 -> bn2 -> (+input) -> relu -> output
    
    If input channels != output channels or stride != 1:
        input -> 1x1 conv -> bn -> (+) -> output
    """
    
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        """
        initialize basic block
        
        args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            stride (int): stride for convolution (used for downsampling)
        """
        super(BasicBlock, self).__init__()
        
        # first convolution block
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, 
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # second convolution block
        self.conv2 = nn.Conv2d(
            out_channels, out_channels * self.expansion, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)
        
        # shortcut connection (identity mapping or 1x1 conv)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, self.expansion * out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * out_channels)
            )
        
        # dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        forward pass of the basic block
        
        args:
            x (torch.Tensor): input tensor
            
        returns:
            torch.Tensor: output tensor after residual connection
        """
        identity = x
        
        # first conv block
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        
        # second conv block
        out = self.bn2(self.conv2(out))
        
        # residual connection
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
    """
    
    def __init__(self, num_classes=10, dropout_rate=0.0):
        """
        initialize resnet18
        
        args:
            num_classes (int): number of output classes
            dropout_rate (float): dropout rate for regularization
        """
        super().__init__()
        
        # initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # residual layers
        self.layer1 = self.make_layer(64, 64, 2, stride=1)
        self.layer2 = self.make_layer(64, 128, 2, stride=2)
        self.layer3 = self.make_layer(128, 256, 2, stride=2)
        self.layer4 = self.make_layer(256, 512, 2, stride=2)
        
        # global average pooling and final fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)
        
        # initialize weights
        self._initialize_weights()
        
        # print model summary
        self._print_model_summary()

    def make_layer(self, in_channels, out_channels, num_blocks, stride):
        """
        create a layer of basic blocks
        
        args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            num_blocks (int): number of basic blocks in the layer
            stride (int): stride for first block (for downsampling)
            
        returns:
            nn.Sequential: sequence of basic blocks
        """
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        forward pass of the network
        
        args:
            x (torch.Tensor): input tensor of shape (batch_size, 3, H, W)
            
        returns:
            torch.Tensor: output tensor of shape (batch_size, num_classes)
        """
        # initial conv block
        out = F.relu(self.bn1(self.conv1(x)))
        
        # residual blocks
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # global average pooling
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        
        # final classification
        out = self.dropout(out)
        out = self.fc(out)
        
        return out
    
    def _initialize_weights(self):
        """
        initialize model weights using kaiming initialization
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def _print_model_summary(self):
        """print model architecture summary"""
        print("\nResNet-18 Architecture:")
        print("-" * 50)
        print("Initial Conv Layer: 3 -> 64 channels")
        print("Layer1: 64 -> 64 channels, 2 blocks")
        print("Layer2: 64 -> 128 channels, 2 blocks")
        print("Layer3: 128 -> 256 channels, 2 blocks")
        print("Layer4: 256 -> 512 channels, 2 blocks")
        print("Global Average Pooling")
        print(f"Fully Connected: 512 -> {self.fc.out_features} classes")
        print("-" * 50)

def count_parameters(model):
    """
    count number of trainable parameters in the model
    
    args:
        model (nn.Module): pytorch model
        
    returns:
        int: number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    # create model and print summary
    model = ResNet18()
    print(f"\nTotal trainable parameters: {count_parameters(model):,}")