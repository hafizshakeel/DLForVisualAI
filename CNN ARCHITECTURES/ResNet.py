"""
ResNet Architecture Implementation

This script implements the ResNet architecture, which is widely used for image classification tasks.
The ResNet models implemented here include ResNet-50, ResNet-101, and ResNet-152.

Implementation by: Hafiz Shakeel Ahmad Awan
Email: hafizshakeel1997@gmail.com
"""

import torch
import torch.nn as nn

# Define the building block for ResNet architecture
class block(nn.Module):
    def __init__(self, in_channels, intermediate_channels, identity_downsample=None, stride=1):
        """
        Initializes the ResNet block with three convolutional layers and a skip connection.

        Args:
            in_channels (int): Number of input channels for the block.
            intermediate_channels (int): Number of intermediate channels for the second convolutional layer.
            identity_downsample (nn.Sequential, optional): Downsampling layer to match dimensions for skip connection.
            stride (int, optional): Stride for the second convolutional layer. Defaults to 1.
        """
        super(block, self).__init__()
        self.expansion = 4
        # Conv Layer Parameters: (in_channels, out_channels, kernel_size, stride, padding)
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(intermediate_channels, intermediate_channels * self.expansion, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        """
        Forward pass through the ResNet block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor after passing through the block.
        """
        identity = x.clone()  # Clone the input to preserve the identity for the skip connection

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        # Apply identity downsampling if needed
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity   # Add the identity (skip connection) and pass through ReLU activation

        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        """
       Initializes the ResNet model by stacking multiple ResNet blocks.

       Args:
           block (nn.Module): The block class to be used (e.g., `block` defined above).
           layers (list): List containing the number of blocks to be used in each layer.
           image_channels (int): Number of input image channels (e.g., 3 for RGB).
           num_classes (int): Number of output classes for classification.
       """
        super(ResNet, self).__init__()

        self.in_channels = 64
        # Conv Layer Parameters: (in_channels, out_channels, kernel_size, stride, padding)
        self.conv1 = nn.Conv2d(image_channels, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, 2, 1)  # Parameters: (kernel_size, stride, padding)

        # Define ResNet layers
        self.layer1 = self._make_layer(block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], intermediate_channels=512, stride=2)

        # Average pooling to get a fixed-size output and fully connected layer for classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        """
         Forward pass through the ResNet model.

         Args:
             x (torch.Tensor): Input tensor of shape (batch_size, image_channels, height, width).

         Returns:
             torch.Tensor: Output tensor of shape (batch_size, num_classes).
         """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Pass through ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling and flatten the tensor
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        """
        Creates a ResNet layer by stacking blocks.

        Args:
            block (nn.Module): The block class to be used.
            num_residual_blocks (int): Number of blocks to be stacked.
            intermediate_channels (int): Number of intermediate channels for the layer.
            stride (int): Stride to be used in the first block of the layer.

        Returns:
            nn.Sequential: A sequential container of stacked blocks.
        """
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, intermediate_channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(block(self.in_channels, intermediate_channels, identity_downsample, stride))

        self.in_channels = intermediate_channels * 4

        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)


# Functions to return specific ResNet architectures with pre-defined layer configurations
def ResNet50(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)


def ResNet101(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 23, 3], img_channel, num_classes)


def ResNet152(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 8, 36, 3], img_channel, num_classes)


# Testing the implementation with a sample input tensor
def test():
    """
    Tests the ResNet implementation by passing a random tensor through the network
    and checking the output shape.
    """
    BATCH_SIZE = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet50(num_classes=1000).to(device)  # Initialize the ResNet-50 model
    x = torch.randn(BATCH_SIZE, 3, 224, 224).to(device)  # Input: batch_size, 3, 224, 224

    y = model(x)  # Forward pass through the model
    print(f"Output shape: {y.shape}")


if __name__ == "__main__":
    test()
