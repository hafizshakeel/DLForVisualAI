"""
Deep Learning Models Implementation

This script contains the implementation of VGG architectures.
It is designed for image classification tasks and demonstrates the VGG model variations.

Implementation by: Hafiz Shakeel Ahmad Awan
Email: hafizshakeel1997@gmail.com
"""

import torch
import torch.nn as nn

""" VGG Architectures """

# Dictionary defining different VGG architectures and each list specifies the number of filters for
# each convolutional layer and 'M' for max pooling layers.
VGG_type = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

class VGGNet(nn.Module):
    """
    VGGNet model for image classification.

    Implements the VGG architecture with convolutional layers followed by fully connected layers.
    The architecture can be chosen from VGG11, VGG13, VGG16, or VGG19 based on the configuration in VGG_type.

    Attributes:
        in_channels (int): Number of input channels (default: 3 for RGB images).
        num_classes (int): Number of output classes (default: 1000 for ImageNet classification).
    """

    def __init__(self, in_channels=3, num_classes=1000):
        """
        Initializes the VGGNet model with specified input channels and number of classes.

        Args:
            in_channels (int): Number of input channels (default: 3 for RGB images).
            num_classes (int): Number of output classes (default: 1000 for ImageNet classification).
        """
        super(VGGNet, self).__init__()
        self.in_channels = in_channels
        # Create convolutional layers using function create_conv_layers (defined below)
        self.conv_layers = self.create_conv_layers(VGG_type["VGG16"])  # Select type of VGG
        # Next, define fully connected layers with specified input and output features
        self.fcs = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # Input: 512*7*7 features, Output: 4096 features
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Dropout layer with 50% probability
            nn.Linear(4096, 4096),  # Input: 4096 features, Output: 4096 features
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)  # Input: 4096 features, Output: num_classes features
        )

    def forward(self, x):
        """
        Forward pass of the VGGNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        x = self.conv_layers(x)  # Apply convolutional layers
        x = x.reshape(x.shape[0], -1)  # Flatten the tensor for fully connected layers
        x = self.fcs(x)  # Apply fully connected layers
        return x

    def create_conv_layers(self, architecture):
        """
        Creates convolutional layers based on the specified architecture.

        Args:
            architecture (list): List defining the VGG architecture, where each element is either an integer
                                  (number of filters) or 'M' (max pooling).

        Returns:
            nn.Sequential: Sequential container of convolutional and pooling layers.
        """
        in_channels = self.in_channels
        layers = []

        for x in architecture:
            if type(x) == int:
                # Add convolutional layer followed by batch normalization and ReLU activation
                out_channels = x
                layers += [nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU()
                           ]
                in_channels = x
            elif x == "M":
                # Add max pooling layer
                layers += [nn.MaxPool2d(2, 2)]  # Parameters: (Kernel_size, stride)

        return nn.Sequential(*layers)  # Return a sequential container of the created layers

# Model test
# model = VGGNet(in_channels=3, num_classes=1000)
# x = torch.randn(64, 3, 224, 224)  # Batch of 64 RGB images of size 224x224
# print(model(x).shape)  # Output shape should be (64, 1000)
