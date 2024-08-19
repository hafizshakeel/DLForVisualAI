"""
Deep Learning Models Implementation

This script contains the implementation of the LeNet-5 deep learning model.
It is designed for image classification tasks and has been used extensively for digit recognition
and other image processing tasks.

Author: Hafiz Shakeel Ahmad Awan
Email: hafizshakeel1997@gmail.com
"""

# Import necessary libraries
import torch
import torch.nn as nn

""" LeNet-5 Architecture """

class LeNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        """
        Initializes the LeNet model with specified input channels and number of classes.

        Args:
            in_channels (int): Number of input channels, e.g., 1 for grayscale images.
            num_classes (int): Number of output classes, e.g., 10 for digit classification.
        """
        super(LeNet, self).__init__()

        # Define activation function and pooling layer
        self.relu = nn.ReLU()  # Rectified Linear Unit activation function
        self.pool = nn.AvgPool2d(2, 2)  # Average pooling with kernel size 2x2 and stride 2

        # Define convolutional layers
        # Parameters: (in_channels, out_channels, kernel_size, stride, padding)
        self.conv1 = nn.Conv2d(in_channels, 6, 5, 1, 0)
        self.conv2 = nn.Conv2d(6, 16, 5, 1, 0)
        self.conv3 = nn.Conv2d(16, 120, 5, 1, 0)

        # Define fully connected layers
        # Parameters: (in_features, out_features)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        """
        Forward pass of the LeNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        x = self.pool(self.relu(self.conv1(x)))  # Output shape: Batch x 6 x 14 x 14
        x = self.pool(self.relu(self.conv2(x)))  # Output shape: Batch x 16 x 5 x 5
        x = self.relu(self.conv3(x))  # Output shape: Batch x 120 x 1 x 1
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # Output shape: Batch x num_classes
        return x

# Example to test the final output shape of the model
# model = LeNet()
# x = torch.randn(64, 1, 32, 32)  # Batch of 64 images, each of size 32x32 with 1 channel
# print(model(x).shape)  # Output shape should be (64, 10)
