"""
Deep Learning Models Implementation

This script implements the AlexNet deep learning model for image classification tasks.

Implementation by: Hafiz Shakeel Ahmad Awan
Email: hafizshakeel1997@gmail.com
"""

# Import necessary libraries
import torch
import torch.nn as nn

""" AlexNet Architecture """

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        """
        Initializes the AlexNet architecture.

        Parameters:
            num_classes (int): Number of output classes for the final classification layer.
        """
        super().__init__()

        # Define 5 convolutional layers with ReLU activation, normalization, and pooling
        # Input size: (3 x 227 x 227) --> as per AlexNet paper
        self.net = nn.Sequential(
            # Layer 1: Conv -> ReLU -> LRN -> MaxPool
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),  # Output: (96 x 55 x 55)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # LRN for better generalization
            nn.MaxPool2d(kernel_size=3, stride=2),  # Output: (96 x 27 x 27)

            # Layer 2: Conv -> ReLU -> LRN -> MaxPool
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),  # Output: (256 x 27 x 27)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # Output: (256 x 13 x 13)

            # Layer 3: Conv -> ReLU
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),  # Output: (384 x 13 x 13)
            nn.ReLU(),

            # Layer 4: Conv -> ReLU
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),  # Output: (384 x 13 x 13)
            nn.ReLU(),

            # Layer 5: Conv -> ReLU -> MaxPool
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),  # Output: (256 x 13 x 13)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # Output: (256 x 6 x 6)
        )

        # Define 3 fully connected (linear) layers with dropout and ReLU activation
        self.linear = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),  # Fully connected layer 1
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),  # Fully connected layer 2
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=num_classes)  # Final classification layer
        )

        # Initialize model parameters
        self.init_parameters()

    def init_parameters(self):
        """
        Initializes the weights and biases of the network.
        - Weights are initialized with a normal distribution (mean=0, std=0.01).
        - Biases are initialized to 0, except for specific layers (conv2, conv4, conv5).
        """
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        # Set specific biases to 1 for conv2, conv4, and conv5 layers (as per paper)
        nn.init.constant_(self.net[4].bias, 1)  # Conv2
        nn.init.constant_(self.net[10].bias, 1)  # Conv4
        nn.init.constant_(self.net[12].bias, 1)  # Conv5

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters:
            x (Tensor): Input tensor of shape (batch_size, 3, 227, 227)

        Returns:
            Tensor: Output predictions of shape (batch_size, num_classes)
        """
        x = self.net(x)  # Pass through convolutional layers
        x = x.view(-1, 256 * 6 * 6)  # Flatten for linear layers
        out = self.linear(x)  # Pass through fully connected layers
        return out


def test_model():
    """
    Tests the AlexNet model for a single forward pass with random input data.
    """
    # Define input size: batch_size=4, channels=3, height=227, width=227
    input_tensor = torch.randn(4, 3, 227, 227)

    # Create an instance of the model
    model = AlexNet(num_classes=1000)
    # print("Model architecture:")
    # print(model)

    # Perform forward pass
    output = model(input_tensor)
    print("\nOutput shape:", output.shape)

    # Check if output dimensions match the expected shape: (1, num_classes)
    assert output.shape == (4, 1000), "Output shape mismatch!"
    print("Test passed: Output shape is correct.")


# Test the model
if __name__ == "__main__":
    test_model()
