"""
Deep Learning Models Implementation

This script contains the implementation of the GoogLeNet (Inception v1) deep learning model.
It is designed for image classification tasks and has been used extensively in various
computer vision tasks, including the ImageNet classification challenge.

Author: Hafiz Shakeel Ahmad Awan
Email: hafizshakeel1997@gmail.com
"""

# Import necessary libraries
import torch
import torch.nn as nn

""" GoogLeNet-Inception Architecture """

class GoogLeNet(nn.Module):
    def __init__(self, aux_logits=True, num_classes=1000):
        """
        Initializes the GoogLeNet model with optional auxiliary classifiers.

        Args:
            aux_logits (bool): If True, include auxiliary classifiers in the model.
            num_classes (int): Number of output classes, e.g., 1000 for ImageNet.
        """
        super(GoogLeNet, self).__init__()
        assert aux_logits == True or aux_logits == False
        self.aux_logits = aux_logits

        self.conv1 = conv_block(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = conv_block(64, 192, kernel_size=3, stride=1, padding=1)


        # Inception modules params: (in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool)
        # Match with table values see GoogLeNet-table.png
        self.inception3a = Inception_module(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception_module(256, 128, 128, 192, 32, 96, 64)

        self.inception4a = Inception_module(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception_module(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception_module(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception_module(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception_module(528, 256, 160, 320, 32, 128, 128)

        self.inception5a = Inception_module(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception_module(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, num_classes)

        # Define auxiliary classifiers if aux_logits is True
        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        else:
            self.aux1 = self.aux2 = None


    def forward(self, x):
        """
        Forward pass of the GoogLeNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            Output tensor or a tuple with auxiliary classifier outputs if aux_logits is True.
        """
        # Apply initial convolution and max pooling
        x = self.conv1(x)
        x = self.maxpool1(x)

        # Apply second convolution and max pooling
        x = self.conv2(x)
        x = self.maxpool2(x)

        # Apply inception modules in the third block
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool2(x)

        # Apply inception module 4a, and if aux_logits is True, output the first auxiliary classifier
        x = self.inception4a(x)
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)

        # Continue with the remaining inception modules
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        # Apply inception module 4e, and if aux_logits is True, output the second auxiliary classifier
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)

        # Final inception modules and fully connected layers
        x = self.inception4e(x)
        x = self.maxpool2(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)  # Flatten the tensor for the fully connected layer
        x = self.dropout(x)
        x = self.fc1(x)

        # Return output, including auxiliary outputs if they are enabled during training
        if self.aux_logits and self.training:
            return aux1, aux2, x
        else:
            return x


""" 
Inception Module 
This module is used to perform multiple convolutions on the same input with different filter sizes, 
allowing the network to capture different types of features at different scales.
"""


# see image Inception-module.png -b
class Inception_module(nn.Module):
    def __init__(
            self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool
    ):
        """
        Initializes the Inception module with the specified number of filters.

        Args:
            in_channels (int): Number of input channels.
            out_1x1 (int): Number of output channels for the 1x1 convolution branch.
            red_3x3 (int): Number of output channels for the 1x1 reduction before 3x3 convolution.
            out_3x3 (int): Number of output channels for the 3x3 convolution branch.
            red_5x5 (int): Number of output channels for the 1x1 reduction before 5x5 convolution.
            out_5x5 (int): Number of output channels for the 5x5 convolution branch.
            out_1x1pool (int): Number of output channels for the 1x1 convolution after pooling.
        """
        super(Inception_module, self).__init__()

        # First branch: 1x1 convolution
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=1)

        # Second branch: 1x1 reduction followed by 3x3 convolution
        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size=1),
            conv_block(red_3x3, out_3x3, kernel_size=(3, 3), padding=1),
        )

        # Third branch: 1x1 reduction followed by 5x5 convolution
        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size=1),
            conv_block(red_5x5, out_5x5, kernel_size=5, padding=2),
        )

        # Fourth branch: 3x3 max pooling followed by 1x1 convolution
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv_block(in_channels, out_1x1pool, kernel_size=1),
        )

    def forward(self, x):
        """
        Forward pass of the Inception module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Concatenated output from all four branches.
        """
        # Concatenate the outputs of the four branches along the channel dimension
        return torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1
        )


""" 
Inception Auxiliary Classifier 
This is an auxiliary classifier used during training to help the gradient flow through the network.
"""

# see image GoogLeNet auxiliary classifier

class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        """
        Initializes the InceptionAux module, which acts as an auxiliary classifier.

        Args:
            in_channels (int): Number of input channels.
            num_classes (int): Number of output classes.
        """
        super(InceptionAux, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.7)
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = conv_block(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        """
        Forward pass of the InceptionAux module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with class probabilities.
        """
        x = self.pool(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)  # Flatten the tensor for fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


""" 
Convolutional Block 
This is a helper class for constructing convolutional layers followed by batch normalization 
and a ReLU activation function.
"""

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        """
        Initializes the conv_block module, which consists of a convolutional layer
        followed by batch normalization and a ReLU activation.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            **kwargs: Additional arguments for the convolutional layer (e.g., kernel_size, stride, padding).
        """
        super(conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """
        Forward pass of the conv_block module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after convolution, batch normalization, and ReLU activation.
        """
        return self.relu(self.batchnorm(self.conv(x)))


if __name__ == "__main__":
    BATCH_SIZE = 5
    x = torch.randn(BATCH_SIZE, 3, 224, 224)  # input rand tensor
    # Instantiate the GoogLeNet model with auxiliary classifiers enabled
    model = GoogLeNet(aux_logits=True, num_classes=1000)
    print(model(x)[2].shape)
    assert model(x)[2].shape == torch.Size([BATCH_SIZE, 1000])
