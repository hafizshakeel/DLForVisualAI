"""
Deep Learning Models Implementation

This script contains implementations of different neural networks:
1. A simple fully connected network (NN)
2. A simple fully convolutional network (CNN)
3. Transfer learning using a pre-trained VGG16 model and GoogLeNet model

Author: Hafiz Shakeel Ahmad Awan
Email: hafizshakeel1997@gmail.com
"""

import torch.nn as nn  # Importing all nn modules
import torch.nn.functional as F  # Importing parameterless functions, including activation functions
import torchvision

""" Simple fully connected network """
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)  # First fully connected layer with 50 hidden units
        self.fc2 = nn.Linear(50, num_classes)  # Second fully connected layer outputting the number of classes

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation to the output of the first layer
        x = self.fc2(x)  # Output layer
        return x


# model = NN(cfg.input_size, cfg.num_classes)  # Example of how to instantiate the NN model

""" Simple fully convolutional network """
class CNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, out_channels=8, kernel_size=3, padding=1)  # First convolutional layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling layer with 2x2 window
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)  # Second convolutional layer
        self.fc = nn.Linear(16 * 7 * 7, num_classes)  # Fully connected layer, outputting the number of classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Apply ReLU and max pooling to the first convolutional layer
        x = self.pool(F.relu(self.conv2(x)))  # Apply ReLU and max pooling to the second convolutional layer
        x = x.reshape(x.shape[0], -1)  # Flatten the output tensor for the fully connected layer
        x = self.fc(x)  # Output layer
        return x


""" Pre-trained model - transfer learning, fine-tuning """
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x  # Identity layer passes input directly to output


def pre_trained(num_classes):
    model = torchvision.models.vgg16(weights="DEFAULT")  # Load pre-trained VGG16 model
    for params in model.parameters():
        params.requires_grad = False  # Freeze all layers

    model.avgpool = Identity()  # Replace avgpool layer with Identity (no operation)
    model.classifier = nn.Sequential(
        nn.Linear(512, 120),  # New fully connected layers for classification
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, num_classes)
    )
    return model


def pre_trained_gnet(num_classes):
    model = torchvision.models.googlenet(weights="DEFAULT")  # Load pre-trained GoogLeNet model

    # Freeze all layers, except the final linear layer
    for param in model.parameters():
        param.requires_grad = False

    # Final layer (not frozen) is updated to match the number of classes
    model.fc = nn.Linear(in_features=1024, out_features=num_classes)

    return model

# model = pre_trained_gnet(cfg.num_classes)  # Example of how to instantiate the pre-trained GoogLeNet model
