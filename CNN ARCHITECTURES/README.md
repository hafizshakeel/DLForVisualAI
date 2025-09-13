# CNN Architectures

Implementation of fundamental convolutional neural network (CNN) architectures that have shaped the field of computer vision.

## Architectures

### LeNet-5 (`LeNet.py`)
- Input: 32x32 grayscale images (1 channel)
- Architecture: 3 convolutional layers (6, 16, 120 filters) with average pooling
- Output: 10 classes (configurable)
- Originally designed for digit recognition

### AlexNet (`AlexNet.py`)
- Input: 227x227 RGB images
- Architecture: 5 convolutional layers with ReLU and LRN
- Features: Local Response Normalization, Dropout (0.5)
- Output: 1000 classes (ImageNet)
- Winner of ILSVRC 2012

### VGG (`VGG.py`)
- Multiple configurations: VGG11, VGG13, VGG16, VGG19
- Input: 224x224 RGB images
- Architecture: Small 3x3 conv filters throughout
- Features: Batch normalization, configurable depth
- Output: 1000 classes (configurable)

### GoogLeNet/Inception v1 (`GoogLeNet-Inception.py`)
- Input: 224x224 RGB images
- Architecture: Multiple inception modules with parallel convolutions
- Features: 
  - Auxiliary classifiers during training
  - 1x1, 3x3, 5x5 parallel convolutions
  - Dimensionality reduction
- Output: 1000 classes (configurable)

### ResNet (`ResNet.py`)
- Variants: ResNet50, ResNet101, ResNet152
- Input: 224x224 RGB images
- Architecture: Deep residual learning with skip connections
- Features: 
  - Bottleneck blocks
  - Identity mappings
  - Batch normalization
- Output: 1000 classes (configurable)

## Usage

Each architecture is implemented as a standalone PyTorch module and can be used independently:

```python
from LeNet import LeNet
from AlexNet import AlexNet
from VGG import VGGNet
from GoogLeNet_Inception import GoogLeNet
from ResNet import ResNet50, ResNet101, ResNet152
```

All implementations include test code that can be run directly:

```python
python LeNet.py  # Runs test with random input
```

## Testing

Each model implementation includes unit tests verifying:
- Input/output dimensions
- Forward pass functionality
- Basic shape assertions