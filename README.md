# Vision CNN Architectures

A collection of PyTorch implementations of fundamental convolutional neural network architectures for computer vision tasks.

## Repository Structure

### CNN ARCHITECTURES/
Contains standalone implementations of classic CNN architectures:
- **LeNet-5**: The CNN architecture for digit recognition (1998)
- **AlexNet**: Winner of ILSVRC 2012, featuring ReLU activations and local response normalization
- **VGG** (11/13/16/19): Deep architectures using small 3x3 convolution filters
- **GoogLeNet** (Inception v1): Network featuring parallel convolution filters of different sizes
- **ResNet** (50/101/152): Very deep networks with residual connections
- Includes architecture diagrams in the Images/ folder for visual reference

### NN BASICS/
Training infrastructure and utilities for model development:
- Basic model implementations (`models.py`)
- Training loops with TensorBoard logging (`train.py`)
- Data loading utilities (`utils.py`, `dataset.py`)
- Configuration management (`config.py`)

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/hafizshakeel/vision-cnn-architectures.git
cd vision-cnn-architectures
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Choose a model to work with:

For training basic models:
```bash
cd NN BASICS
# Edit config.py to set your parameters
python train.py
```

For using specific architectures:
```bash
cd CNN ARCHITECTURES
# Each architecture can be imported and used independently
# See test_model() in each file for usage examples
```

## License

This project is licensed under the MIT License.

## Contact
  
Email: hafizshakeel1997@gmail.com
