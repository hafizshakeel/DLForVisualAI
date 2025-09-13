# Neural Network Basics

Training infrastructure and utilities for implementing and training neural networks, with a focus on computer vision tasks.

## Components

### Models (`models.py`)
- **Simple Neural Network**: Fully connected network with configurable input size and classes
- **Basic CNN**: Two-layer CNN with max pooling
- **Transfer Learning**:
  - VGG16-based model with custom classifier
  - GoogLeNet-based model with frozen feature extraction

### Training Infrastructure (`train.py`)
- TensorBoard integration for visualization:
  - Loss and accuracy tracking
  - Model graph visualization
  - Embedding visualization
  - Parameter histograms
- Hyperparameter grid search
- Automatic GPU/CPU device selection
- Training loop with progress tracking
- Model checkpointing

### Data Management (`dataset.py`, `utils.py`)
- Built-in support for MNIST dataset
- Custom dataset implementation (CatsAndDogs):
  - CSV-based data loading
  - Image preprocessing
  - Data augmentation support
- Data loading utilities:
  - Batch processing
  - Train/test splitting
  - Data loading workers

### Configuration (`config.py`)
Dataset paths:
- Training/validation/test data paths
- Ground truth paths
- Sample output folder

Model parameters:
- Learning rate (default: 1e-3)
- Batch size (default: 64)
- Number of epochs (default: 5)
- Weight decay (default: 0.0001)
- Input dimensions (default: 784)
- Number of classes (default: 10)

Runtime options:
- Model checkpointing
- GPU usage
- Data loader workers
- Gradient clipping

## Usage

1. Configure parameters in `config.py`:
```python
# Example configuration
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=5)
```

2. Run training:
```bash
python train.py
```

3. Monitor training:
```bash
tensorboard --logdir=runs
```

## Features

- Support for both standard (MNIST) and custom datasets
- Configurable training pipeline
- TensorBoard integration for visualization
- Model checkpointing and loading
- GPU acceleration support
- Progress tracking with tqdm