import torch
import argparse
import torchvision.datasets as datasets  # Import standard datasets from torchvision
import torchvision.transforms as transforms  # Import transformations for data augmentation
from torch.utils.data import DataLoader, random_split  # Import utilities for data loading and splitting
from dataset import CatsAndDogs  # Import custom dataset for cats and dogs classification

# Function to convert string input to boolean
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')  # Raise error if input is not a valid boolean

# Function to load standard MNIST dataset for online training
def load_data_online(cfg, batch_size):
    # Download and transform training and test data
    train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)

    # Create data loaders for training and test datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

# Function to load custom dataset (e.g., Cats and Dogs)
def load_data_custom(cfg):
    # Load custom dataset with specified transformations
    datasets = CatsAndDogs(cfg.train_ori_data_path, root_dir=cfg.train_data_path, transform=transforms.ToTensor())
    # Split dataset into training and test data
    train_data, test_data = random_split(datasets, [5, 5])
    # Create data loaders for training and test datasets
    train_loader = DataLoader(dataset=train_data, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=cfg.batch_size, shuffle=True)
    return train_loader, test_loader

# Function to save model checkpoints
def save_checkpoints(state, filename="test_model.pth.tar"):
    print("Saving checkpoints")
    torch.save(state, filename)  # Save the model and optimizer states

# Function to load model checkpoints
def load_checkpoints(checkpoints, model, optimizer):
    print("Loading checkpoints")
    # Load model state and optimizer state from the checkpoint
    model.load_state_dict(checkpoints["state_dict"])
    optimizer.load_state_dict(checkpoints["optimizer"])

# Function to check model accuracy on a given dataset
def check_accuracy(loader, model, device):
    num_correct = 0  # Initialize count of correct predictions
    num_samples = 0  # Initialize count of total samples
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for x, y in loader:  # Iterate over batches of data
            # Move data and labels to the specified device (GPU/CPU)
            x = x.to(device=device)
            y = y.to(device=device)

            # Uncomment to reshape data if needed for NN models
            # x = x.reshape(x.shape[0], -1)

            scores = model(x)  # Perform forward pass to get predictions
            _, prediction = scores.max(1)  # Get the predicted class (with the highest score)

            # Accumulate correct predictions and total samples
            num_correct += (prediction == y).sum()
            num_samples += prediction.size(0)

        model.train()  # Set the model back to training mode
        return num_correct / num_samples  # Return the accuracy as a fraction of correct predictions
