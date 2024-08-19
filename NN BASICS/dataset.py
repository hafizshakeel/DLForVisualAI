import torch
from torch.utils.data import Dataset  # Import Dataset class for custom datasets
import pandas as pd  # Import pandas for handling CSV files
import os  # Import os for file path management
from PIL import Image  # Import PIL for image processing


class CatsAndDogs(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file containing image paths and labels.
            root_dir (str): Directory where images are stored.
            transform (callable, optional): Optional transform to be applied on an image sample.
        """
        self.annotations = pd.read_csv(csv_file)  # Read CSV file containing image paths and labels
        self.root_dir = root_dir  # Set the root directory where images are stored
        self.transform = transform  # Set any transformations to apply to the images

    def __len__(self):
        """
        Returns:
            int: The total number of samples in the dataset.
        """
        return len(self.annotations)  # Return the length of the dataset (number of samples)

    def __getitem__(self, item):
        """
        Args:
            item (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, target) where target is the class label of the image.
        """
        # Construct the full path to the image file
        image_path = os.path.join(self.root_dir, self.annotations.iloc[item, 0])
        image = Image.open(image_path)  # Open the image file
        target = torch.tensor(int(self.annotations.iloc[item, 1]))  # Retrieve the label and convert it to a tensor

        # Apply transformations to the image if provided
        if self.transform:
            image = self.transform(image)

        return image, target  # Return the image and its corresponding label
