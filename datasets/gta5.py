import os
import torch
import numpy as np
from numba import njit

from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize

@njit
def convert_label(label):
    # Iterate over each pixel value and assign 255 if it is not in the labels
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i][j] > 18:
                label[i][j] = 255
    
    # Return the labels
    return label

class GTA5(Dataset):
    def __init__(self, root_dir, augmentations=None):
        super(GTA5, self).__init__()
        # Save the root directory
        self.root_dir = root_dir
        # Define the transform for the image and label
        self.transform_image = Compose([
            # Resize the image to 1280x720
            Resize((720, 1280)),
            ToTensor(),
            Normalize(
                mean=[0.5084, 0.5021, 0.4838], 
                std=[0.2490, 0.2440, 0.2424],

            )
        ])
        self.transform_label = Compose([
            Resize((720, 1280)),
        ])
        
        # Save image and label paths
        self.image_paths = os.path.join(self.root_dir, 'images')
        self.label_paths = os.path.join(self.root_dir, 'labels')

    def __getitem__(self, idx):
        # The images and labels should have the same name
        # Each imaage has the following name: 00000.png
        # Each label has the following name: 00000.png
        image_path = os.path.join(self.image_paths, f'{str(idx+1).zfill(5)}.png')
        label_path = os.path.join(self.label_paths, f'{str(idx+1).zfill(5)}.png')

        # Open the image and label
        image = Image.open(image_path)
        label = Image.open(label_path)
        # Apply the transform
        image = self.transform_image(image)
        label = self.transform_label(label)
        # Convert the label to the same format as cityscapes
        label = convert_label(np.array(label))
        # Transform the label to a tensor
        label = torch.tensor(label).long()

        # Augment the image

        # Return the image and label
        return image, label

    def __len__(self):
        # Get the number of images
        return len(os.listdir(self.image_paths))
