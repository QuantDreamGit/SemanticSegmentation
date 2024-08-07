import os
import cv2
import torch
import numpy as np
import albumentations as A

from numba import njit
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
from PIL import Image

@njit
def convert_label(label):
    # Iterate over each pixel value and assign 255 if it is not in the labels
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i][j] > 18:
                label[i][j] = 255

    # Return the labels
    return label

class CityScapes(Dataset):
    def __init__(self, root_dir, split='train',
                 custom_transform_image=None, custom_transform_label=None):
        super(CityScapes, self).__init__()
        # Save the root directory of the dataset
        self.root_dir = root_dir
        # Save the custom transformations
        self.custom_transform_image = custom_transform_image
        self.custom_transform_label = custom_transform_label
        # Define the transformations for the label
        self.transform_label = A.Compose([
            A.Resize(height=512, width=1024),
        ])
        # Define the transformations for the image
        self.transform_image = Compose([
            Resize((512,1024)),
            ToTensor(),
            Normalize(
                mean=[0.2954, 0.3339, 0.2950], 
                std=[0.1822, 0.1852, 0.1807]
            )
        ])

        # Image and label directories
        self.image_dir = os.path.join(root_dir, 'images', split)
        self.label_dir = os.path.join(root_dir, 'gtFine', split)
        # Get the list of cities
        self.cities = sorted(os.listdir(self.image_dir))

    def __getitem__(self, idx):
        # Find the city and image index
        for city in self.cities:
            # Get the list of images in the city
            images = os.listdir(os.path.join(self.image_dir, city))
            # Check if the index is in the range of the city
            if idx < len(images):
                # If it is, break the loop
                break
            # Otherwise, subtract the number of images in the city from the index
            # This will move the index to the next city
            idx -= len(images)

        # Load the image and label
        # Check if there are custom transformations
        if self.custom_transform_image != None:
            # Load the image with the custom transformation
            image = self.custom_transform_image(
                image=np.array(Image.open(os.path.join(self.image_dir, city, images[idx])))
            )['image']
        else:
            # Load the image with the default transformation
            image = self.transform_image(Image.open(os.path.join(self.image_dir, city, images[idx])))

        # Load the single channel label
        label = cv2.imread(
            os.path.join(self.label_dir, city, images[idx].replace('leftImg8bit', 'gtFine_labelTrainIds')), 
            cv2.IMREAD_GRAYSCALE)
        
        label = self.transform_label(image=label)['image']
        # Convert the label
        label = convert_label(label)
        # Transform the label to a tensor
        label = torch.tensor(np.array(label)).long()

        # Return the image and label
        return image, label

    def __len__(self):
        # Sum the number of images in each city
        # This is the total number of images in the dataset
        return sum(len(os.listdir(os.path.join(self.image_dir, city))) for city in self.cities)