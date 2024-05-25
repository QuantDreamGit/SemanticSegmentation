import os
import torch
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
from PIL import Image

class CityScapes(Dataset):
    def __init__(self, root_dir, split='train', mode='multiple', raw_label=False):
        super(CityScapes, self).__init__()
        # Save the root directory of the dataset
        self.root_dir = root_dir
        # Save the Channel mode
        self.mode = mode
        # Save the label raw mode
        self.raw_label = raw_label
        # Define the transformations for the label
        self.transform_label = Compose([
            Resize((512, 1024)),
        ])
        # Define the transformations for the image
        self.transform_image = Compose([
            Resize((512, 1024)),
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

    def convert_from_image_to_label(self, image):
        # Convert the image to a numpy array
        image = np.array(image)

        # Create a new image where each pixel is labeled
        labels = {
            0: 0,   # road
            1: 1,   # sidewalk
            2: 2,   # building
            3: 3,   # wall
            4: 4,   # fence
            5: 5,   # pole
            6: 6,   # traffic light
            7: 7,   # traffic sign
            8: 8,   # vegetation
            9: 9,   # terrain
            10: 10, # sky
            11: 11, # person
            12: 12, # rider
            13: 13, # car
            14: 14, # truck
            15: 15, # bus
            16: 16, # train
            17: 17, # motorcycle
            18: 18, # bicycle
        }
        # Iterate over each pixel value and assign 255 if it is not in the labels
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i][j] > 18:
                    image[i][j] = 255

        # Convert the image to a tensor
        return torch.tensor(image, dtype=torch.long)

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
        image = self.transform_image(Image.open(os.path.join(self.image_dir, city, images[idx])))

        # Load the label
        if self.mode == 'multiple':
            # Load the color label
            label = self.transform_label(Image.open(os.path.join(self.label_dir, city, images[idx].replace('leftImg8bit', 'gtFine_color'))))
        else:
            # Load the single channel label
            label = self.transform_label(Image.open(os.path.join(self.label_dir, city, images[idx].replace('leftImg8bit', 'gtFine_labelTrainIds'))))
            if self.raw_label == False:
                # Convert the label to the correct format
                label = self.convert_from_image_to_label(label)

        # Return the image and label
        return image, label

    def __len__(self):
        # Sum the number of images in each city
        # This is the total number of images in the dataset
        return sum(len(os.listdir(os.path.join(self.image_dir, city))) for city in self.cities)