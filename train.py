# TODO: Define here your training and validation loops.

import torch
import warnings
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from datasets.cityscapes import CityScapes
from tqdm import tqdm

from models.deeplabv2.deeplabv2 import get_deeplab_v2
from utils import fast_hist

warnings.filterwarnings("ignore")
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def train():
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the model
    model = get_deeplab_v2(root_dir='models/deeplabv2').to(device)

    # Load the dataset
    dataset = CityScapes(root_dir='datasets/Cityspaces', split='val', mode='single', label_raw=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Since we are dealing with classification, we will use the cross entropy loss
    # However, we need to ignore the class 0 (background) when calculating the loss
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Set the model to training mode
    model.train()
    # Train the model
    for epoch in range(10):
        for i, (image, label) in tqdm(enumerate(dataloader)):
            image = image.to(device)
            label = label.to(device)
            # Zero the gradient
            optimizer.zero_grad()
            # Forward pass
            output = model(image)
            # Calculate the loss
            loss = criterion(output, label)
            # Backward pass
            loss.backward()
            # Optimize
            optimizer.step()    

train()
