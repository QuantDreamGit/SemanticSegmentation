from torch.utils.data import Dataset
import os

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize, Compose

from PIL import Image
import matplotlib.pyplot as plt

# TODO: implement here your custom dataset class for GTA5


class GTA5(Dataset):
    def __init__(self,root_dir):
        super(GTA5, self).__init__()
        # Save the root directory of the dataset
        self.root_dir = root_dir
        # Reduce the size of the images to 512x1024
        self.transform = Compose([
            Resize((512, 1024)),
            ToTensor()
        ])
        # Image and label directories
        self.image_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'labels')
        

        pass

    def __getitem__(self, idx):
        
        images = os.listdir(self.image_dir)
        
        
    
        # Load the image and label
        image = self.transform(Image.open(os.path.join(self.image_dir,  images[idx])))
        label = self.transform(Image.open(os.path.join(self.label_dir, images[idx])))
        
        # Return the image and label
        return image, label

        pass

    def __len__(self):

        # Sum the number of images in each city
        # This is the total number of images in the dataset
        return len(os.listdir(self.image_dir))

        pass



# Create the dataset
dataset = GTA5(root_dir="datasets/GTA5")
print(len(dataset))
# Get the first image and label
image, label = dataset[2499]
# Convert PyTorch tensor to numpy array for plotting
image = image.permute(1, 2, 0).numpy()
label = label.squeeze().numpy()

# Plot the image and label
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(label, cmap='viridis')
plt.title('Label')
plt.axis('off')
plt.show()