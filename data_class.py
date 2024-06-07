import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random

class MNISTCSVDataset(Dataset):
    def __init__(self, csv_file, transform=None, mask_width=7):
        self.data = pd.read_csv(csv_file)
        self.labels = self.data.iloc[:, 0].values
        self.images = self.data.iloc[:, 1:].values.astype(np.float32)

        # Calculate mean and std for Z-score normalization
        self.mean = self.images.mean()
        self.std = self.images.std()

        # Normalize the images
        self.images = (self.images - self.mean) / self.std

        # Optional additional transformations
        self.transform = transform

        # Masking parameters
        self.mask_width = mask_width

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx].reshape(28, 28)  # Reshape to 28x28
        label = self.labels[idx]

        # Convert to tensor
        img = torch.tensor(img).unsqueeze(0)  # Add channel dimension
        label = torch.tensor(label, dtype=torch.long)

        # Create masked version of the image
        masked_img = img.clone()
        masked_img = self.apply_mask(masked_img)

        # Apply additional transformations if any
        if self.transform:
            img = self.transform(img)
            masked_img = self.transform(masked_img)

        return img, masked_img, label

    def apply_mask(self, img):
        # Get the image dimensions
        _, height, width = img.size()

        # Determine the top-left corner of the mask
        top = random.randint(0, height - self.mask_width)
        left = random.randint(0, width - self.mask_width)

        # Apply the mask
        img[:, top:top+self.mask_width, left:left+self.mask_width] = 0

        return img

# # Example usage
# csv_file = '/home/george-vengrovski/Documents/studying/cs_572/final_project/dev.csv'  # Update this path to your CSV file
# dataset = MNISTCSVDataset(csv_file, mask_width=7)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# # Example of how to iterate over the dataloader and visualize masked and unmasked images
# import matplotlib.pyplot as plt

# def visualize_batch(dataloader, num_images=2):
#     batch = next(iter(dataloader))
#     unmasked_images, masked_images, labels = batch
    
#     plt.figure(figsize=(10, 5))
#     for i in range(num_images):
#         plt.subplot(2, num_images, i+1)
#         unmasked_img = unmasked_images[i].squeeze(0).numpy()  # Remove the channel dimension and convert to numpy
#         plt.imshow(unmasked_img, cmap='gray')
#         plt.title(f"Unmasked Label: {labels[i].item()}")
#         plt.axis('off')

#         plt.subplot(2, num_images, i+1+num_images)
#         masked_img = masked_images[i].squeeze(0).numpy()  # Remove the channel dimension and convert to numpy
#         plt.imshow(masked_img, cmap='gray')
#         plt.title(f"Masked Label: {labels[i].item()}")
#         plt.axis('off')
    
#     plt.show()

# # Visualize the first couple of images
# visualize_batch(dataloader, num_images=2)
