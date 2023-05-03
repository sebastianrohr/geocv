import os
import random
import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image

# Add augmented images to the dataset, to the same file each is located, adding 'aug' as  image name suffix

# Set up the augmentation pipeline
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.7),
    transforms.RandomRotation(degrees=(-10, 10)),
    transforms.RandomGrayscale(p=0.5),
    transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.95, 1.05), shear=(-5, 5)),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.1),
    transforms.ToTensor()

    #normalize function, done by tokenizer
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #transforms.Normalize(mean=[0.5], std=[0.5])
])

# Define the paths to the folders with the images
data_dir = "output"
city_folders = ["copenhagen", "madrid", "london", "moscow"]
subfolders = ["panos", "singles"]

# Loop over the city folders and augment the images
for city_folder in city_folders:
    # Go over panos and singles) for this city
    for subfolder in subfolders:
        # Get path to the folder with the images
        subfolder_path = os.path.join(data_dir, city_folder, subfolder)

        # Get a list of the file names for the images in this subfolder
        file_names = os.listdir(subfolder_path)

        # Loop over the images and augment them
        for file_name in file_names:
            # Load the image
            image_path = os.path.join(subfolder_path, file_name)
            image_np = cv2.imread(image_path)
            image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

            # Apply the augmentation pipeline to the image
            augmented_image = transform(image)

            # Save the augmented image to the same subfolder as the original image with aug as name suffix
            new_file_name = f"{os.path.splitext(file_name)[0]}_aug{os.path.splitext(file_name)[1]}"
            new_image_path = os.path.join(subfolder_path, new_file_name)

            # Convert the augmented image to a NumPy array and save it
            cv2.imwrite(new_image_path, (augmented_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
