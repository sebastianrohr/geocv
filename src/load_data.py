import os
import random
import pandas as pd
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split

def train_test_set_loader(path="output/", test_size=0.1, val_size=0.1 ,example=False, panos=False):
    # Set the path to your main directory containing the four subdirectories
    main_directory = os.path.normpath(path)

    # Get the subdirectory names
    subdirectories = os.listdir(main_directory)

    # Create empty lists to store image paths and labels
    image_paths = []
    labels = []

    # Loop through each subdirectory
    for subdirectory in subdirectories:
        subdirectory_path = os.path.join(main_directory, subdirectory)
        if panos:
            img_path = os.path.join(subdirectory_path, "panos")
        else:
            img_path = os.path.join(subdirectory_path, "singles")            
        if os.path.isdir(img_path):
            # Get all image file names within the subdirectory
            image_files = os.listdir(img_path)
            # Loop through each image file
            for image_file in image_files:
                image_path = os.path.join(img_path, image_file)
                # Append the image path and its corresponding label (subdirectory name) to the lists
                image_paths.append(image_path)
                labels.append(subdirectory)
    
    # Split the image paths and labels into train, test, and validation sets
    test_val_size = test_size + val_size
    val_size = val_size / test_val_size
    train_image_paths, testval_image_paths, train_labels, testval_labels = train_test_split(image_paths, labels, test_size=test_val_size, random_state=42)
    test_image_paths, val_image_paths, test_labels, val_labels = train_test_split(testval_image_paths, testval_labels, test_size=val_size, random_state=42)

    # Print the number of images in each set
    print("Number of training images:", len(train_image_paths))
    print("Number of test images:", len(test_image_paths))
    print("Number of validation images:", len(val_image_paths))

    # Now you can pass these lists directly to your machine learning model for further processing
    # For example, you can use a deep learning library such as TensorFlow or PyTorch to load and preprocess the images
    # Here's an example using the Pillow library to load and display an image from the training set

    if example:
        # Load and display an example image from the training set
        example_index = 0  # Change this to view different images
        example_image_path = train_image_paths[example_index]
        example_label = train_labels[example_index]
        example_image = Image.open(example_image_path)
        example_image.show()
        print("Image path:", example_image_path)
        print("Label:", example_label)

    # Move the train images to a train directory
    train_directory = "./train/"
    if os.path.exists(train_directory):
        shutil.rmtree(train_directory)
    os.makedirs(train_directory, exist_ok=True)
    for image_path, label in zip(train_image_paths, train_labels):
        label_directory = os.path.join(train_directory, label)
        os.makedirs(label_directory, exist_ok=True)
        image_filename = os.path.basename(image_path)
        shutil.copy(image_path, os.path.join(label_directory, image_filename))

    # Move the test images to a test directory
    test_directory = "./test/"
    if os.path.exists(test_directory):
        shutil.rmtree(test_directory)
    os.makedirs(test_directory, exist_ok=True)
    for image_path, label in zip(test_image_paths, test_labels):
        label_directory = os.path.join(test_directory, label)
        os.makedirs(label_directory, exist_ok=True)
        image_filename = os.path.basename(image_path)
        shutil.copy(image_path, os.path.join(label_directory, image_filename))

    # Move the validation images to a validation directory
    val_directory = "./val/"
    if os.path.exists(val_directory):
        shutil.rmtree(val_directory)
    os.makedirs(val_directory, exist_ok=True)
    for image_path, label in zip(val_image_paths, val_labels):
        label_directory = os.path.join(val_directory, label)
        os.makedirs(label_directory, exist_ok=True)
        image_filename = os.path.basename(image_path)
        shutil.copy(image_path, os.path.join(label_directory, image_filename))

    print("Images have been split into train, test, and validation sets.")
