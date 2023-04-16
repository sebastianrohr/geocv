# Not working yet, not sure if its possible to do

import os
import shutil
import glob

madrid_folder = 'output\madrid'
copenhagen_folder = 'output\copenhagen'
london_folder = 'output\london'
moscow_folder = 'output\moscow'


# Loop through each folder and upload any new images to the Kaggle dataset
for folder_path in [madrid_folder, copenhagen_folder,london_folder, moscow_folder]:
    # Find all .jpg files in the folder
    jpg_files = glob.glob(os.path.join(folder_path, '*.jpg'))
    
    # Define the path to the Kaggle dataset folder
    dataset_folder = os.path.join(os.getcwd(), 'geocv-dataset')
    kaggle_folder = os.path.join(dataset_folder, os.path.basename(folder_path))

    
    # Loop through each image and copy it to the Kaggle dataset folder
    for jpg_file in jpg_files:
        # Define the path to the image in the Kaggle dataset folder
        kaggle_file = os.path.join(kaggle_folder, os.path.basename(jpg_file))
        
        # Copy the image to the Kaggle dataset folder if it doesn't already exist
        if not os.path.exists(kaggle_file):
            shutil.copy(jpg_file, kaggle_file)
    
    # Create a new version of the Kaggle dataset with the updated images
    os.system(f'kaggle datasets version -p {dataset_folder} -m "Update {os.path.basename(folder_path)} folder"')
