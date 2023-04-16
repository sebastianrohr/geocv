import os
import zipfile
from io import BytesIO
from urllib.request import urlopen

# To get API key, go in kaggle to 'Your Profile' -> 'Account' -> 'Create New API Token'
# then place the file in C:\Users\username\.kaggle

# If error 401 unathorized hapens, redownload a new api key and do the above steps again


# Download the dataset zip file
os.system("kaggle datasets download -d sebastianrohr/geocv-dataset -p {os.getcwd()}")

# Extract the images from the zip file and stream them to your project
for filename in os.listdir(os.getcwd()):
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            for folder in ['copenhagen', 'madrid', 'london', 'moscow']:
                for file in zip_ref.namelist():
                    if file.startswith(folder) and file.endswith(".jpg"):
                        img_data = zip_ref.read(file)
                        # Here, you can use the img_data variable to stream the image data to your machine learning project.
                        # For example, if you're using OpenCV to process the images, you can convert the image data to a NumPy array:
                        # import cv2
                        # img_np = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
