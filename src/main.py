import requests
import os
import cv2
import numpy as np
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Geolocation using Google Street View and computer vision.')
parser.add_argument('-l', '--location', type=str, required=True, 
                    help='Location to geolocate')
args = parser.parse_args()

# Define Google Street View API endpoint
url = 'https://maps.googleapis.com/maps/api/streetview'

# Define parameters
location = args.location
size = '640x640'
fov = '120'
heading = '0'
api_key = 'AIzaSyChPNBO4t214jrW1eO1qTd8jlUYTLO3A_8'

# Define output directory
output_dir = '../output'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Define file name
file_name = os.path.join(output_dir, f'{location}.jpg')

# Define panorama parameters
pano_size = '4096x2048'
pano_fov = '360'
pano_heading = '0'
pano_pitch = '0'

# Define Google Street View API endpoint for panoramic image
pano_url = f'{url}?size={pano_size}&location={location}&fov={pano_fov}&key={api_key}'

# Request panoramic image
pano_response = requests.get(pano_url)

# Load panoramic image
pano_image = np.asarray(bytearray(pano_response.content), dtype=np.uint8)
pano_image = cv2.imdecode(pano_image, cv2.IMREAD_COLOR)
cv2.imwrite(file_name, pano_image)
cv2.imshow('Panoramic Image', pano_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # Define number of rotations (set to 3 for demonstration purposes)
# num_rotations = 3

# # Define rotation angle
# rotation_angle = 360 / num_rotations

# # Rotate and save image for each heading
# pano_image = []

# for i in range(num_rotations):
#     # Define parameters for regular Street View image
#     heading = str(i * rotation_angle)
#     params = {'size': size, 'location': location, 'fov': fov, 'heading': heading, 'key': api_key}

#     # Request image from Google Street View API
#     response = requests.get(url, params=params)

#     # Load image using OpenCV
#     image = np.asarray(bytearray(response.content), dtype=np.uint8)
#     image = cv2.imdecode(image, cv2.IMREAD_COLOR)
#     pano_image.append(image)

# # Stitch images together to form a panoramic image
# pano_stitch = np.concatenate((pano_image), axis=1)

# # Save panoramic image
# cv2.imwrite(file_name, pano_stitch)

# # Display panoramic image
# cv2.imshow('Panoramic Image', pano_stitch)
# cv2.waitKey(0)
# cv2.destroyAllWindows()