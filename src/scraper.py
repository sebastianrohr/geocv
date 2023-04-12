import requests
import os
import cv2
import numpy as np
import argparse
import math
import random

api_key = 'AIzaSyChPNBO4t214jrW1eO1qTd8jlUYTLO3A_8'
parser = argparse.ArgumentParser(description='Geolocation using Google Street View and computer vision.')
parser.add_argument('-l', '--location', type=str, required=True, 
                    help='Location to geolocate')
parser.add_argument('-n', '--number', type=int, required=True, 
                    help='Number of location')
args = parser.parse_args()
location = args.location

def get_coordinates(location) -> dict:
    response = requests.get(f"https://maps.googleapis.com/maps/api/geocode/json?address={location}&key={api_key}")
    response_json = response.json()
    latitude = response_json["results"][0]["geometry"]["location"]["lat"]
    longitude = response_json["results"][0]["geometry"]["location"]["lng"]

    # Use the Haversine formula to calculate the latitude and longitude ranges that correspond to a 10-kilometer radius
    lat_range = 0.00904371733 * 10
    lng_range = 0.01096640438 * 10 * math.cos(latitude * math.pi / 180)

    bbox = {
        "north": latitude + lat_range,
        "south": latitude - lat_range,
        "east": longitude + lng_range,
        "west": longitude - lng_range
    }

    lat = random.uniform(bbox['south'], bbox['north'])
    lon = random.uniform(bbox['west'], bbox['east'])

    print(f'Latitude: {lat}, Longitude: {lon}')
    return lat, lon

def run(lat, lon, location_name):
    url = 'https://maps.googleapis.com/maps/api/streetview'

    location = f'{lat},{lon}'
    size = '640x640' #i have tried increasing this, but it does not seem to make a difference
    fov = '120'

    output_dir = f'../output/{location_name.lower()}'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    file_name = os.path.join(output_dir, f'{location}.jpg')
    pano_size = '4096x2048'
    pano_fov = '360'

    pano_url = f'{url}?size={pano_size}&location={location}&fov={pano_fov}&key={api_key}'
    pano_response = requests.get(pano_url)

    pano_image = np.asarray(bytearray(pano_response.content), dtype=np.uint8)
    pano_image = cv2.imdecode(pano_image, cv2.IMREAD_COLOR)

    num_rotations = 3
    rotation_angle = 360 / num_rotations

    pano_image = []
    for i in range(num_rotations):
        heading = str(i * rotation_angle)
        params = {'size': size, 'location': location, 'fov': fov, 'heading': heading, 'key': api_key}

        response = requests.get(url, params=params)

        image = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        pano_image.append(image)

    pano_stitch = np.concatenate((pano_image), axis=1)

    cv2.imwrite(file_name, pano_stitch)

if __name__ == '__main__':
    for i in range(args.number):
        lat, lon = get_coordinates(location)
        run(lat, lon, location)