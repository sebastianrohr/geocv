import requests
import os
import cv2
import numpy as np
import argparse
import math
import random
import folium

api_key = 'AIzaSyChPNBO4t214jrW1eO1qTd8jlUYTLO3A_8'
parser = argparse.ArgumentParser(description='Geolocation using Google Street View and computer vision.')
parser.add_argument('-l', '--location', type=str, required=True, 
                    help='Location to geolocate')
parser.add_argument('-n', '--number', type=int, required=False, 
                    help='Number of location')
args = parser.parse_args()
location = args.location

def get_coordinates(location) -> dict:
    response = requests.get(f"https://maps.googleapis.com/maps/api/geocode/json?address={location}&key={api_key}")
    response_json = response.json()
    latitude = response_json["results"][0]["geometry"]["location"]["lat"]
    longitude = response_json["results"][0]["geometry"]["location"]["lng"]

    output_dir = os.path.normpath(f'output/{location.lower()}/')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

    bbox_coords = [(bbox['north'], bbox['west']), (bbox['south'], bbox['east'])]

    m = folium.Map(location=(lat, lon), zoom_start=12)
    folium.Rectangle(bbox_coords, color='red', fill_opacity=0.2).add_to(m)
    m.save(os.path.normpath(f'{output_dir}/map.html'))

    return lat, lon, output_dir

def run(lat, lon, output_dir):
    url = 'https://maps.googleapis.com/maps/api/streetview'

    location = f'{lat},{lon}'
    size = '640x640' #i have tried increasing this, but it does not seem to make a difference
    fov = '120'

    file_name = os.path.normpath(f'{output_dir}{location}.jpg')
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
    if args.number is None:
        args.number = 1
    for i in range(args.number):
        print(f'Generating {location.title()} images... ({i+1}/{args.number})')
        lat, lon, output_dir = get_coordinates(location)
        run(lat, lon, output_dir)

# if __name__ == '__main__':
#     get_coordinates(location)