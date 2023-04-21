from kaggle.api.kaggle_api_extended import KaggleApi
import os
from find_duplicates import find_duplicates

# set the API key for authentication
api = KaggleApi()
api.authenticate()

# set the dataset name and download path
user = "sebastianrohr"
dataset_name = 'geocv-dataset'
download_url = user + '/' + dataset_name
# get cwd with os
cwd = os.getcwd()
# set download path
download_path = os.path.join(cwd, 'output')

# download the dataset
api.dataset_download_files(download_url, path=download_path, quiet=False, force=True, unzip=True)

print(f"{dataset_name} has been downloaded and unzipped successfully!")

find_duplicates('output/copenhagen/panos', delete=True)
find_duplicates('output/london/panos', delete=True)
find_duplicates('output/madrid/panos', delete=True)
find_duplicates('output/moscow/panos', delete=True)