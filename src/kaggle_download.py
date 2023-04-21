from kaggle.api.kaggle_api_extended import KaggleApi

# set the API key for authentication
api = KaggleApi()
api.authenticate()

# set the dataset name and download path
user = "sebastianrohr"
dataset_name = 'geocv-dataset'
download_url = user + '/' + dataset_name
download_path = 'output'

# download the dataset
api.dataset_download_files(download_url, path=download_path, quiet=False, force=True, unzip=True)

print(f"{dataset_name} has been downloaded and unzipped successfully!")