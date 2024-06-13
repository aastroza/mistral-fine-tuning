import requests
import os
from loguru import logger

def download_file(url: str, local_filename: str):
    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(local_filename), exist_ok=True)

    # Download the file and save it locally
    response = requests.get(url)
    if response.status_code == 200:
        with open(local_filename, 'wb') as file:
            file.write(response.content)
        logger.info(f"File downloaded successfully and saved to {local_filename}")
    else:
        logger.error(f"Failed to download the file. HTTP Status code: {response.status_code}")
