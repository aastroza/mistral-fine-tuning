import requests
import os
from loguru import logger
import re

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

def extract_routines_ids(folder='transcripts'):
    # Initialize an empty list to store the routine_ids
    routine_ids = []

    # Regular expression pattern to match the routine_id in the filename
    pattern = re.compile(r'routine_(\d+)_transcript\.jsonl')

    # Iterate over the files in the directory
    for filename in os.listdir(folder):
        # Match the filename against the pattern
        match = pattern.match(filename)
        if match:
            # Extract the routine_id and append to the list
            routine_ids.append(int(match.group(1)))
    
    return routine_ids