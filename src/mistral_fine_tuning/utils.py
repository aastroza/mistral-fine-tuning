import requests
import os
from loguru import logger
import re
import pandas as pd
import json

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

def read_fine_tuning_file(file_path: str) -> pd.DataFrame:

    # Function to decode double-encoded Unicode sequences in the text field
    def decode_unicode_escapes(text):
        try:
            # Correcting the text encoded incorrectly
            corrected_text = text.encode('latin1').decode('utf-8')
            return corrected_text
        except UnicodeDecodeError:
            return text

    # Function to handle the keywords field
    def process_keywords(keywords_str):
        try:
            # Convert the string representation of the list to an actual list
            keywords_list = eval(keywords_str)
            # Convert list to a string with single quotes
            return str(keywords_list).replace('"', "'")
        except Exception as e:
            logger.error(f"Error processing keywords: {e}")
            return keywords_str

    # Initialize an empty list to store processed JSON objects
    processed_json_list = []

    # Read and process the file line by line
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Parse the JSON object
            json_obj = json.loads(line)
            
            # Decode the text field
            json_obj['text'] = decode_unicode_escapes(json_obj['text'])
            
            # Process the keywords field
            json_obj['keywords'] = process_keywords(json_obj['keywords'])
            
            # Append the processed JSON object to the list
            processed_json_list.append(json_obj)

    # Load the list of JSON objects into a pandas DataFrame
    df = pd.json_normalize(processed_json_list)

    return df