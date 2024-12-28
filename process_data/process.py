import pandas as pd
import yaml
from typing import Dict
import re
import logging

# Set up logging
logging.basicConfig(
    filename='data_cleaning.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def clean_text(text: str) -> str:
    """Clean and standardize text data."""
    if not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]*>', '', text)
    # Remove special characters except basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    # Standardize whitespace
    text = ' '.join(text.split())
    return text

def load_yaml_config(yaml_path: str) -> Dict:
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"can't find file: {yaml_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"YAML wrong format: {str(e)}")

config = load_yaml_config("data.yaml")

file_list = [
    # "exl3ia/roleplayerguild--all_char-ic-ooc_part5.csv",
    # "gqilvh/roleplayerguild--all_char-ic-ooc_part8.csv",
    # "luxmdg/roleplayerguild--all_char-ic-ooc_part2.csv",
    # "nhhnj0/roleplayerguild--all_char-ic-ooc_part7.csv",
    # "o4bo59/roleplayerguild--all_char-ic-ooc_part3.csv",
    # "q9mys3/roleplayerguild--all_char-ic-ooc_part6.csv",
    "uuub3g/roleplayerguild--all_char-ic-ooc_part1.csv",
    # "wpceng/roleplayerguild--all_char-ic-ooc_part4.csv"
]

# modify here if root dir of data change
dir_path = config["dir_path"]

for file_name in file_list:
    file_path = f"{dir_path}{file_name}"
    df = pd.read_csv(file_path)
    logging.info(f"Processing file: {file_name}")
    
    # Keep only IC data
    df1 = df[df['thread_type']=="IC"].copy()
    
    # Remove duplicate messages
    initial_count = len(df1)
    df1.drop_duplicates(subset=['message'], inplace=True)
    duplicates_removed = initial_count - len(df1)
    logging.info(f"Removed {duplicates_removed} duplicate messages")
    
    for title, group in df1.groupby('thread_title'):
        safe_title = "".join(x for x in title if x.isalnum() or x in (' ', '-', '_'))[:100]
        if not safe_title.strip():
            logging.warning(f"Skipped empty title: {title}")
            continue
            
        # Clean and process messages
        messages = group['message'].apply(clean_text)
        # Remove empty messages
        messages = messages[messages.str.len() > 0]
        
        if len(messages) == 0:
            logging.warning(f"No valid messages for thread: {safe_title}")
            continue
            
        # save space (remove this limitation in production)
        messages = messages[:3]
        
        with open(f'{dir_path}/../threads/{safe_title}.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(messages))
        
        logging.info(f"Processed thread: {safe_title} with {len(messages)} messages")

print("Data processing finished!")
logging.info("Data processing completed successfully")