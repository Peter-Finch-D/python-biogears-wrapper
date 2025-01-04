# digital_twin/utils.py

import re
from datetime import datetime

def print_with_timestamp(message):
    """
    Prints a message with the current timestamp.
    """
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def sanitize_filename(label):
    """
    Sanitize the label to create a safe filename.
    Replaces spaces with underscores, '/' with '_per_', and removes other special characters.
    """
    label = label.lower().replace(" ", "_").replace("/", "_per_")
    label = re.sub(r'[^\w\-_.]', '', label)  # Remove any character that is not alphanumeric, '-', '_', or '.'
    return label
