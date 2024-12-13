"""
Script to download and preprocess the DeepFashion dataset.
"""
import os
import shutil
import pandas as pd
from pathlib import Path
import requests
from tqdm import tqdm

def download_file(url, destination):
    """
    Download a file from url to destination with progress bar.
    
    Args:
        url (str): URL to download from
        destination (str): Local path to save the file
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(destination, 'wb') as f, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(block_size):
            size = f.write(data)
            pbar.update(size)

def prepare_dataset_structure(root_dir):
    """
    Create necessary directories for the DeepFashion dataset.
    
    Args:
        root_dir (str): Root directory for the dataset
    """
    directories = [
        'img',
        'anno',
        'eval',
        'splits'
    ]
    
    for directory in directories:
        os.makedirs(os.path.join(root_dir, directory), exist_ok=True)

def main():
    """Main function to download and prepare the DeepFashion dataset."""
    root_dir = Path('data/deepfashion')
    root_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset structure
    prepare_dataset_structure(root_dir)
    
    print("Note: The DeepFashion dataset requires manual download from the official website.")
    print("Please visit: http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html")
    print("Download the Category and Attribute Prediction Benchmark dataset.")
    print(f"\nAfter downloading, please place the files in: {root_dir}")
    print("\nExpected directory structure:")
    print("data/deepfashion/")
    print("├── img/")
    print("├── anno/")
    print("│   ├── list_attr_cloth.txt")
    print("│   ├── list_attr_img.txt")
    print("│   ├── list_category_cloth.txt")
    print("│   ├── list_category_img.txt")
    print("│   └── list_landmarks.txt")
    print("├── eval/")
    print("└── splits/")
    print("    ├── train.txt")
    print("    ├── val.txt")
    print("    └── test.txt")

if __name__ == "__main__":
    main() 