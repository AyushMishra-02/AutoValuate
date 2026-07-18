import os
import pandas as pd
import requests

def download_dataset(url, dest_path):
    print(f"Downloading dataset from {url}...")
    response = requests.get(url)
    if response.status_code == 200:
        with open(dest_path, 'wb') as f:
            f.write(response.content)
        print(f"Dataset successfully downloaded to {dest_path}")
    else:
        print(f"Failed to download dataset. Status code: {response.status_code}")

if __name__ == "__main__":
    # URL to a public mirror of the "CAR DETAILS FROM CAR DEKHO.csv" dataset
    DATA_URL = "https://raw.githubusercontent.com/IMvision12/Car-Price_prediction/master/CAR%20DETAILS%20FROM%20CAR%20DEKHO.csv"
    
    # Ensure data directory exists
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    dest_file = os.path.join(data_dir, 'car_data.csv')
    
    download_dataset(DATA_URL, dest_file)
    
    # Verify the download
    if os.path.exists(dest_file):
        df = pd.read_csv(dest_file)
        print(f"Dataset shape: {df.shape}")
        print(df.head())
