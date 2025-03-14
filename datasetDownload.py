import os
import gdown

# Function to download file from Google Drive
def download_file(file_id, output_filename):
    url = f'https://drive.google.com/uc?id={file_id}'
    
    # Check if file already exists to avoid redownloading
    if os.path.exists(output_filename):
        print(f"File '{output_filename}' already exists. Skipping download.")
        return
    
    print(f"Downloading from: {url}")
    print(f"To: {output_filename}")
    
    try:
        gdown.download(url, output_filename, quiet=False)
        print("Download completed.")
    except Exception as e:
        print(f"Error downloading file {output_filename}: {e}")

# File IDs and output filenames
file_ids = ["1PUur-pJ1_gfQSr7_S8816_FIzB99gjBG"]  # Replace with your file IDs
output_filenames = ["dataset.rar"]  # Output filenames for downloaded files

# Download files
for file_id, output_filename in zip(file_ids, output_filenames):
    print(f"\nDownloading file with ID: {file_id}")
    download_file(file_id, output_filename)
