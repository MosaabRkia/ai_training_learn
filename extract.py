import os
import patoolib
from tqdm import tqdm

# Get current folder full path
current_folder = os.getcwd()

# Function to extract files with progress
def extract_with_progress(archive_path, output_dir):
    try:
        # Open the archive file to get the total number of files
        archive = patoolib.util.Archive(archive_path)
        total_files = len(archive)
        
        # Initialize progress bar
        with tqdm(total=total_files, desc="Extracting", unit="file") as pbar:
            # Custom function to extract and update the progress bar
            def progress_callback(filename, num_files, *args):
                pbar.update(1)

            # Extract the archive and track progress
            patoolib.extract_archive(archive_path, outdir=output_dir, progress_callback=progress_callback)

        print(f"Successfully extracted {archive_path} to {output_dir}")
    except patoolib.util.PatoolError as e:
        print(f"Failed to extract {archive_path}: {e}")

# Extract the "dataset.rar" archive into the current folder
extract_with_progress("dataset.rar", current_folder)
