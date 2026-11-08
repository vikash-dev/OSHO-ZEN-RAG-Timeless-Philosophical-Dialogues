import os
from zipfile import ZipFile

def unzip_files_indexed(target_folder):
    # Ensure the directory path exists
    if not os.path.exists(target_folder):
        print(f"Error: The directory '{target_folder}' does not exist.")
        return

    # Find and sort all .zip files in the directory
    zip_files = [f for f in os.listdir(target_folder) if f.lower().endswith('.zip')]
    zip_files.sort()

    if not zip_files:
        print("No .zip files found in the specified directory.")
        return

    print(f"Found {len(zip_files)} zip file(s). Starting extraction...")

    # Iterate through each zip file with a 1-based index counter
    for i, file_name in enumerate(zip_files, start=1):
        zip_path = os.path.join(target_folder, file_name)
        
        # Define the target output folder name (e.g., index_1, index_2)
        output_folder_name = f"index_{i}"
        output_dir = os.path.join(target_folder, output_folder_name)
        
        # Create the new folder if it doesn't already exist
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Open and extract the zip archive
            with ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            print(f"Successfully extracted '{file_name}' to '{output_folder_name}'")
        except Exception as e:
            print(f"Failed to extract '{file_name}': {e}")

if __name__ == "__main__":
    # Replace this with the path to your folder containing the zip files
    # Use '.' for the current directory where the script is located
    folder_to_process = "./all_index" 
    
    unzip_files_indexed(folder_to_process)
