import os
import numpy as np

def merge_keypoints_files(input_folder, output_file):
    """
    Merge all keypoints .npy files in a folder into a single .npy file sequentially.

    Args:
        input_folder (str): Folder containing the .npy files.
        output_file (str): Path to save the merged .npy file.
    """
    merged_data = []

    # Traverse through all .npy files in the folder
    for file in sorted(os.listdir(input_folder)):  # Sort to maintain order
        if file.endswith(".npy"):
            file_path = os.path.join(input_folder, file)
            keypoints = np.load(file_path)
            merged_data.append(keypoints)

            print(f"Merged: {file_path}, Shape: {keypoints.shape}")

    # Concatenate all arrays
    merged_data = np.concatenate(merged_data, axis=0)

    # Save the merged array
    np.save(output_file, merged_data)
    print(f"\nAll keypoints merged into: {output_file}, Final Shape: {merged_data.shape}")

# Input and output paths
input_folder = r"output_data\keypoints\pre-processed\wrong_key_raw"  # Replace with the folder containing .npy files
output_file = r"output_data\keypoints\merged\wrong_key_merged_raw"       # Path to save the merged keypoints

# Run the merging function
merge_keypoints_files(input_folder, output_file)
