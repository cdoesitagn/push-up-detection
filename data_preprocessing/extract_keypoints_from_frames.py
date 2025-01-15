import os
import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def extract_keypoints_from_image(image_path):
    """
    Extract keypoints from a single image using Mediapipe Pose.
    Args:
        image_path (str): Path to the image file.
    Returns:
        np.ndarray: Flattened array of keypoints [x, y, z, visibility] or zeros if no keypoints detected.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return np.zeros(33 * 4)  # 33 keypoints, each with x, y, z, visibility

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        return np.array(keypoints)
    else:
        return np.zeros(33 * 4)  # Return zeros if no landmarks are detected

def process_subfolders(base_folder, output_folder):
    """
    Process all images in subfolders and extract keypoints using Mediapipe.
    Args:
        base_folder (str): Base folder containing subfolders of images.
        output_folder (str): Folder to save the keypoints .npy files for each subfolder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(base_folder):
        for subdir in dirs:
            subfolder_path = os.path.join(root, subdir)
            keypoints_list = []
            for file in os.listdir(subfolder_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(subfolder_path, file)
                    keypoints = extract_keypoints_from_image(image_path)
                    keypoints_list.append(keypoints)

            # Save keypoints for the subfolder
            if keypoints_list:
                keypoints_array = np.array(keypoints_list)
                output_file = os.path.join(output_folder, f"{subdir}_keypoints.npy")
                np.save(output_file, keypoints_array)
                print(f"Saved keypoints for {subdir}: {output_file}")

# Input and output paths
base_folder = r"output_data\frames\wrong_frames"  # Replace with the folder containing images in subfolders
output_folder = r"output_data\keypoints\pre-processed\wrong_key_raw"        # Replace with the folder to save extracted keypoints

# Run the processing
process_subfolders(base_folder, output_folder)

# Close Mediapipe Pose
pose.close()
