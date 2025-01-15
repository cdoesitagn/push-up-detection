import numpy as np

def normalize_keypoints(keypoints):
    """
    Normalize keypoints for all frames across all videos.
    
    Args:
        keypoints (np.ndarray): Keypoints array of shape (num_samples, 132).
        num_frames_per_video (int): Number of frames per video.

    Returns:
        np.ndarray: Normalized keypoints of the same shape as input.
    """
    normalized_keypoints = []

    for frame in keypoints:
        # Reshape frame into (33, 4): [x, y, z, visibility]
        reshaped_keypoints = frame.reshape(-1, 4)

        # Extract relevant keypoints for shoulders and hips
        left_shoulder = reshaped_keypoints[11][:3]  # x, y, z of left shoulder
        right_shoulder = reshaped_keypoints[12][:3]  # x, y, z of right shoulder
        left_hip = reshaped_keypoints[23][:3]  # x, y, z of left hip
        right_hip = reshaped_keypoints[24][:3]  # x, y, z of right hip

        # Calculate torso length as the average distance between shoulders and hips
        torso_length = (
            np.linalg.norm(left_shoulder - left_hip) +
            np.linalg.norm(right_shoulder - right_hip)
        ) / 2

        # Avoid division by zero
        if torso_length == 0:
            torso_length = 1

        # Normalize x, y, z values by the torso length
        normalized_sample = reshaped_keypoints.copy()
        normalized_sample[:, :3] /= torso_length  # Normalize x, y, z

        # Flatten back to original shape
        normalized_keypoints.append(normalized_sample.flatten())

    return np.array(normalized_keypoints)

# Load merged keypoints file
merged_keypoints_file = r"output_data\keypoints\merged\wrong_key_merged_raw\wrong_key_merged_raw.npy"
keypoints = np.load(merged_keypoints_file)

# Normalize keypoints
normalized_keypoints = normalize_keypoints(keypoints)

# Save normalized keypoints
normalized_keypoints_file = r"output_data\processed_key\wrong_keypoints_normalized_keypoints.npy"
np.save(normalized_keypoints_file, normalized_keypoints)

print(f"Original keypoints shape: {keypoints.shape}")
print(f"Normalized keypoints saved to: {normalized_keypoints_file}")
print(f"Normalized keypoints shape: {normalized_keypoints.shape}")
