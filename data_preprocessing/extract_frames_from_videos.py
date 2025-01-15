import cv2
import os

def extract_frames_from_video(video_path, output_folder, num_frames=30):
    """
    Extract evenly spaced frames from a video and save them as images.

    Args:
        video_path (str): Path to the video file.
        output_folder (str): Folder to save the extracted frames.
        num_frames (int): Number of frames to extract.
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Determine frame indices to extract
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    current_frame = 0
    saved_frames = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame in frame_indices:
            frame_filename = os.path.join(
                output_folder, f"frame_{saved_frames + 1:03d}.jpg"
            )
            cv2.imwrite(frame_filename, frame)
            saved_frames += 1

        current_frame += 1

        if saved_frames == num_frames:
            break

    cap.release()
    print(f"Extracted {saved_frames} frames from {video_path}")


def extract_frames_from_folder(video_folder, output_base_folder, num_frames=30):
    """
    Extract frames from all videos in a folder.

    Args:
        video_folder (str): Path to the folder containing videos.
        output_base_folder (str): Base folder to save extracted frames.
        num_frames (int): Number of frames to extract from each video.
    """
    for filename in os.listdir(video_folder):
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
            video_path = os.path.join(video_folder, filename)
            output_folder = os.path.join(output_base_folder, os.path.splitext(filename)[0])
            extract_frames_from_video(video_path, output_folder, num_frames)


# Paths
video_folder = r"data\Wrong sequence"  # Replace with the path to your video folder
output_base_folder = r"output_data\frames\wrong_frames"  # Replace with the path to save extracted frames

# Extract frames from all videos in the folder
extract_frames_from_folder(video_folder, output_base_folder, num_frames=30)
