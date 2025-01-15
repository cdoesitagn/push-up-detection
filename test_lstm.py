import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# Load the trained LSTM model
lstm_model = load_model("model\model.h5")  # Replace with your model path

# Initialize Mediapipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def extract_keypoints_from_frame(frame):
    """
    Extract keypoints and bounding box from a single frame using Mediapipe.
    Args:
        frame (np.ndarray): A frame captured from the webcam.
    Returns:
        tuple: Flattened keypoints (132,) and bounding box (xmin, ymin, xmax, ymax).
    """
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        keypoints = []
        landmarks = results.pose_landmarks.landmark

        # Calculate bounding box
        xs = [landmark.x for landmark in landmarks]
        ys = [landmark.y for landmark in landmarks]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        # Scale bounding box to image dimensions
        h, w, _ = frame.shape
        bbox = (int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h))

        # Extract keypoints
        for landmark in landmarks:
            keypoints.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        return np.array(keypoints), bbox, results
    else:
        return None, None, None  # No keypoints detected

def predict_with_webcam():
    """
    Use the webcam to capture frames, extract keypoints, and predict using the LSTM model.
    """
    cap = cv2.VideoCapture(0)  # Open webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    frame_buffer = []  # Buffer to store 30 frames
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Extract keypoints and bounding box from the frame
        keypoints, bbox, results = extract_keypoints_from_frame(frame)
        if keypoints is not None and keypoints.shape == (132,):
            frame_buffer.append(keypoints)  # Add keypoints to buffer

            # If buffer has 30 frames, predict
            if len(frame_buffer) == 30:
                input_data = np.array(frame_buffer)  # Shape: (30, 132)
                input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension: (1, 30, 132)
                prediction = lstm_model.predict(input_data, verbose=0)
                label = "Correct Push-Up" if prediction[0][0] > 0.5 else "Wrong Push-Up"

                # Draw bounding box
                if bbox:
                    (xmin, ymin, xmax, ymax) = bbox
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                # Draw pose landmarks
                if results:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Display prediction label
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Clear buffer after prediction
                frame_buffer = []
        else:
            label = "No Pose Detected"

        # Show the frame
        cv2.imshow("Push-Up Detection", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the webcam prediction
predict_with_webcam()

# Close Mediapipe resources
pose.close()
