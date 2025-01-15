import cv2
import numpy as np
from joblib import load
import mediapipe as mp

# Load the trained SVM model
svm_model = load("knn_model.pkl")  # Replace with your SVM model file path

# Initialize Mediapipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def extract_keypoints_from_frame(frame):
    """
    Extract keypoints and Mediapipe results from a single frame using Mediapipe.
    Args:
        frame (np.ndarray): A frame captured from the webcam.
    Returns:
        tuple: Flattened keypoints (132,), bounding box (xmin, ymin, xmax, ymax), and results object.
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
    Use the webcam to capture frames, extract keypoints, and predict using the SVM model.
    """
    cap = cv2.VideoCapture(0)  # Open webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Extract keypoints, bounding box, and results from the frame
        keypoints, bbox, results = extract_keypoints_from_frame(frame)
        if keypoints is not None and keypoints.shape == (132,):
            # Reshape and predict with the SVM model
            keypoints = keypoints.reshape(1, -1)  # Reshape to (1, 132)
            prediction = svm_model.predict(keypoints)
            label = "Correct Push-Up" if prediction[0] == 1 else "Wrong Push-Up"

            # Draw bounding box
            (xmin, ymin, xmax, ymax) = bbox
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Draw pose landmarks
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        else:
            label = "No Pose Detected"

        # Display prediction label
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
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
