# push-up-detection

**Datasource**:

https://www.kaggle.com/datasets/mohamadashrafsalama/pushup

About Datasets:

100 short videos about push up pose including:

- 50 correct sequence

- 50 wrong sequence

Data Preprocessing:

- Extract frames from the videos

- Extract Keypoints using Mediapipe

- Normalize Keypoint and Make Datasets 

Steps to run: 

1. Run by order: extract_frames_from_videos.py, extract_keypoints_from_frames.py, merge_keypoints.py, normalize_keypoints.py

2. Run model folder, 2 .ipynb files

3. Test with test_lstm.py, test.py
