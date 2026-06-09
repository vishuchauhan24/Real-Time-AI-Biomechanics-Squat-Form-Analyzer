import cv2
import os
import glob
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def calculate_joint_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return int(np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0))))

# Setup MediaPipe Model
MODEL_PATH = "pose_landmarker.task"
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.IMAGE)

dataset = []

# Define directories to scan
folders = {"Good": "videos/good/", "Bad": "videos/bad/"}
VALID_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".MOV")

print("Starting automated data ingestion pipeline...")

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    for label, folder_path in folders.items():
        # Get all files in the directory
        all_files = os.listdir(folder_path)
        # Filter files to only include valid video formats
        video_files = [os.path.join(folder_path, f) for f in all_files if f.endswith(VALID_EXTENSIONS)]
        
        print(f"\nProcessing folder [{label}] — Found {len(video_files)} valid videos.")
        
        for video_path in video_files:
            print(f" -> Parsing: {os.path.basename(video_path)}")
            cap = cv2.VideoCapture(video_path)            
            while cap.isOpened():
                success, frame = cap.read()
                if not success: break # Video ended, move to next file

                height, width, _ = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = landmarker.detect(mp_image)

                if detection_result.pose_landmarks:
                    for landmarks in detection_result.pose_landmarks:
                        sh_l, sh_r = landmarks[11], landmarks[12]
                        hip_l, hip_r = landmarks[23], landmarks[24]
                        knee_l, knee_r = landmarks[25], landmarks[26]
                        ank_l, ank_r = landmarks[27], landmarks[28]
                        
                        l_knee, r_knee, l_hip, r_hip = None, None, None, None

                        # Extract Left Side
                        if hip_l.visibility > 0.5 and knee_l.visibility > 0.5 and ank_l.visibility > 0.5:
                            l_knee = calculate_joint_angle([hip_l.x*width, hip_l.y*height], [knee_l.x*width, knee_l.y*height], [ank_l.x*width, ank_l.y*height])
                            l_hip = calculate_joint_angle([sh_l.x*width, sh_l.y*height], [hip_l.x*width, hip_l.y*height], [knee_l.x*width, knee_l.y*height])
                        
                        # Extract Right Side
                        if hip_r.visibility > 0.5 and knee_r.visibility > 0.5 and ank_r.visibility > 0.5:
                            r_knee = calculate_joint_angle([hip_r.x*width, hip_r.y*height], [knee_r.x*width, knee_r.y*height], [ank_r.x*width, ank_r.y*height])
                            r_hip = calculate_joint_angle([sh_r.x*width, sh_r.y*height], [hip_r.x*width, hip_r.y*height], [knee_r.x*width, knee_r.y*height])

                        # Only add to dataset if features were reliably captured
                        if l_knee and r_knee:
                            dataset.append({
                                "left_knee": l_knee,
                                "right_knee": r_knee,
                                "left_hip": l_hip,
                                "right_hip": r_hip,
                                "label": label
                            })
            cap.release()

# Save compiled tabular data
if len(dataset) > 0:
    df = pd.DataFrame(dataset)
    df.to_csv("advanced_biomechanics_data.csv", index=False)
    print(f"\n[PIPELINE COMPLETE] Generated dataset with {len(df)} frames saved to advanced_biomechanics_data.csv!")
else:
    print("\n[ERROR] No video data was successfully parsed. Check folder paths.")
