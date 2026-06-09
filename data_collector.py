import cv2
import os
import urllib.request
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

MODEL_PATH = "pose_landmarker.task"
if not os.path.exists(MODEL_PATH):
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    urllib.request.urlretrieve(url, MODEL_PATH)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.IMAGE)

cap = cv2.VideoCapture(0)
dataset = []
current_state = "Idle"

print("SYSTEM READY. 'g' = Record Good, 'b' = Record Bad, 's' = Stop, 'q' = Save/Exit")

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = landmarker.detect(mp_image)

        # Initialize fallback defaults
        l_knee, r_knee, l_hip, r_hip = 180, 180, 180, 180

        if detection_result.pose_landmarks:
            for landmarks in detection_result.pose_landmarks:
                # Extract Left and Right Lower Body Nodes
                sh_l, sh_r = landmarks[11], landmarks[12]
                hip_l, hip_r = landmarks[23], landmarks[24]
                knee_l, knee_r = landmarks[25], landmarks[26]
                ank_l, ank_r = landmarks[27], landmarks[28]
                
                # Check Visibility for Left Side Features
                if hip_l.visibility > 0.5 and knee_l.visibility > 0.5 and ank_l.visibility > 0.5:
                    l_knee = calculate_joint_angle([hip_l.x*width, hip_l.y*height], [knee_l.x*width, knee_l.y*height], [ank_l.x*width, ank_l.y*height])
                    l_hip = calculate_joint_angle([sh_l.x*width, sh_l.y*height], [hip_l.x*width, hip_l.y*height], [knee_l.x*width, knee_l.y*height])
                
                # Check Visibility for Right Side Features
                if hip_r.visibility > 0.5 and knee_r.visibility > 0.5 and ank_r.visibility > 0.5:
                    r_knee = calculate_joint_angle([hip_r.x*width, hip_r.y*height], [knee_r.x*width, knee_r.y*height], [ank_r.x*width, ank_r.y*height])
                    r_hip = calculate_joint_angle([sh_r.x*width, sh_r.y*height], [hip_r.x*width, hip_r.y*height], [knee_r.x*width, knee_r.y*height])

        # HUD feedback
        status_color = (0, 255, 0) if current_state == "Good" else (0, 0, 255) if current_state == "Bad" else (255, 0, 0)
        cv2.putText(frame, f"MODE: {current_state} (Rows: {len(dataset)})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, f"L_Knee: {l_knee} | R_Knee: {r_knee}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        if current_state != "Idle":
            dataset.append({
                "left_knee": l_knee,
                "right_knee": r_knee,
                "left_hip": l_hip,
                "right_hip": r_hip,
                "label": current_state
            })

        cv2.imshow('Advanced Feature Collector', frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('g'): current_state = "Good"
        elif key == ord('b'): current_state = "Bad"
        elif key == ord('s'): current_state = "Idle"
        elif key == ord('q'): break

cap.release()
cv2.destroyAllWindows()

if len(dataset) > 0:
    df = pd.DataFrame(dataset)
    df.to_csv("advanced_biomechanics_data.csv", index=False)
    print(f"\n[SUCCESS] Generated multi-feature dataset with {len(df)} records!")
