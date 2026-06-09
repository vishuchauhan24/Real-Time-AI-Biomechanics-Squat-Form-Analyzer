import cv2
import os
import pickle
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Vector Math Helper
def calculate_joint_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return int(np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0))))

# 2. Load the Upgraded ML Brain
try:
    with open("advanced_squat_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("[SUCCESS] Loaded advanced multi-feature machine learning model.")
except FileNotFoundError:
    print("[ERROR] advanced_squat_model.pkl not found. Please run train_advanced_model.py first.")
    exit()

# 3. Setup MediaPipe Tasks
MODEL_PATH = "pose_landmarker.task"
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.IMAGE)

# 4. Initialize Video Stream with Robust Loading
cap = cv2.VideoCapture(0)

print("Warming up MacBook Air camera sensor...")
time.sleep(2)  # Holds execution briefly to let camera hardware initialize completely

if not cap.isOpened():
    print("[CRITICAL ERROR] Could not open webcam. Check macOS System Settings > Privacy & Security > Camera permissions.")
    exit()

# Full body skeleton connection mapping
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Upper body
    (11, 23), (12, 24), (23, 24),                     # Torso
    (23, 25), (24, 26), (25, 27), (26, 28)            # Legs
]

print("Launching advanced live analyzer... Press 'q' in the video window to quit.")

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        
        # If a frame fails to read temporarily, skip to the next loop instead of breaking
        if not success:
            continue 

        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = landmarker.detect(mp_image)

        prediction = "No Body Detected"
        display_color = (128, 128, 128) # Grey default
        l_knee, r_knee, l_hip, r_hip = 180, 180, 180, 180

        if detection_result.pose_landmarks:
            for landmarks in detection_result.pose_landmarks:
                sh_l, sh_r = landmarks[11], landmarks[12]
                hip_l, hip_r = landmarks[23], landmarks[24]
                knee_l, knee_r = landmarks[25], landmarks[26]
                ank_l, ank_r = landmarks[27], landmarks[28]
                
                # Verify left side visibility to extract features
                if hip_l.visibility > 0.5 and knee_l.visibility > 0.5 and ank_l.visibility > 0.5:
                    l_knee = calculate_joint_angle([hip_l.x*width, hip_l.y*height], [knee_l.x*width, knee_l.y*height], [ank_l.x*width, ank_l.y*height])
                    l_hip = calculate_joint_angle([sh_l.x*width, sh_l.y*height], [hip_l.x*width, hip_l.y*height], [knee_l.x*width, knee_l.y*height])
                
                # Verify right side visibility to extract features
                if hip_r.visibility > 0.5 and knee_r.visibility > 0.5 and ank_r.visibility > 0.5:
                    r_knee = calculate_joint_angle([hip_r.x*width, hip_r.y*height], [knee_r.x*width, knee_r.y*height], [ank_r.x*width, ank_r.y*height])
                    r_hip = calculate_joint_angle([sh_r.x*width, sh_r.y*height], [hip_r.x*width, hip_r.y*height], [knee_r.x*width, knee_r.y*height])

                # Run inference only if tracking is securely locked onto the body core
                if hip_l.visibility > 0.5 and hip_r.visibility > 0.5:
                    # Order matches your advanced_data_collector columns exactly
                    features = [[l_knee, r_knee, l_hip, r_hip]]
                    prediction = model.predict(features)[0]
                    
                    # Core form classification UI mapping
                    display_color = (0, 255, 0) if prediction == "Good" else (0, 0, 255)

                # Draw baseline structural links
                for edge in POSE_CONNECTIONS:
                    pt1, pt2 = landmarks[edge[0]], landmarks[edge[1]]
                    if pt1.visibility > 0.5 and pt2.visibility > 0.5:
                        cv2.line(frame, (int(pt1.x*width), int(pt1.y*height)), (int(pt2.x*width), int(pt2.y*height)), display_color, 2)

                # Draw joint nodes
                for landmark in landmarks:
                    if landmark.visibility > 0.5:
                        cv2.circle(frame, (int(landmark.x*width), int(landmark.y*height)), 4, (255, 255, 255), -1)

        # Draw UI Heads-Up Display (HUD)
        cv2.rectangle(frame, (0, 0), (350, 110), (0, 0, 0), -1)
        cv2.putText(frame, f"ANALYSIS: {prediction}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, display_color, 2)
        cv2.putText(frame, f"KNEES - L: {l_knee} R: {r_knee}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"HIPS  - L: {l_hip} R: {r_hip}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Real-Time Advanced AI Form Classifier', frame)
        
        # Terminate cleanly when 'q' is tapped on screen focus
        if cv2.waitKey(10) & 0xFF == ord('q'): 
            break

cap.release()
cv2.destroyAllWindows()
print("System closed down successfully.")
