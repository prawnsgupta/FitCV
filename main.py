import cv2
import mediapipe as mp
import numpy as np
import torch
import os
from collections import deque
from config import DEVICE, WINDOW_SIZE, NUM_LANDMARKS, NUM_FEATURES, FEEDBACK_MAP, ERROR_CONNECTIONS, CAMERA_ID
from model import PostureModel
from rep_counter import RepCounter
from features import calculate_angle, normalize_landmarks

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def get_angles(frame_landmarks):
    angles = []
    # Elbows, Shoulders, Hips, Knees
    angles.append(calculate_angle(frame_landmarks[11], frame_landmarks[13], frame_landmarks[15])) # L_Elbow
    angles.append(calculate_angle(frame_landmarks[12], frame_landmarks[14], frame_landmarks[16])) # R_Elbow
    angles.append(calculate_angle(frame_landmarks[23], frame_landmarks[11], frame_landmarks[13])) # L_Shoulder
    angles.append(calculate_angle(frame_landmarks[24], frame_landmarks[12], frame_landmarks[14])) # R_Shoulder
    angles.append(calculate_angle(frame_landmarks[11], frame_landmarks[23], frame_landmarks[25])) # L_Hip
    angles.append(calculate_angle(frame_landmarks[12], frame_landmarks[24], frame_landmarks[26])) # R_Hip
    angles.append(calculate_angle(frame_landmarks[23], frame_landmarks[25], frame_landmarks[27])) # L_Knee
    angles.append(calculate_angle(frame_landmarks[24], frame_landmarks[26], frame_landmarks[28])) # R_Knee
    # Spine verticality
    v_l = frame_landmarks[23].copy(); v_l[1] += 0.5
    angles.append(calculate_angle(frame_landmarks[11], frame_landmarks[23], v_l))
    v_r = frame_landmarks[24].copy(); v_r[1] += 0.5
    angles.append(calculate_angle(frame_landmarks[12], frame_landmarks[24], v_r))
    # Ankles (Knee-Ankle-Toe)
    angles.append(calculate_angle(frame_landmarks[25], frame_landmarks[27], frame_landmarks[31])) # L_Ankle
    angles.append(calculate_angle(frame_landmarks[26], frame_landmarks[28], frame_landmarks[32])) # R_Ankle
    return np.array(angles) / 180.0

def extract_features(results):
    if not results.pose_landmarks:
        return None
    
    points = []
    for lm in results.pose_landmarks.landmark:
        points.append([lm.x, lm.y, lm.z])
    
    points = np.array(points)
    normalized_points = normalize_landmarks(points)
    angles = get_angles(points)
    
    return np.concatenate([normalized_points.flatten(), angles])

def draw_custom_landmarks(image, pose_landmarks, current_mode, label_idx):
    if not pose_landmarks:
        return
        
    h, w, c = image.shape
    
    # Identify error connections
    error_bones = []
    if current_mode and label_idx > 0 and current_mode in ERROR_CONNECTIONS:
        if label_idx in ERROR_CONNECTIONS[current_mode]:
            error_bones = ERROR_CONNECTIONS[current_mode][label_idx]
            
    # Draw connections (bones)
    for connection in mp_pose.POSE_CONNECTIONS:
        start_idx = connection[0]
        end_idx = connection[1]
        
        start_lm = pose_landmarks.landmark[start_idx]
        end_lm = pose_landmarks.landmark[end_idx]
        
        # Check visibility
        if start_lm.visibility < 0.5 or end_lm.visibility < 0.5:
            continue
            
        start_point = (int(start_lm.x * w), int(start_lm.y * h))
        end_point = (int(end_lm.x * w), int(end_lm.y * h))
        
        # Default color is green. If error, turn red.
        color = (0, 255, 0) # Green in BGR
        thickness = 2
        
        is_error = False
        for (u, v) in error_bones:
            if (start_idx == u and end_idx == v) or (start_idx == v and end_idx == u):
                is_error = True
                break
                
        if is_error:
            color = (0, 0, 255) # Red in BGR
            thickness = 4
            
        cv2.line(image, start_point, end_point, color, thickness)
        
    # Draw landmarks (joints)
    for lm in pose_landmarks.landmark:
        if lm.visibility < 0.5:
            continue
        point = (int(lm.x * w), int(lm.y * h))
        cv2.circle(image, point, 4, (255, 255, 255), -1)

def load_model(exercise_mode):
    save_path = os.path.join("weights", f"{exercise_mode}_model.pth")
    if not os.path.exists(save_path):
        print(f"No trained model found for {exercise_mode}. Please train it first.")
        return None
        
    num_classes = len(FEEDBACK_MAP[exercise_mode])
    model = PostureModel(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    model.eval()
    return model

def main():
    cap = cv2.VideoCapture(CAMERA_ID)
    
    current_mode = None
    model = None
    window = deque(maxlen=WINDOW_SIZE)
    # Smoothing buffer for predictions
    prediction_buffer = deque(maxlen=5)
    current_feedback = "Waiting for data..."
    current_label_idx = 0
    rep_counter = RepCounter()
    
    print("--- Real-time Inference Started ---")
    print("Press 'P' for Pushup mode, 'S' for Squat mode, 'B' for Bicep Curl, 'L' for Lateral Raise, 'R' for Bent-Over Row.")
    print("Press 'Q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        results = pose.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            if current_mode and model:
                features = extract_features(results)
                if features is not None:
                    window.append(features)
                    
                    if current_mode == 'Squat':
                        current_feedback, current_label_idx = rep_counter.update_squat_heuristic(results.pose_landmarks)
                    elif len(window) == WINDOW_SIZE:
                        # Prepare input tensor
                        input_data = np.array(window).reshape(1, WINDOW_SIZE, -1)
                        input_tensor = torch.tensor(input_data, dtype=torch.float32).to(DEVICE)
                        
                        # Run inference
                        with torch.no_grad():
                            output = model(input_tensor)
                            # Convert to probabilities
                            probs = torch.softmax(output, dim=1)
                            conf, predicted = torch.max(probs, 1)
                            
                            # CONFIDENCE THRESHOLD
                            pred_idx = predicted.item()
                            if pred_idx != 0 and conf.item() < 0.8:
                                pred_idx = 0
                                
                            prediction_buffer.append(pred_idx)
                            
                        # Smooth prediction via majority vote
                        current_label_idx = max(set(prediction_buffer), key=list(prediction_buffer).count)
                        current_feedback = FEEDBACK_MAP[current_mode][current_label_idx]
            
            # Draw the colored skeleton
            draw_custom_landmarks(image, results.pose_landmarks, current_mode, current_label_idx)
                            
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            current_mode = 'Pushup'
            model = load_model(current_mode)
            window.clear()
            prediction_buffer.clear()
            current_feedback = "Waiting for data..."
            current_label_idx = 0
        elif key == ord('s'):
            current_mode = 'Squat'
            model = load_model(current_mode)
            window.clear()
            prediction_buffer.clear()
            current_feedback = "Waiting for data..."
            current_label_idx = 0
        elif key == ord('b'):
            current_mode = 'BicepCurl'
            model = load_model(current_mode)
            window.clear()
            prediction_buffer.clear()
            current_feedback = "Waiting for data..."
            current_label_idx = 0
        elif key == ord('l'):
            current_mode = 'LateralRaise'
            model = load_model(current_mode)
            window.clear()
            prediction_buffer.clear()
            current_feedback = "Waiting for data..."
            current_label_idx = 0
        elif key == ord('r'):
            current_mode = 'BentOverRow'
            model = load_model(current_mode)
            window.clear()
            prediction_buffer.clear()
            current_feedback = "Waiting for data..."
            current_label_idx = 0

        # Display info
        mode_text = f"Mode: {current_mode if current_mode else 'None (P:Push, S:Squat, B:Curl, L:Lateral, R:Row)'}"
        cv2.putText(image, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        if current_mode:
            # Color feedback text: Green for correct, Red for incorrect
            color = (0, 255, 0) if current_feedback == "Correct" else (0, 0, 255)
            cv2.putText(image, f"Feedback: {current_feedback}", (10, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        cv2.imshow('Real-time Posture Correction', image)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
