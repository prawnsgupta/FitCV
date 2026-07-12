from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import cv2
import mediapipe as mp
import numpy as np
import torch
import os
from collections import deque
from config import DEVICE, WINDOW_SIZE, FEEDBACK_MAP, ERROR_CONNECTIONS, CAMERA_ID
from main import extract_features, draw_custom_landmarks, load_model
from rep_counter import RepCounter

app = FastAPI()

# Create static dir if it doesn't exist
if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

class AppState:
    def __init__(self):
        self.current_mode = None
        self.model = None
        self.window = deque(maxlen=WINDOW_SIZE)
        self.prediction_buffer = deque(maxlen=5)
        self.current_feedback = "Select an exercise to begin."
        self.current_label_idx = 0
        self.rep_counter = RepCounter()
        self.is_active = False

state = AppState()
camera = cv2.VideoCapture(CAMERA_ID)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return FileResponse("static/index.html")

@app.post("/set_mode/{mode}")
async def set_mode(mode: str):
    if mode in FEEDBACK_MAP:
        state.current_mode = mode
        state.model = load_model(mode)
        state.window.clear()
        state.prediction_buffer.clear()
        state.current_feedback = "Ready to start."
        state.current_label_idx = 0
        state.rep_counter.reset(mode)
        state.is_active = False
        return {"status": "success", "mode": mode}
    return {"status": "error", "message": "Invalid mode"}

@app.get("/state")
async def get_state():
    stats = state.rep_counter.get_stats()
    return {
        "mode": state.current_mode,
        "feedback": state.current_feedback,
        "correct_reps": stats["correct_reps"],
        "incorrect_reps": stats["incorrect_reps"],
        "error_frequencies": stats.get("error_frequencies", {}),
        "is_active": state.is_active
    }

@app.post("/start")
async def start_session():
    if state.current_mode:
        state.is_active = True
        state.current_feedback = "Waiting for data..."
    return {"status": "started"}

@app.post("/stop")
async def stop_session():
    state.is_active = False
    state.current_feedback = "Paused."
    return {"status": "stopped"}

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            results = pose.process(image)
            
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                if state.current_mode and state.model:
                    if not state.is_active:
                        state.current_label_idx = 0
                        state.window.clear()
                    else:
                        features = extract_features(results)
                        if features is not None:
                            state.window.append(features)
                            
                            if state.current_mode == 'Squat':
                                feedback, label_idx = state.rep_counter.update_squat_heuristic(results.pose_landmarks)
                                state.current_feedback = feedback
                                state.current_label_idx = label_idx
                            elif len(state.window) == WINDOW_SIZE:
                                input_data = np.array(state.window).reshape(1, WINDOW_SIZE, -1)
                                input_tensor = torch.tensor(input_data, dtype=torch.float32).to(DEVICE)
                                
                                with torch.no_grad():
                                    output = state.model(input_tensor)
                                    probs = torch.softmax(output, dim=1)
                                    conf, predicted = torch.max(probs, 1)
                                    
                                    pred_idx = predicted.item()
                                    if pred_idx != 0 and conf.item() < 0.8:
                                        pred_idx = 0
                                        
                                    state.prediction_buffer.append(pred_idx)
                                    
                                state.current_label_idx = max(set(state.prediction_buffer), key=list(state.prediction_buffer).count)
                                state.current_feedback = FEEDBACK_MAP[state.current_mode][state.current_label_idx]

                                # Update rep counter
                                state.rep_counter.update(results.pose_landmarks, state.current_mode, state.current_label_idx)

                draw_custom_landmarks(image, results.pose_landmarks, state.current_mode, state.current_label_idx)

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
