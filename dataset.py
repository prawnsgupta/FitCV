import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from config import DATA_DIR, WINDOW_SIZE, NUM_LANDMARKS, NUM_FEATURES, FEEDBACK_MAP
from features import normalize_landmarks

def calculate_angle(a, b, c):
    """Calculates the angle between three points (b is the vertex)"""
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

class PostureDataset(Dataset):
    def __init__(self, exercise_mode, data_dir=DATA_DIR, window_size=WINDOW_SIZE, augment=False):
        self.exercise_mode = exercise_mode
        self.window_size = window_size
        self.data_dir = os.path.join(data_dir, exercise_mode)
        self.augment = augment
        
        self.samples = []
        self.labels = []
        
        self._load_data()

    def _get_angles(self, frame_landmarks):
        """
        Calculates 10 biomechanical angles from raw landmarks
        frame_landmarks shape: (33, 3)
        """
        # Landmark indices:
        # L_Shoulder: 11, R_Shoulder: 12
        # L_Elbow: 13, R_Elbow: 14
        # L_Wrist: 15, R_Wrist: 16
        # L_Hip: 23, R_Hip: 24
        # L_Knee: 25, R_Knee: 26
        # L_Ankle: 27, R_Ankle: 28
        
        angles = []
        # 1. Left Elbow (Shoulder-Elbow-Wrist)
        angles.append(calculate_angle(frame_landmarks[11], frame_landmarks[13], frame_landmarks[15]))
        # 2. Right Elbow
        angles.append(calculate_angle(frame_landmarks[12], frame_landmarks[14], frame_landmarks[16]))
        # 3. Left Shoulder (Hip-Shoulder-Elbow)
        angles.append(calculate_angle(frame_landmarks[23], frame_landmarks[11], frame_landmarks[13]))
        # 4. Right Shoulder
        angles.append(calculate_angle(frame_landmarks[24], frame_landmarks[12], frame_landmarks[14]))
        # 5. Left Hip (Shoulder-Hip-Knee)
        angles.append(calculate_angle(frame_landmarks[11], frame_landmarks[23], frame_landmarks[25]))
        # 6. Right Hip
        angles.append(calculate_angle(frame_landmarks[12], frame_landmarks[24], frame_landmarks[26]))
        # 7. Left Knee (Hip-Knee-Ankle)
        angles.append(calculate_angle(frame_landmarks[23], frame_landmarks[25], frame_landmarks[27]))
        # 8. Right Knee
        angles.append(calculate_angle(frame_landmarks[24], frame_landmarks[26], frame_landmarks[28]))
        # 9. Left Spine Angle
        vertical_pt_l = frame_landmarks[23].copy(); vertical_pt_l[1] += 0.5
        angles.append(calculate_angle(frame_landmarks[11], frame_landmarks[23], vertical_pt_l))
        # 10. Right Spine Angle
        vertical_pt_r = frame_landmarks[24].copy(); vertical_pt_r[1] += 0.5
        angles.append(calculate_angle(frame_landmarks[12], frame_landmarks[24], vertical_pt_r))
        
        # 11. Left Ankle (Knee-Ankle-Index finger of foot)
        # Landmark 31 is the foot index (toe)
        angles.append(calculate_angle(frame_landmarks[25], frame_landmarks[27], frame_landmarks[31]))
        # 12. Right Ankle
        angles.append(calculate_angle(frame_landmarks[26], frame_landmarks[28], frame_landmarks[32]))
        
        return np.array(angles) / 180.0 # Normalize to 0-1

    def _load_data(self):
        if not os.path.exists(self.data_dir):
            print(f"Warning: Data directory {self.data_dir} does not exist.")
            return

        temp_samples = {}
        clip_counts = {}
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".csv"):
                filepath = os.path.join(self.data_dir, filename)
                df = pd.read_csv(filepath)
                
                if len(df) < self.window_size:
                    continue
                    
                label = int(df['label'].iloc[0])
                # Skip files containing labels not present in the current feedback mapping
                if label not in FEEDBACK_MAP[self.exercise_mode]:
                    continue
                    
                if label not in temp_samples:
                    temp_samples[label] = []
                    clip_counts[label] = 0
                
                clip_counts[label] += 1
                
                feature_cols = []
                for i in range(NUM_LANDMARKS):
                    feature_cols.extend([f'x{i}', f'y{i}', f'z{i}'])
                    
                raw_data = df[feature_cols].values
                raw_data = raw_data.reshape(-1, NUM_LANDMARKS, NUM_FEATURES)
                
                # Combine coordinates with angles
                processed_frames = []
                for frame in raw_data:
                    angles = self._get_angles(frame)
                    normalized_frame = normalize_landmarks(frame)
                    feat = np.concatenate([normalized_frame.flatten(), angles])
                    processed_frames.append(feat)
                
                processed_frames = np.array(processed_frames)
                
                # Create rolling windows
                max_windows_per_file = 5 
                step = max(1, (len(processed_frames) - self.window_size + 1) // max_windows_per_file)
                
                for i in range(0, len(processed_frames) - self.window_size + 1, step):
                    window = processed_frames[i : i + self.window_size]
                    temp_samples[label].append(window)

        if not temp_samples: return
            
        print(f"\n--- Dataset Summary for {self.exercise_mode} ---")
        for label, count in sorted(clip_counts.items()):
            print(f"Label {label}: {count} clips recorded")
            
        min_samples = min(len(s) for s in temp_samples.values())
        print(f"Balancing: Using {min_samples} windows from EACH label.\n")
        
        for label, samples in temp_samples.items():
            np.random.seed(42)
            np.random.shuffle(samples)
            self.samples.extend(samples[:min_samples])
            self.labels.extend([label] * min_samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx].copy()
        label = self.labels[idx]
        
        # Data Augmentation: Add subtle noise to coordinates (first 99 features)
        if self.augment:
            noise = np.random.normal(0, 0.002, size=(self.window_size, 99))
            sample[:, :99] += noise
            
        # Reshape for ST-GCN: (T, N, F)
        # Note: We now have 109 features. We'll handle this in model.py
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

if __name__ == "__main__":
    # Test dataset
    try:
        ds = PostureDataset('Pushup')
        print(f"Loaded {len(ds)} windows for Pushup")
        if len(ds) > 0:
            sample, label = ds[0]
            print(f"Sample shape: {sample.shape}, Label: {label}")
    except Exception as e:
        print(f"Error: {e}")
