"""Pure-numpy feature helpers shared by the live app, dataset, and evaluation.

Kept free of cv2/mediapipe imports so training and offline evaluation can run
headless (no camera stack initialization just to compute geometry).
"""
import numpy as np


def calculate_angle(a, b, c):
    """Angle in degrees between three points, with b as the vertex."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def normalize_landmarks(points):
    """Centers the skeleton around the hips and scales by torso length."""
    points = np.array(points)
    hip_mid = (points[23] + points[24]) / 2.0
    centered = points - hip_mid

    shoulder_mid = (points[11] + points[12]) / 2.0
    torso_length = np.linalg.norm(shoulder_mid - hip_mid)

    if torso_length > 1e-6:
        normalized = centered / torso_length
    else:
        normalized = centered
    return normalized
