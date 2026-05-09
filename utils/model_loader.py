from ultralytics import YOLO
from config import DETECTION_MODEL, POSE_MODEL

def load_models():
    det_model  = YOLO(DETECTION_MODEL)
    ball_model = YOLO(DETECTION_MODEL)
    pose_model = YOLO(POSE_MODEL)
    return det_model, ball_model, pose_model