# Padel Shot Detection Prototype

This prototype detects and tracks padel players, rackets, and balls from match videos and classifies shots (Forehand, Backhand, Serve/Smash) using pose features and rule-based logic. It outputs an annotated video plus per-frame JSON detections.

## Overview

Core features:
- YOLOv8 detection + tracking for players, rackets, and ball
- Pose estimation with keypoint-based shot features
- Stable multi-player IDs across tracker ID changes
- Ball trail visualization and shot labels on video
- JSON export with per-frame detections and pose features

## Project Structure
```
|-- config.py                - Global settings and default paths
|-- main.py                  - Main pipeline entrypoint
|-- Padel shot detector .py  - Legacy/alternate script
|-- README.md                - Project documentation
|-- requirements.txt         - Python dependencies
|-- shot_detections.json     - Example output JSON
|-- env/                     - Local Python virtual environment
|-- json/
|   |-- custom_detections.json - Example custom detections
|-- models/
|   |-- __init__.py           
|   |-- best.pt               - Custom detection model weights
|   |-- yolov8n-pose.pt       - Pose model weights
|-- pose/
|   |-- __init__.py           
|   |-- features.py           - Pose feature extraction
|-- shots/
|   |-- __init__.py           
|   |-- classifier.py         - Rule-based shot classifier
|-- tracking/
|   |-- __init__.py          
|   |-- ball_detection.py     - Ball detection helpers
|   |-- drawing.py            - Visualization utilities
|   |-- stable_player_id.py   - Stable player ID assignment
|-- utils/
|   |-- __init__.py          
|   |-- model_loader.py       - Model loading helpers
|-- video/
|   |-- __init__.py           
|   |-- ffmpeg_utils.py       - Video re-encode utilities
|   |-- writer.py             - Video writer setup
```

---
 
## Inference Samples
 
| Sample | Link |
|--------|------|
| Inference Video 1 | [Watch on Drive](https://drive.google.com/file/d/1YU1tdObcEHsx9hq00dAaWjplTaI-eq84/view?usp=sharing) |
| Models | [link to download model form Drive](https://drive.google.com/drive/u/0/folders/1LaZCAYqWHaSErXr8-h2G8t9qR_AkodBS) |
 
---

## Setup

1. Create and activate a Python environment (recommended).
2. Install dependencies:

```bash
pip install ultralytics opencv-python numpy
```

3. Ensure the model weights exist:
- models/best.pt (custom detection)
- models/yolov8n-pose.pt (pose model)

## Usage

### Main pipeline (recommended)

The main pipeline reads settings from config.py. Default paths in config.py are:
- Input video: vedio/paddlevedio.mp4
- Output video: Inference/paddlevedio.mp4
- Output JSON: shot_detections.json

Run:

```bash
python main.py
```

Override input/output paths:

```bash
python main.py \
  --input vedio/paddlevedio.mp4 \
  --output-video Inference/paddlevedio.mp4 \
  --output-json shot_detections.json
```

Arguments:
- --input: input video path
- --output-video: output annotated video path
- --output-json: output JSON path

Example with custom paths:

```bash
python main.py \
  --input vedio/infernce_sample_video.mp4 \
  --output-video Inference/custom_output.mp4 \
  --output-json outputs/custom_detections.json
```

## Configuration

Edit config.py to change:
- DETECTION_MODEL, POSE_MODEL
- VIDEO_PATH, OUTPUT_VIDEO, OUTPUT_JSON
- CONF_THRESHOLD, BALL_CONF_THRESHOLD
- N_PLAYERS, WINDOW_FRAMES, BALL_TRAIL_LEN
- Pose keypoint indices (shoulder, elbow, wrist, hip)

## Outputs

### Annotated video
- Location: Inference/paddlevedio.mp4 (default)
- Overlays: player/racket/ball boxes, ball trail, pose skeletons, shot labels
- Encoding: attempts ffmpeg H.264 re-encode; falls back to raw AVI if ffmpeg is missing

### JSON detections
- Location: shot_detections.json (default)
- Structure (per frame):

```json
{
  "frame": 1,
  "detections": [
    {
      "track_id": 3,
      "player_label": "Player-1",
      "shot": "Forehand",
      "pose_features": {
        "swing_side": "right",
        "wrist_x_relative": 42.1,
        "wrist_height_rel": 18.6,
        "elbow_angle": 121.4,
        "shoulder_angle": -2.3
      },
      "bbox": {"x1": 100, "y1": 50, "x2": 240, "y2": 420}
    },
    {
      "class": "Ball",
      "conf": 0.42,
      "bbox": {"x1": 520, "y1": 310, "x2": 540, "y2": 330}
    }
  ]
}
```

## Approach Explanation

### Methodology

The pipeline follows a detect-track-pose-classify flow that is optimized for padel rallies captured from a fixed camera. First, a custom YOLOv8 detection model identifies players, rackets, and the ball per frame, then a tracker assigns short-term IDs. Because tracker IDs can switch after occlusions, a stable ID layer re-associates players using spatial consistency and movement continuity. In parallel, YOLOv8 pose estimation extracts keypoints for each detected player. These keypoints are converted into compact features (wrist position relative to shoulders, elbow and shoulder angles, swing side) that provide a lightweight representation of the swing geometry.

Shot classification is rule-based on those pose features, with temporal smoothing over a window of frames to reduce flicker. The rules are designed around padel-specific constraints: forehand/backhand separation from swing side and wrist lateral displacement, and serve/smash heuristics from shoulder and wrist height. The output layer overlays detections, poses, ball trail, and shot labels on the video, and stores per-frame JSON that includes both raw detections and the derived pose features for downstream analysis.

### Challenges Faced

- Ball detection is noisy because the ball is small, fast, and frequently blurred; a low confidence threshold helps recall but increases false positives.
- Racket detection is sensitive to motion blur and partial occlusion, which can affect swing-side inference.
- Tracking IDs can swap during crossings or when players are close to the net; the stable ID logic reduces but does not eliminate these switches.
- Pose keypoints are less reliable on wide-angle or low-resolution inputs, which can introduce errors in elbow/shoulder angles.
- Rule-based shot labels struggle with fast volleys, defensive blocks, and ambiguous swing mechanics.

### Improvements To Make

- Train a temporal shot classifier (e.g., 1D CNN/Transformer over pose sequences) to replace heuristic rules.
- Add ball re-identification with trajectory filtering (Kalman + appearance cues) to cut false positives.
- Incorporate court line detection to normalize player positions and improve left/right swing inference.
- Add automated evaluation (precision/recall for detections, per-class F1 for shots) and a labeled validation set.
- Export frame-aligned features for model training (pose + ball trajectory + player velocity).

## Models

- Detection: models/100epoch.pt (custom YOLOv8 trained on padel classes)
- Pose: models/yolov8n-pose.pt (COCO 17-point)
- Backup/older weights: old_backup/

## Limitations

- Rule-based shot classification can mislabel edge cases and fast volleys.
- Ball detection uses a low threshold; false positives are possible.
- Tracking assumes N_PLAYERS players; unusual camera angles or occlusions reduce stability.

