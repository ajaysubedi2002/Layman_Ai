import os
import json
import cv2
from ultralytics import YOLO
from config import *

# Define the path to your custom trained model
# model_path = 'old_backup/100epoch.pt'

# Load a custom trained model
model = YOLO(DETECTION_MODEL)

# Process results and convert to JSON
# Define the path to your video file
video_path = 'vedio/infernce_sample_video.mp4'

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

output_video = 'detections.mp4'
video_writer = cv2.VideoWriter(
    output_video,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height),
)

# Use streaming inference to avoid accumulating all results in RAM
# stream=True yields a generator of Results objects
results = model(source=video_path, conf=0.25, stream=True, save=False, show=False)

# Save results incrementally to JSON to avoid large memory usage
output_json = 'detections.json'
frame_count = 0
first = True

try:
    with open(output_json, 'w') as f:
        f.write('[\n')
        for r in results:
            frame_count += 1
            annotated_frame = r.plot()
            video_writer.write(annotated_frame)
            frame_detections = {
                "frame": frame_count,
                "num_detections": len(r.boxes),
                "detections": []
            }

            # Extract detection information for each box
            if len(r.boxes) > 0:
                for i in range(len(r.boxes)):
                    bbox = r.boxes.xyxy[i]
                    detection = {
                        "id": i + 1,
                        "bbox": {
                            "x1": float(bbox[0]),
                            "y1": float(bbox[1]),
                            "x2": float(bbox[2]),
                            "y2": float(bbox[3])
                        },
                        "confidence": float(r.boxes.conf[i]),
                        "class_id": int(r.boxes.cls[i]),
                        "class_name": r.names[int(r.boxes.cls[i])]
                    }
                    frame_detections["detections"].append(detection)

            # Write comma between objects
            if not first:
                f.write(',\n')
            json.dump(frame_detections, f, indent=2)
            first = False

        f.write('\n]\n')
except KeyboardInterrupt:
    # Ensure JSON is well-formed if user interrupts
    if not first:
        with open(output_json, 'a') as f:
            f.write('\n]\n')
    print('\nInterrupted by user. JSON file may be partial but closed.')

video_writer.release()

print(f"\nJSON file saved as: {output_json}")
print(f"Video file saved as: {output_video}")
print(f"Total frames processed: {frame_count}")
print(f"  Class IDs: {r.boxes.cls}")            # Class IDs