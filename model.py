import os
import json
import cv2
from ultralytics import YOLO

# Define the path to your custom trained model
model_path = 'best (1).pt'

# Load a custom trained model
model = YOLO(model_path)

# Process results and convert to JSON
# Define the path to your video file
video_path = 'input_sample_video.mp4'

# Use streaming inference to avoid accumulating all results in RAM
# stream=True yields a generator of Results objects
results = model(source=video_path, conf=0.25, stream=True, save=True)

# Save results incrementally to JSON to avoid large memory usage
output_json = 'detections.json'
frame_count = 0
first = True

try:
    with open(output_json, 'w') as f:
        f.write('[\n')
        for r in results:
            frame_count += 1
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

print(f"\nJSON file saved as: {output_json}")
print(f"Total frames processed: {frame_count}")
print(f"  Class IDs: {r.boxes.cls}")            # Class IDs