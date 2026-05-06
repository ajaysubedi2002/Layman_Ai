import os
import json
import cv2
import subprocess
from ultralytics import YOLO

MODEL_PATH      = "latest.pt"
VIDEO_PATH      = "/home/ajay-subedi/Desktop/object-detections/paddlevedio.mp4"
OUTPUT_VIDEO    = "padel_tracked_output.mp4"
OUTPUT_JSON     = "detections_tracked.json"

PLAYER_CLASS    = "Padel-Players"
CONF_THRESHOLD  = 0.25

PLAYER_COLORS = [
    (0,   255, 255),   # Player 1 — cyan
    (0,   165, 255),   # Player 2 — orange
    (0,   255,   0),   # Player 3 — green
    (255,   0, 255),   # Player 4 — magenta
]
OTHER_COLOR   = (200, 200, 200)
FONT          = cv2.FONT_HERSHEY_SIMPLEX
BOX_THICKNESS = 2



def get_video_writer(output_path: str, fps: float, width: int, height: int):
    """
    Try codecs in priority order.
    H.264 (avc1/X264) gives the most compatible .mp4 output.
    Falls back to MJPG (.avi) as last resort.
    """
    codecs = [
        ("avc1", output_path),           # H.264 — best compatibility
        ("X264", output_path),           # H.264 alternate tag (Linux)
        ("H264", output_path),           # H.264 alternate tag
        ("mp4v", output_path),           # MPEG-4 — wide support
        ("XVID", output_path.replace(".mp4", ".avi")),  # fallback .avi
        ("MJPG", output_path.replace(".mp4", ".avi")),  # last resort
    ]

    for codec, path in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if writer.isOpened():
            print(f"✅ Using codec: {codec}  →  {path}")
            return writer, path
        writer.release()

    raise RuntimeError("❌ No working video codec found on this system.")


def reencode_with_ffmpeg(input_path: str, output_path: str) -> bool:
    """
    Re-encode to H.264 + yuv420p — plays on VLC, Windows, macOS, mobile.
    Returns True if successful.
    """
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",      # required for QuickTime / Windows Media Player
        "-movflags", "+faststart",  # makes .mp4 streamable / seekable
        "-an",                      # no audio track
        output_path
    ]
    try:
        result = subprocess.run(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=600
        )
        if result.returncode != 0:
            print("FFmpeg stderr:", result.stderr.decode()[-500:])
        return result.returncode == 0
    except FileNotFoundError:
        return False
    except subprocess.TimeoutExpired:
        return False



def get_player_label(track_id: int, id_map: dict) -> str:
    if track_id not in id_map:
        id_map[track_id] = f"Player-{len(id_map) + 1}"
    return id_map[track_id]


def draw_box(frame, x1, y1, x2, y2, label, conf, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
    text = f"{label}  {conf:.2f}"
    (tw, th), baseline = cv2.getTextSize(text, FONT, 0.6, 1)
    banner_y = max(y1 - th - baseline - 4, 0)
    cv2.rectangle(frame,
                  (x1, banner_y),
                  (x1 + tw + 6, banner_y + th + baseline + 4),
                  color, -1)
    cv2.putText(frame, text,
                (x1 + 3, banner_y + th + 2),
                FONT, 0.6, (0, 0, 0), 1, cv2.LINE_AA)



def run():
    model = YOLO(MODEL_PATH)

    # ── Video info ───────────────────────────────────────────
    cap    = cv2.VideoCapture(VIDEO_PATH)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"Input : {VIDEO_PATH}")
    print(f"Video : {width}×{height}  {fps:.1f} fps  {total} frames\n")

    # Write frames to a temp AVI first (most reliable intermediate format)
    temp_output = OUTPUT_VIDEO.replace(".mp4", "_temp.avi")
    writer, actual_temp = get_video_writer(temp_output, fps, width, height)

    # ── State ────────────────────────────────────────────────
    player_id_map: dict = {}
    all_frames    = []
    frame_idx     = 0

    # ── Tracking ─────────────────────────────────────────────
    results = model.track(
        source  = VIDEO_PATH,
        conf    = CONF_THRESHOLD,
        stream  = True,
        persist = True,
        tracker = "bytetrack.yaml",
    )

    for r in results:
        frame_idx += 1
        frame = r.orig_img.copy()

        frame_data = {"frame": frame_idx, "detections": []}

        if r.boxes is not None and len(r.boxes):
            track_ids = (
                r.boxes.id.int().tolist()
                if r.boxes.id is not None
                else [None] * len(r.boxes)
            )

            for box, tid in zip(r.boxes, track_ids):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf     = float(box.conf[0])
                cls_id   = int(box.cls[0])
                cls_name = r.names[cls_id]

                if cls_name == PLAYER_CLASS and tid is not None:
                    display_label = get_player_label(tid, player_id_map)
                    idx   = int(display_label.split("-")[1]) - 1
                    color = PLAYER_COLORS[idx % len(PLAYER_COLORS)]
                else:
                    display_label = cls_name
                    color = OTHER_COLOR

                draw_box(frame, x1, y1, x2, y2, display_label, conf, color)

                frame_data["detections"].append({
                    "track_id"    : tid,
                    "player_label": display_label,
                    "class_name"  : cls_name,
                    "confidence"  : round(conf, 4),
                    "bbox"        : {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                })

        cv2.putText(frame, f"Frame {frame_idx}/{total}",
                    (10, height - 10), FONT, 0.5, (255, 255, 255), 1)

        writer.write(frame)
        all_frames.append(frame_data)

        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx}/{total} frames ...")

    writer.release()
    print(f"\n📹  Raw frames written to: {actual_temp}")

    # ── Re-encode with FFmpeg → H.264 mp4 ───────────────────
    print("Re-encoding to H.264 for compatibility...")
    success = reencode_with_ffmpeg(actual_temp, OUTPUT_VIDEO)

    if success:
        os.remove(actual_temp)
        print(f"Final video saved: {OUTPUT_VIDEO}")
        print("    Plays in VLC, Windows Media Player, QuickTime, browsers.")
    else:
        # FFmpeg missing — keep the raw file
        raw_out = OUTPUT_VIDEO.replace(".mp4", "_raw.avi")
        if os.path.exists(actual_temp):
            os.rename(actual_temp, raw_out)
        print(" FFmpeg not found — install it for best compatibility:")
        print("     Ubuntu/Debian : sudo apt install ffmpeg")
        print("     macOS         : brew install ffmpeg")
        print("     Windows       : https://ffmpeg.org/download.html")
        print(f"\n   Raw video saved as: {raw_out}")
        print("   Open it with VLC (VLC plays all codecs).")

    # ── Save JSON ────────────────────────────────────────────
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_frames, f, indent=2)

    print(f"\n JSON  saved : {OUTPUT_JSON}")
    print(f"Players seen: {len(player_id_map)}")
    for tid, label in player_id_map.items():
        print(f"    Track ID {tid:>4}  →  {label}")


if __name__ == "__main__":
    run()