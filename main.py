import os
import cv2
import json
import numpy as np
from collections import deque

from config import *
from utils.model_loader import load_models
from tracking.stable_player_id import StablePlayerID
from tracking.drawing import draw_box, draw_ball_trail, draw_skeleton, draw_shot_label
from tracking.ball_detection import detect_balls

from pose.features import extract_pose_features, match_pose_to_player
from shots.classifier import classify_shot

from video.writer import get_writer
from video.ffmpeg_utils import reencode, reencode_with_match


def run():    
    det_model, ball_model, pose_model = load_models()
    cap   = cv2.VideoCapture(VIDEO_PATH)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"Video: {W}×{H}  {fps:.1f}fps  {total} frames\n")

    # ── Writer setup (temp file → re-encoded final) ───────────
    os.makedirs(os.path.dirname(OUTPUT_VIDEO) or ".", exist_ok=True)
    in_ext    = os.path.splitext(VIDEO_PATH)[1] or ".mp4"
    base      = os.path.splitext(OUTPUT_VIDEO)[0]
    temp      = f"{base}_temp{in_ext}"
    writer, temp_path = get_writer(temp, fps, W, H)

    # ── ReID tracker ─────────────────────────────────────────
    diag      = np.hypot(W, H)
    reid_dist = max(90, int(0.12 * diag))
    reid = StablePlayerID(
        N_PLAYERS,
        reid_dist=reid_dist,
        memory=None,
        size_ratio_min=0.35,
    )

    # ── Per-player state ──────────────────────────────────────
    player_racket_hist: dict[str, deque] = {}
    player_shot:        dict[str, str]   = {}
    player_shot_timer:  dict[str, int]   = {}

    ball_trail: deque = deque(maxlen=BALL_TRAIL_LEN)

    PLAYER_COLORS = [
        (0, 255, 255),
        (0, 165, 255),
        (0, 255,   0),
        (255,  0, 255),
    ]

    all_data  = []
    frame_idx = 0

    # ── Detection + tracking stream ──────────────────────────
    print("Starting tracking pipeline...")
    det_results = det_model.track(
        source=VIDEO_PATH,
        conf=CONF_THRESHOLD,
        stream=True,
        persist=True,
        tracker="botsort.yaml",
    )

    for r in det_results:
        frame_idx += 1
        frame = r.orig_img.copy()

        # ── Pose estimation ───────────────────────────────────
        pose_results   = pose_model(frame, verbose=False)
        pose_kpts_list = []
        if pose_results and pose_results[0].keypoints is not None:
            kp_data = pose_results[0].keypoints.data
            pose_kpts_list = [
                kp_data[i].cpu().numpy()
                for i in range(kp_data.shape[0])
                if kp_data[i].shape[0] > R_SHOULDER and kp_data[i].shape[1] >= 3
            ]

        # ── Ball detection (unified, conflict-free) ───────────
        ball_boxes = detect_balls(ball_model, frame, r.boxes, r.names)

        for x1, y1, x2, y2, conf in ball_boxes:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            ball_trail.append((cx, cy))
            draw_box(frame, x1, y1, x2, y2, "Ball", conf, BALL_COLOR)

        draw_ball_trail(frame, ball_trail)

        # ── Parse player / racket detections ──────────────────
        active_ids   = set()
        id_centres   = {}
        racket_boxes = []
        frame_data   = {"frame": frame_idx, "detections": []}

        if r.boxes is not None and len(r.boxes):
            track_ids = (
                r.boxes.id.int().tolist()
                if r.boxes.id is not None
                else [None] * len(r.boxes)
            )

            for box, tid in zip(r.boxes, track_ids):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf     = float(box.conf[0])
                cls_name = r.names[int(box.cls[0])]

                # Ball already handled above
                if cls_name == BALL_CLASS:
                    continue

                if cls_name == RACKET_CLASS:
                    racket_boxes.append((x1, y1, x2, y2))
                    draw_box(frame, x1, y1, x2, y2, "Racket", conf, (180, 180, 180))

                elif cls_name == PLAYER_CLASS and tid is not None:
                    active_ids.add(tid)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    id_centres[tid] = (cx, cy)

                    label   = reid.get_label(tid, x1, y1, x2, y2, frame_idx, conf)
                    num = (int(label.split("-")[1]) - 1) % len(PLAYER_COLORS)
                    p_color = PLAYER_COLORS[num]
                    # num     = int(label.split("-")[1]) - 1
                    # p_color = PLAYER_COLORS[num % len(PLAYER_COLORS)]

                    draw_box(frame, x1, y1, x2, y2, label, conf, p_color)

                    # ── Nearest pose to this player ───────────
                    best_kpts = match_pose_to_player(pose_kpts_list, (x1, y1, x2, y2))

                    if best_kpts is not None:
                        draw_skeleton(frame, best_kpts, p_color)

                    pose_feats = extract_pose_features(best_kpts)

                    # ── Update racket history ─────────────────
                    if label not in player_racket_hist:
                        player_racket_hist[label] = deque(maxlen=WINDOW_FRAMES)

                    for rb in racket_boxes:
                        rx, ry = (rb[0] + rb[2]) // 2, (rb[1] + rb[3]) // 2
                        if np.hypot(rx - cx, ry - cy) < 200:
                            player_racket_hist[label].append((rx, ry))

                    # ── Classify shot ─────────────────────────
                    shot = classify_shot(
                        pose_feats, player_racket_hist[label], H
                    )

                    if shot != "Unknown":
                        player_shot[label]       = shot
                        player_shot_timer[label] = int(fps * 1.5)

                    if player_shot_timer.get(label, 0) > 0:
                        player_shot_timer[label] -= 1
                        draw_shot_label(
                            frame, x1, y1,
                            player_shot.get(label, "Unknown"),
                            label,
                        )

                    frame_data["detections"].append({
                        "track_id"     : tid,
                        "player_label" : label,
                        "shot"         : player_shot.get(label, "Unknown"),
                        "pose_features": pose_feats,
                        "bbox"         : {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    })

        # ── Add ball detections to frame JSON ─────────────────
        for x1, y1, x2, y2, conf in ball_boxes:
            frame_data["detections"].append({
                "class": "Ball",
                "conf" : round(conf, 4),
                "bbox" : {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            })

        reid.mark_lost(frame_idx, active_ids, id_centres)

        # ── Frame counter overlay ─────────────────────────────
        cv2.putText(
            frame, f"Frame {frame_idx}/{total}",
            (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

        writer.write(frame)
        all_data.append(frame_data)

        if frame_idx % 50 == 0:
            shots   = {l: player_shot.get(l, "Unknown") for l in reid.active.values()}
            n_balls = len(ball_boxes)
            print(f"  Frame {frame_idx}/{total}  balls={n_balls}  shots={shots}")

    #Finalise video 
    writer.release()
    print("\nRe-encoding to H.264 …")

    if reencode(temp_path, OUTPUT_VIDEO):
        try:
            os.remove(temp_path)
        except Exception:
            pass
        print(f"Video saved: {OUTPUT_VIDEO}")
    elif reencode_with_match(temp_path, OUTPUT_VIDEO, VIDEO_PATH):
        try:
            os.remove(temp_path)
        except Exception:
            pass
        print(f"Video saved: {OUTPUT_VIDEO}")
    else:
        raw = OUTPUT_VIDEO.replace(".mp4", "_raw.avi")
        os.rename(temp_path, raw)
        print(f"FFmpeg unavailable — saved raw video as: {raw}  (open with VLC)")

    print("Saving match data...")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_data, f, indent=2)

    print(f"JSON saved: {OUTPUT_JSON}")
    print(f"Done — {frame_idx} frames processed.")


if __name__ == "__main__":
    run()