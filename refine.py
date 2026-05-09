import cv2
import json
import os
import subprocess
import numpy as np
from collections import deque
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
DETECTION_MODEL     = 'old_backup/100epoch.pt'
POSE_MODEL          = "models/yolov8n-pose.pt"

VIDEO_PATH          = "vedio/infernce_sample_video.mp4"
OUTPUT_VIDEO        = "Inference/inference23.mp4"
OUTPUT_JSON         = "shot_detections.json"

PLAYER_CLASS        = "Padel-Players"
RACKET_CLASS        = "Racket"
BALL_CLASS          = "Ball"
CONF_THRESHOLD      = 0.25
BALL_CONF_THRESHOLD = 0.10

N_PLAYERS           = 4
WINDOW_FRAMES       = 12    # ~0.4s at 30fps
MIN_RACKET_SPEED    = 15    # pixels/frame
BALL_TRAIL_LEN      = 20    # frames to keep ball trail

# ─────────────────────────────────────────────────────────────
# YOLOv8 Pose keypoint indices (COCO 17-point)
# ─────────────────────────────────────────────────────────────
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW,    R_ELBOW    = 7, 8
L_WRIST,    R_WRIST    = 9, 10
L_HIP,      R_HIP      = 11, 12


# ─────────────────────────────────────────────────────────────
# Pose feature extraction
# ─────────────────────────────────────────────────────────────

def angle_between(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))


def extract_pose_features(kpts):
    if kpts is None or kpts.shape[0] < 17:
        return None

    def kp(idx):   return kpts[idx, :2]
    def conf(idx): return float(kpts[idx, 2])

    required = [L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST, L_ELBOW, R_ELBOW]
    if any(conf(i) < 0.3 for i in required):
        return None

    ls, rs = kp(L_SHOULDER), kp(R_SHOULDER)
    le, re = kp(L_ELBOW),    kp(R_ELBOW)
    lw, rw = kp(L_WRIST),    kp(R_WRIST)
    lh, rh = kp(L_HIP),      kp(R_HIP)

    if lw[1] < rw[1]:
        swing_wrist, swing_elbow, swing_shoulder, side = lw, le, ls, "left"
    else:
        swing_wrist, swing_elbow, swing_shoulder, side = rw, re, rs, "right"

    body_centre_x    = (lh[0] + rh[0]) / 2
    wrist_x_relative = swing_wrist[0] - body_centre_x
    shoulder_y       = (ls[1] + rs[1]) / 2
    wrist_height_rel = shoulder_y - swing_wrist[1]
    elbow_angle      = angle_between(swing_shoulder, swing_elbow, swing_wrist)
    shoulder_angle   = float(np.degrees(np.arctan2(rs[1]-ls[1], rs[0]-ls[0])))

    return {
        "swing_side"       : side,
        "wrist_x_relative" : float(wrist_x_relative),
        "wrist_height_rel" : float(wrist_height_rel),
        "elbow_angle"      : elbow_angle,
        "shoulder_angle"   : shoulder_angle,
    }


# ─────────────────────────────────────────────────────────────
# Rule-based shot classifier
# ─────────────────────────────────────────────────────────────

def classify_shot(pose_feats, racket_history, frame_h):
    if pose_feats is None:
        return "Unknown"

    wrist_x   = pose_feats["wrist_x_relative"]
    wrist_h   = pose_feats["wrist_height_rel"]
    elbow_ang = pose_feats["elbow_angle"]
    side      = pose_feats["swing_side"]

    racket_height_norm = 0.5
    if len(racket_history) >= 2:
        pts   = list(racket_history)
        racket_height_norm = pts[-1][1] / frame_h

    if racket_height_norm < 0.35 and wrist_h > 20:
        return "Serve/Smash"

    if side == "right":
        is_forehand = wrist_x > 0 and elbow_ang > 100
    else:
        is_forehand = wrist_x < 0 and elbow_ang > 100

    if is_forehand:
        return "Forehand"
    if elbow_ang < 145:
        return "Backhand"

    return "Unknown"


# ─────────────────────────────────────────────────────────────
# Stable player ID tracker
# ─────────────────────────────────────────────────────────────

class StablePlayerID:
    def __init__(self, n=None, reid_dist=140, memory=None,
                 size_ratio_min=0.45, velocity_alpha=0.7, debug=False):
        self.n              = n
        self.reid_dist      = float(reid_dist)
        self.memory         = memory
        self.size_ratio_min = float(size_ratio_min)
        self.velocity_alpha = float(velocity_alpha)
        self.debug          = debug
        self.active:   dict[int, str]  = {}
        self.profiles: dict[str, dict] = {}
        self.lost:     dict[str, dict] = {}
        self._next                     = 1
        self._frame_cursor             = -1
        self._assigned_this_frame: set[str] = set()

    def _dist(self, a, b):
        return float(np.hypot(a[0]-b[0], a[1]-b[1]))

    def _size_ratio(self, s1, s2):
        w1,h1 = max(float(s1[0]),1.0), max(float(s1[1]),1.0)
        w2,h2 = max(float(s2[0]),1.0), max(float(s2[1]),1.0)
        return min(min(w1,w2)/max(w1,w2), min(h1,h2)/max(h1,h2))

    def _predict_center(self, profile, frame):
        cx, cy = profile["c"]
        vx, vy = profile.get("v", (0.0, 0.0))
        dt = min(max(0, int(frame - profile.get("f", frame))), 30)
        return (cx + vx*dt, cy + vy*dt)

    def _touch_profile(self, label, center, size, frame, conf):
        if label not in self.profiles:
            self.profiles[label] = {
                "c": (float(center[0]), float(center[1])),
                "s": (float(size[0]),   float(size[1])),
                "f": int(frame), "v": (0.0, 0.0), "conf": float(conf),
            }
            return
        p = self.profiles[label]
        prev_cx, prev_cy = p["c"]
        dt = max(1, int(frame - p.get("f", frame)))
        raw_vx = (float(center[0]) - prev_cx) / dt
        raw_vy = (float(center[1]) - prev_cy) / dt
        prev_vx, prev_vy = p.get("v", (0.0, 0.0))
        vx = self.velocity_alpha*prev_vx + (1-self.velocity_alpha)*raw_vx
        vy = self.velocity_alpha*prev_vy + (1-self.velocity_alpha)*raw_vy
        conf_w = min(max(float(conf), 0.0), 1.0)
        pos_w  = 0.35 + 0.65*conf_w
        p["c"] = (float(prev_cx*(1-pos_w) + float(center[0])*pos_w),
                  float(prev_cy*(1-pos_w) + float(center[1])*pos_w))
        p["s"] = (float(0.8*p["s"][0] + 0.2*float(size[0])),
                  float(0.8*p["s"][1] + 0.2*float(size[1])))
        p["v"] = (float(vx), float(vy))
        p["f"] = int(frame)
        p["conf"] = float(0.8*p.get("conf", conf_w) + 0.2*conf_w)

    def _match_existing_label(self, center, size, frame):
        best_label, best_score = None, float("inf")
        for label, profile in self.profiles.items():
            if label in self._assigned_this_frame:
                continue
            pred_c     = self._predict_center(profile, frame)
            dist       = self._dist(center, pred_c)
            size_ratio = self._size_ratio(size, profile["s"])
            if size_ratio < self.size_ratio_min:
                continue
            gap  = max(0, int(frame - int(profile.get("f", frame))))
            gate = self.reid_dist * (1.0 + min(gap/45.0, 1.5))
            if dist > gate:
                continue
            score = dist + 0.35*(1.0-size_ratio)*self.reid_dist
            if score < best_score:
                best_score, best_label = score, label
        return best_label

    def get_label(self, tid, x1, y1, x2, y2, frame, conf=1.0):
        if frame != self._frame_cursor:
            self._frame_cursor = int(frame)
            self._assigned_this_frame.clear()
        cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
        size   = (max(1.0, x2-x1), max(1.0, y2-y1))
        if tid in self.active:
            label = self.active[tid]
            if label in self.profiles:
                pred_c = self._predict_center(self.profiles[label], frame)
                dist   = self._dist((cx,cy), pred_c)
                if dist <= self.reid_dist*2.2 and label not in self._assigned_this_frame:
                    self._assigned_this_frame.add(label)
                    self._touch_profile(label, (cx,cy), size, frame, conf)
                    return label
            del self.active[tid]
        label = self._match_existing_label((cx,cy), size, frame)
        if label is None:
            label = f"Player-{self._next}"
            self._next += 1
        self.active[tid] = label
        self._assigned_this_frame.add(label)
        self._touch_profile(label, (cx,cy), size, frame, conf)
        if self.debug:
            print(f"[ID] frame={frame} tid={tid} -> {label}")
        return label

    def mark_lost(self, frame, active_ids, id_centres):
        for tid in list(set(self.active) - active_ids):
            label = self.active.pop(tid)
            if label in self.profiles and tid in id_centres:
                c = id_centres[tid]
                self.profiles[label]["c"] = (float(c[0]), float(c[1]))
                self.profiles[label]["f"] = int(frame)
        active_labels = set(self.active.values())
        self.lost = {l:p for l,p in self.profiles.items() if l not in active_labels}
        if self.memory is not None:
            stale = [l for l,p in self.profiles.items()
                     if frame - int(p.get("f", frame)) > int(self.memory)]
            for l in stale:
                self.profiles.pop(l, None)
                self.lost.pop(l, None)


# ─────────────────────────────────────────────────────────────
# Video writer  (mp4v first — most reliable on Linux)
# ─────────────────────────────────────────────────────────────

def get_writer(path, fps, w, h):
    for codec, out in [
        ("mp4v", path),
        ("avc1", path),
        ("XVID", path.replace(".mp4", ".avi")),
    ]:
        fw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*codec), fps, (w, h))
        if fw.isOpened():
            print(f"Codec: {codec} → {out}")
            return fw, out
        fw.release()
    raise RuntimeError("No working video codec found.")


def reencode(src, dst):
    cmd = ["ffmpeg", "-y", "-i", src,
           "-c:v", "libx264", "-preset", "fast",
           "-crf", "23", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", "-an", dst]
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=600)
        return r.returncode == 0
    except Exception:
        return False


def probe_input_codec(path):
    try:
        p = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name",
            "-of", "default=noprint_wrappers=1:nokey=1", path,
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
        vcodec = p.stdout.strip().splitlines()[0] if p.stdout else None
    except Exception:
        vcodec = None
    try:
        q = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=codec_name",
            "-of", "default=noprint_wrappers=1:nokey=1", path,
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
        has_audio = bool(q.stdout.strip())
    except Exception:
        has_audio = False
    return vcodec, has_audio


def codec_to_encoder(vcodec):
    if not vcodec:
        return "libx264"
    return {
        "h264": "libx264", "h265": "libx265", "hevc": "libx265",
        "mpeg4": "mpeg4",  "vp9": "libvpx-vp9", "vp8": "libvpx",
        "av1": "libaom-av1",
    }.get(vcodec.lower(), "libx264")


def reencode_with_match(src, dst, input_video_path):
    vcodec, has_audio = probe_input_codec(input_video_path)
    encoder = codec_to_encoder(vcodec)
    cmd = ["ffmpeg", "-y", "-i", src, "-i", input_video_path,
           "-map", "0:v", "-map", "1:a?",
           "-c:v", encoder, "-preset", "fast", "-crf", "23",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    cmd += ["-c:a", "copy"] if has_audio else ["-an"]
    cmd += [dst]
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=1200)
        return r.returncode == 0
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────

SHOT_COLORS = {
    "Forehand"   : (0,   255, 150),
    "Backhand"   : (255, 165,   0),
    "Serve/Smash": (0,   140, 255),
    "Unknown"    : (180, 180, 180),
}
BALL_COLOR = (0, 200, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_box(frame, x1, y1, x2, y2, label, conf, color):
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    text = f"{label}  {conf:.2f}"
    (tw, th), bl = cv2.getTextSize(text, FONT, 0.55, 1)
    by = max(y1-th-bl-4, 0)
    cv2.rectangle(frame, (x1,by), (x1+tw+6, by+th+bl+4), color, -1)
    cv2.putText(frame, text, (x1+3, by+th+2), FONT, 0.55, (0,0,0), 1, cv2.LINE_AA)


def draw_shot_label(frame, x1, y1, shot, player_label):
    if shot == "Unknown":
        return
    color = SHOT_COLORS.get(shot, (255, 255, 255))
    text  = f">> {shot}"
    cv2.putText(frame, text, (x1, y1-25), FONT, 0.5, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x1, y1-25), FONT, 0.5, color,   1, cv2.LINE_AA)


def draw_skeleton(frame, kpts, color=(0, 255, 200)):
    if kpts is None:
        return
    pairs = [
        (L_SHOULDER, R_SHOULDER), (L_SHOULDER, L_ELBOW), (R_SHOULDER, R_ELBOW),
        (L_ELBOW, L_WRIST),       (R_ELBOW,    R_WRIST),
        (L_SHOULDER, L_HIP),      (R_SHOULDER,  R_HIP),
    ]
    for a, b in pairs:
        if kpts[a,2] > 0.3 and kpts[b,2] > 0.3:
            cv2.line(frame,
                     (int(kpts[a,0]), int(kpts[a,1])),
                     (int(kpts[b,0]), int(kpts[b,1])),
                     color, 2, cv2.LINE_AA)
    for i in [L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST]:
        if kpts[i,2] > 0.3:
            cv2.circle(frame, (int(kpts[i,0]), int(kpts[i,1])), 5, color, -1)


def draw_ball_trail(frame, trail):
    """Draw a fading trail of past ball positions."""
    pts = list(trail)
    for i in range(1, len(pts)):
        alpha  = i / len(pts)
        radius = max(2, int(6 * alpha))
        color  = (int(BALL_COLOR[0]*alpha),
                  int(BALL_COLOR[1]*alpha),
                  int(BALL_COLOR[2]*alpha))
        cv2.circle(frame, pts[i], radius, color, -1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────
# Unified ball detector
# KEY FIX: uses a SEPARATE model instance (ball_model) so that
# calling .predict() never conflicts with the ongoing .track()
# stream running on det_model.
# ─────────────────────────────────────────────────────────────

def detect_balls(ball_model, frame, tracker_boxes, tracker_names):
    """
    Returns list of (x1, y1, x2, y2, conf) for every ball found.

    Two sources are merged:
      1. Tracker output  — fast but may drop the ball at low conf.
      2. Dedicated predict pass on ball_model (separate YOLO instance)
         at low BALL_CONF_THRESHOLD, filtering by class *name* so the
         class index never matters.
    """
    balls = []

    # ── Source 1: tracker ────────────────────────────────────
    if tracker_boxes is not None and len(tracker_boxes):
        for box in tracker_boxes:
            cls_name = tracker_names[int(box.cls[0])]
            if cls_name == BALL_CLASS:
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                balls.append((x1, y1, x2, y2, float(box.conf[0])))

    # ── Source 2: dedicated predict (no class filter) ────────
    # Using ball_model (separate YOLO instance) avoids the hang
    # caused by calling .predict() on the same object mid-.track().
    pred = ball_model.predict(frame, conf=BALL_CONF_THRESHOLD, verbose=False)
    if pred and pred[0].boxes is not None:
        for box in pred[0].boxes:
            cls_name = tracker_names[int(box.cls[0])]   # filter by name
            if cls_name != BALL_CLASS:
                continue
            conf = float(box.conf[0])
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())

            # Skip duplicates already found by the tracker
            duplicate = any(
                abs(x1-bx1) < 20 and abs(y1-by1) < 20
                for bx1,by1,_,_,_ in balls
            )
            if not duplicate:
                balls.append((x1, y1, x2, y2, conf))

    return balls


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def run():
    # ── Load models ──────────────────────────────────────────
    # det_model  → used for .track() — stateful, never call .predict() on it
    # ball_model → separate instance of the same weights for .predict() only
    # pose_model → yolov8n-pose (auto-downloads on first run)
    det_model  = YOLO(DETECTION_MODEL)
    ball_model = YOLO(DETECTION_MODEL)   # ← separate instance: the critical fix
    pose_model = YOLO(POSE_MODEL)

    # Verify class names so BALL_CLASS string can be confirmed
    print("Model classes:", det_model.names)

    cap   = cv2.VideoCapture(VIDEO_PATH)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"Video: {W}×{H}  {fps:.1f}fps  {total} frames\n")

    os.makedirs(os.path.dirname(OUTPUT_VIDEO) or ".", exist_ok=True)
    in_ext     = os.path.splitext(VIDEO_PATH)[1] or ".mp4"
    base       = os.path.splitext(OUTPUT_VIDEO)[0]
    temp       = f"{base}_temp{in_ext}"
    writer, temp_path = get_writer(temp, fps, W, H)

    diag      = np.hypot(W, H)
    reid_dist = max(90, int(0.12 * diag))
    reid = StablePlayerID(
        N_PLAYERS,
        reid_dist=reid_dist,
        memory=None,
        size_ratio_min=0.42,
    )

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
            cx, cy = (x1+x2)//2, (y1+y2)//2
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
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    id_centres[tid] = (cx, cy)

                    label   = reid.get_label(tid, x1, y1, x2, y2, frame_idx, conf)
                    num     = int(label.split("-")[1]) - 1
                    p_color = PLAYER_COLORS[num % len(PLAYER_COLORS)]

                    draw_box(frame, x1, y1, x2, y2, label, conf, p_color)

                    # ── Nearest pose to this player ───────────
                    best_kpts, best_dist = None, float("inf")
                    for kpts in pose_kpts_list:
                        if (kpts.ndim != 2 or kpts.shape[0] <= R_SHOULDER
                                or kpts.shape[1] < 3):
                            continue
                        if kpts[L_SHOULDER,2] > 0.2 and kpts[R_SHOULDER,2] > 0.2:
                            pcx = (kpts[L_SHOULDER,0] + kpts[R_SHOULDER,0]) / 2
                            pcy = (kpts[L_SHOULDER,1] + kpts[R_SHOULDER,1]) / 2
                            d   = np.hypot(pcx-cx, pcy-cy)
                            if d < best_dist:
                                best_dist, best_kpts = d, kpts

                    if best_kpts is not None and best_dist < 150:
                        draw_skeleton(frame, best_kpts, p_color)

                    pose_feats = extract_pose_features(best_kpts)

                    # ── Update racket history ─────────────────
                    if label not in player_racket_hist:
                        player_racket_hist[label] = deque(maxlen=WINDOW_FRAMES)

                    for rb in racket_boxes:
                        rx, ry = (rb[0]+rb[2])//2, (rb[1]+rb[3])//2
                        if np.hypot(rx-cx, ry-cy) < 200:
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
                        "bbox"         : {"x1":x1,"y1":y1,"x2":x2,"y2":y2},
                    })

        # Add ball detections to JSON
        for x1, y1, x2, y2, conf in ball_boxes:
            frame_data["detections"].append({
                "class": "Ball",
                "conf" : round(conf, 4),
                "bbox" : {"x1":x1,"y1":y1,"x2":x2,"y2":y2},
            })

        reid.mark_lost(frame_idx, active_ids, id_centres)

        # Frame counter overlay
        cv2.putText(
            frame, f"Frame {frame_idx}/{total}",
            (10, H-10), FONT, 0.5, (255,255,255), 1,
        )

        writer.write(frame)
        all_data.append(frame_data)

        if frame_idx % 100 == 0:
            shots   = {l: player_shot.get(l,"Unknown") for l in reid.active.values()}
            n_balls = len(ball_boxes)
            print(f"  Frame {frame_idx}/{total}  balls={n_balls}  shots={shots}")

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

    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_data, f, indent=2)

    print(f"JSON saved: {OUTPUT_JSON}")
    print(f"Done — {frame_idx} frames processed.")


if __name__ == "__main__":
    run()