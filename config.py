DETECTION_MODEL     = "models/best.pt"
POSE_MODEL          = "models/yolov8n-pose.pt"
VIDEO_PATH          = "vedio/paddlevedio.mp4"
OUTPUT_VIDEO        = "Inference/paddlevedio.mp4"
OUTPUT_JSON         = "shot_detections.json"

PLAYER_CLASS        = "Padel-Players"
RACKET_CLASS        = "Racket"
BALL_CLASS          = "Ball"
BALL_COLOR = (0, 200, 255)


CONF_THRESHOLD      = 0.25
BALL_CONF_THRESHOLD = 0.10

N_PLAYERS           = 4
WINDOW_FRAMES       = 12
BALL_TRAIL_LEN      = 20

L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW,    R_ELBOW    = 7, 8
L_WRIST,    R_WRIST    = 9, 10
L_HIP,      R_HIP      = 11, 12
