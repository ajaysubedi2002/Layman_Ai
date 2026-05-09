import cv2
from config import *

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


    if tracker_boxes is not None and len(tracker_boxes):
        for box in tracker_boxes:
            cls_name = tracker_names[int(box.cls[0])]
            if cls_name == BALL_CLASS:
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                balls.append((x1, y1, x2, y2, float(box.conf[0])))
 
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