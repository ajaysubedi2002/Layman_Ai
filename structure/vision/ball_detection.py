from config import *


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