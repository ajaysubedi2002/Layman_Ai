def classify_shot(pose_feats, racket_history, frame_h):
    if pose_feats is None:
        return "Unknown"

    wrist_x   = pose_feats["wrist_x_relative"]
    wrist_h   = pose_feats["wrist_height_rel"]
    elbow_ang = pose_feats["elbow_angle"]
    side      = pose_feats["swing_side"]

    racket_height_norm = 0.5
    if len(racket_history) >= 2:
        pts = list(racket_history)
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