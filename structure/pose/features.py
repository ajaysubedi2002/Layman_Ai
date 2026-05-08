from utils.geometry import angle_between
import numpy as np

L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW,    R_ELBOW    = 7, 8
L_WRIST,    R_WRIST    = 9, 10
L_HIP,      R_HIP      = 11, 12

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
    lw, rw = kp(L_WRIST),   kp(R_WRIST)
    lh, rh = kp(L_HIP),     kp(R_HIP)

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


def match_pose_to_player(kpts_list, bbox):
    """
    Match pose keypoints to a player bbox using nose/center keypoint.
    """
    x1, y1, x2, y2 = bbox
    for kpts in kpts_list:
        px = int(kpts[0][0])  # nose x
        py = int(kpts[0][1])  # nose y
        if x1 < px < x2 and y1 < py < y2:
            return kpts
    return None