import cv2

def get_writer(path, fps, w, h):
    fw = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not fw.isOpened():
        raise RuntimeError(f"Cannot open video writer at: {path}")
    return fw, path  # ← return both the writer AND the path