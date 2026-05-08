import subprocess


def reencode(src, dst):
    """
    Re-encode a video to H.264 using ffmpeg.
    Produces widely compatible MP4 output.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", src,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        dst,
    ]
    try:
        r = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=600,
        )
        return r.returncode == 0
    except Exception:
        return False


def probe_input_codec(path):
    """
    Detect video codec and whether audio stream exists
    using ffprobe.
    """
    try:
        p = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
        vcodec = p.stdout.strip().splitlines()[0] if p.stdout else None
    except Exception:
        vcodec = None

    try:
        q = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=codec_name",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
        has_audio = bool(q.stdout.strip())
    except Exception:
        has_audio = False

    return vcodec, has_audio


def codec_to_encoder(vcodec):
    """
    Map input codec to a matching ffmpeg encoder.
    """
    if not vcodec:
        return "libx264"

    return {
        "h264": "libx264",
        "h265": "libx265",
        "hevc": "libx265",
        "mpeg4": "mpeg4",
        "vp9": "libvpx-vp9",
        "vp8": "libvpx",
        "av1": "libaom-av1",
    }.get(vcodec.lower(), "libx264")


def reencode_with_match(src, dst, input_video_path):
    """
    Re-encode output video using an encoder that matches
    the original input video codec. Also preserves audio
    if present.
    """
    vcodec, has_audio = probe_input_codec(input_video_path)
    encoder = codec_to_encoder(vcodec)

    cmd = [
        "ffmpeg", "-y",
        "-i", src,
        "-i", input_video_path,
        "-map", "0:v",
        "-map", "1:a?",
        "-c:v", encoder,
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
    ]

    if has_audio:
        cmd += ["-c:a", "copy"]
    else:
        cmd += ["-an"]

    cmd += [dst]

    try:
        r = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=1200,
        )
        return r.returncode == 0
    except Exception:
        return False