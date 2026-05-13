import io
import tempfile
import os

import cv2
import numpy as np

MAX_EXTRACTED_FRAMES = 20


def get_video_info(video_bytes: bytes) -> dict:
    tmp = _write_temp(video_bytes)
    try:
        cap = cv2.VideoCapture(tmp)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return {
            "fps": fps,
            "total_frames": total,
            "duration_sec": round(total / fps, 2) if fps > 0 else 0,
        }
    finally:
        _cleanup(tmp)


def extract_frames(
    video_bytes: bytes,
    frame_interval: int = 30,
) -> list[tuple[bytes, str]]:
    tmp = _write_temp(video_bytes)
    try:
        cap = cv2.VideoCapture(tmp)
        if not cap.isOpened():
            raise ValueError("无法打开视频文件")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, frame_interval)

        frames: list[tuple[bytes, str]] = []
        idx = 0
        while idx < total and len(frames) < MAX_EXTRACTED_FRAMES:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                break
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            name = f"frame_{len(frames):04d}.jpg"
            frames.append((buf.tobytes(), name))
            idx += interval

        cap.release()
        return frames
    finally:
        _cleanup(tmp)


def _write_temp(data: bytes) -> str:
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.write(fd, data)
    os.close(fd)
    return path


def _cleanup(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass
