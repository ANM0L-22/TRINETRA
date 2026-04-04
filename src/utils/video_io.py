"""
Trinetra — Video I/O utilities
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Tuple, Optional
from loguru import logger


class VideoReader:
    """Iterator-based video reader."""

    def __init__(self, source):
        """
        Args:
            source: File path (str/Path) or camera index (int).
        """
        self.source = source
        self.cap = cv2.VideoCapture(str(source) if not isinstance(source, int) else source)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video source: {source}")

        self.fps    = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total  = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_id = 0

    def __iter__(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield self.frame_id, frame
            self.frame_id += 1

    def __len__(self):
        return self.total

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()


class VideoWriter:
    """Wrapper around cv2.VideoWriter."""

    def __init__(self, output_path: str, fps: float, width: int, height: int):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # type: ignore
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        self.path = output_path

    def write(self, frame: np.ndarray):
        self.writer.write(frame)

    def release(self):
        self.writer.release()
        logger.success(f"Video saved: {self.path}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()


def resize_frame(frame: np.ndarray, max_width: int = 1280) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (max_width, int(h * scale)))


def frame_to_jpg_bytes(frame: np.ndarray, quality: int = 85) -> bytes:
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()
