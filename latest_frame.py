"""Always-latest frame reader (drops backlog while inference is slow)."""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np


class LatestFrameCapture:
    """Background reader that keeps only the newest frame.

    OpenCV/FFmpeg often queues many RTSP frames while YOLO runs; reading
    sequentially then feels several seconds late. This discards stale frames.
    """

    def __init__(self, cap: cv2.VideoCapture):
        self._cap = cap
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._alive = True
        self._stopped = False
        self._thread = threading.Thread(target=self._loop, name="rtsp-grab", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stopped:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                with self._lock:
                    self._alive = False
                time.sleep(0.02)
                continue
            with self._lock:
                self._frame = frame
                self._alive = True

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def is_alive(self) -> bool:
        with self._lock:
            return self._alive

    def release(self) -> None:
        self._stopped = True
        self._thread.join(timeout=2.0)
        self._cap.release()
