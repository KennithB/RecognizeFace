import threading
import time
from typing import Optional, Union
import cv2
import numpy as np


class CameraStream:
    """
    Threaded camera stream reader for USB webcams and IP / RTSP network streams.
    Decouples frame acquisition from ML inference to prevent stream buffer latency.
    """

    def __init__(self, source: Union[int, str] = 0):
        self.source = source
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.running: bool = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self.last_read_time = 0.0

    def start(self) -> bool:
        """Opens camera and begins background capture thread."""
        self.stop()

        # Parse source (convert numeric strings like "0", "1" to integers for USB)
        parsed_source = self.source
        if isinstance(self.source, str) and self.source.strip().isdigit():
            parsed_source = int(self.source.strip())

        # For RTSP, disable buffering where supported
        if isinstance(parsed_source, str) and parsed_source.startswith("rtsp://"):
            self.cap = cv2.VideoCapture(parsed_source, cv2.CAP_FFMPEG)
        elif isinstance(parsed_source, int):
            self.cap = cv2.VideoCapture(parsed_source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(parsed_source)


        if not self.cap or not self.cap.isOpened():
            return False

        # Attempt to configure standard webcam parameters
        if isinstance(parsed_source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return True

    def _capture_loop(self):
        """Worker thread continuously pulling the latest frame from the buffer."""
        consecutive_failures = 0
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                consecutive_failures = 0
                with self.lock:
                    self.frame = frame
                    self.last_read_time = time.time()
            else:
                consecutive_failures += 1
                if consecutive_failures > 30:
                    # Connection lost; sleep briefly
                    time.sleep(0.1)

            time.sleep(0.005)

    def read_frame(self) -> Optional[np.ndarray]:
        """Returns a copy of the most recent frame."""
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
        return None

    def is_active(self) -> bool:
        return self.running and (self.cap is not None and self.cap.isOpened())

    def stop(self):
        """Stops background capture and releases capture device."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
            self.thread = None

        if self.cap:
            self.cap.release()
            self.cap = None

        with self.lock:
            self.frame = None
