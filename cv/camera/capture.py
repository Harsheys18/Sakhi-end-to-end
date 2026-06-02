"""
camera/capture.py

Camera capture abstraction wrapping OpenCV VideoCapture.

Responsibilities:
  - Open / release camera with auto-retry
  - Enforce resolution settings
  - Expose frames as numpy RGB arrays (pipeline convention)
  - Report actual FPS and frame dimensions
  - Timestamp each frame at capture time

Design: no state outside the CameraCapture class. No I/O side effects
other than the camera device itself.

Usage:
    cam = CameraCapture(device=0, width=640, height=480)
    cam.open()

    while cam.is_open():
        frame = cam.read()      # RGB uint8 or None on failure
        if frame is None:
            continue
        # use frame ...

    cam.release()

    # OR as context manager:
    with CameraCapture() as cam:
        frame = cam.read()
"""

import time
import cv2 as cv
import numpy as np


class CameraCapture:
    """
    Thin wrapper around cv2.VideoCapture for the social perception pipeline.

    Outputs frames as RGB uint8 numpy arrays (H×W×3).
    Timestamps are monotonic milliseconds since epoch.
    """

    def __init__(
        self,
        device:       int   = 0,
        width:        int   = 640,
        height:       int   = 480,
        target_fps:   float = 30.0,
        retry_delay_s: float = 0.5,
        max_retries:  int   = 5,
    ):
        """
        Args:
            device        : Camera index (0 = default webcam).
            width         : Requested frame width in pixels.
            height        : Requested frame height in pixels.
            target_fps    : Requested FPS (best-effort; hardware may differ).
            retry_delay_s : Seconds to wait between open retries.
            max_retries   : Max attempts to open the camera before giving up.
        """
        self.device        = device
        self.width         = width
        self.height        = height
        self.target_fps    = target_fps
        self.retry_delay_s = retry_delay_s
        self.max_retries   = max_retries

        self._cap: cv.VideoCapture | None = None

        # Actual dimensions reported by the driver (may differ from requested)
        self.actual_width:  int   = width
        self.actual_height: int   = height
        self.actual_fps:    float = target_fps

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def open(self) -> bool:
        """
        Open the camera device. Retries up to max_retries times.

        Returns:
            True if successfully opened, False otherwise.
        """
        for attempt in range(1, self.max_retries + 1):
            cap = cv.VideoCapture(self.device)
            if cap.isOpened():
                cap.set(cv.CAP_PROP_FRAME_WIDTH,  self.width)
                cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv.CAP_PROP_FPS,          self.target_fps)

                # Read back actual values (driver may clamp)
                self.actual_width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
                self.actual_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
                self.actual_fps    = float(cap.get(cv.CAP_PROP_FPS)) or self.target_fps

                self._cap = cap
                return True

            cap.release()
            if attempt < self.max_retries:
                time.sleep(self.retry_delay_s)

        return False

    def release(self) -> None:
        """Release the camera device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def is_open(self) -> bool:
        """True if the camera is currently open."""
        return self._cap is not None and self._cap.isOpened()

    # ── Frame acquisition ──────────────────────────────────────────────────────

    def read(self) -> tuple[np.ndarray | None, int]:
        """
        Capture one frame.

        Returns:
            (rgb_frame, timestamp_ms)
              rgb_frame    : HxWx3 uint8 RGB array, or None on failure.
              timestamp_ms : Monotonic milliseconds at capture time.
        """
        ts_ms = int(time.time() * 1000)

        if self._cap is None or not self._cap.isOpened():
            return None, ts_ms

        ok, bgr = self._cap.read()
        if not ok or bgr is None:
            return None, ts_ms

        rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
        return rgb, ts_ms

    @property
    def frame_shape(self) -> tuple[int, int]:
        """(width, height) of captured frames."""
        return (self.actual_width, self.actual_height)

    # ── Context manager ────────────────────────────────────────────────────────

    def __enter__(self) -> "CameraCapture":
        if not self.open():
            raise RuntimeError(
                f"Failed to open camera device {self.device} "
                f"after {self.max_retries} attempts."
            )
        return self

    def __exit__(self, *_) -> None:
        self.release()

    def __repr__(self) -> str:
        status = f"{self.actual_width}×{self.actual_height}@{self.actual_fps:.0f}fps"
        return f"CameraCapture(device={self.device}, {status}, open={self.is_open()})"
