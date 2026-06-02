"""
temporal/eye_temporal.py

Temporal blink tracking over a rolling time window.

Consumes the per-frame output of features/eye_features.py and maintains
rolling deques to produce stable temporal estimates.

Tracks:
  - blink_rate          : blinks per minute over the trailing window
  - avg_blink_duration  : mean duration (ms) of completed blink events
  - eye_open_duration   : continuous ms the eye has been open (current streak)
  - eye_closure_duration: continuous ms the eye has been closed (current streak)
  - blink_trend         : slope of blink-rate over two halves of the window
                          positive → rate increasing (fatigue signal)
                          negative → rate decreasing
  - ear_mean / ear_std  : rolling EAR statistics (used by downstream modules)

Design rules:
  - No side effects. No printing. No drawing. No global state.
  - All state lives in the BlinkTracker instance (one per face).
  - update() is O(1) amortized (deque pop is O(1)).
"""

from collections import deque
import numpy as np


# ---------------------------------------------------------------------------
# Configuration defaults (can be overridden at construction time)
# ---------------------------------------------------------------------------

DEFAULT_WINDOW_SECONDS   = 30       # Rolling window for blink rate
DEFAULT_MIN_BLINK_GAP_MS = 100      # Ignore blinks closer than this (noise)
DEFAULT_BLINK_MIN_DUR_MS = 50       # Blinks shorter than this are noise
DEFAULT_BLINK_MAX_DUR_MS = 500      # Blinks longer than this are eye closures
DEFAULT_EAR_BUFFER_LEN   = 90       # EAR history depth (frames, ~3 s at 30 fps)


class BlinkTracker:
    """
    Stateful temporal tracker for one face's eye behavior.

    Instantiate one BlinkTracker per face slot. If the number of detected
    faces is dynamic, manage a dict of trackers keyed by face_idx in the
    caller.

    Usage:
        tracker = BlinkTracker()

        # inside the main capture loop:
        eye_feats = extract_eye_features(landmarks, face_idx=0, image_wh=wh)
        temporal  = tracker.update(eye_feats, timestamp_ms=int(time.time()*1000))
    """

    def __init__(
        self,
        window_seconds:   float = DEFAULT_WINDOW_SECONDS,
        min_blink_gap_ms: int   = DEFAULT_MIN_BLINK_GAP_MS,
        blink_min_dur_ms: int   = DEFAULT_BLINK_MIN_DUR_MS,
        blink_max_dur_ms: int   = DEFAULT_BLINK_MAX_DUR_MS,
        ear_buffer_len:   int   = DEFAULT_EAR_BUFFER_LEN,
    ):
        self.window_ms        = window_seconds * 1000
        self.min_blink_gap_ms = min_blink_gap_ms
        self.blink_min_dur_ms = blink_min_dur_ms
        self.blink_max_dur_ms = blink_max_dur_ms

        # --- Rolling EAR buffer (fixed-length deque) ---
        self._ear_buf: deque[float] = deque(maxlen=ear_buffer_len)

        # --- Blink event timestamps (ms) within the rolling window ---
        # Each entry is the timestamp of a confirmed blink peak.
        self._blink_times: deque[float] = deque()

        # --- Blink duration tracking (state machine) ---
        # States: "open" | "closing" | "closed"
        self._eye_state:         str   = "open"
        self._state_entry_ms:    float = 0.0    # when we entered current state
        self._blink_start_ms:    float = 0.0    # when eye started closing
        self._last_blink_end_ms: float = -1e9   # debounce guard

        # Completed blink durations (ms) — capped deque for rolling avg
        self._blink_durations: deque[float] = deque(maxlen=50)

        # --- Continuous open/closed streak ---
        self._open_since_ms:   float = 0.0
        self._closed_since_ms: float = 0.0

        # --- First update flag ---
        self._initialized = False

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def update(self, eye_features: dict, timestamp_ms: float) -> dict:
        """
        Ingest one frame's eye features and return enriched temporal output.

        Args:
            eye_features  : dict from features/eye_features.extract_eye_features()
            timestamp_ms  : current time in milliseconds (monotonic preferred)

        Returns:
            {
                # --- Raw pass-through (from eye_features) ---
                "left_EAR"             : float
                "right_EAR"            : float
                "avg_EAR"              : float
                "blink_detected"       : bool

                # --- Temporal estimates ---
                "blink_rate"           : float   # blinks/minute, trailing window
                "avg_blink_duration_ms": float   # ms, rolling mean of recent blinks
                "eye_open_duration_ms" : float   # ms, current continuous open streak
                "eye_closure_duration_ms": float # ms, current continuous closed streak
                "blink_trend"          : float   # +ve = rate rising, -ve = falling
                "ear_mean"             : float   # rolling mean EAR
                "ear_std"              : float   # rolling std  EAR (instability proxy)

                # --- Metadata ---
                "window_seconds"       : float
                "confidence"           : float   # inherited from eye_features
            }
        """
        avg_ear       = eye_features.get("avg_EAR", 0.0)
        blink_frame   = eye_features.get("blink_detected", False)
        ear_conf      = eye_features.get("ear_confidence", 0.0)

        # --- Initialise timestamps on first call ---
        if not self._initialized:
            self._open_since_ms   = timestamp_ms
            self._closed_since_ms = timestamp_ms
            self._state_entry_ms  = timestamp_ms
            self._initialized     = True

        # --- Update EAR rolling buffer ---
        self._ear_buf.append(avg_ear)

        # --- Advance blink state machine ---
        self._advance_state(blink_frame, avg_ear, timestamp_ms)

        # --- Expire old blink timestamps outside rolling window ---
        cutoff = timestamp_ms - self.window_ms
        while self._blink_times and self._blink_times[0] < cutoff:
            self._blink_times.popleft()

        # --- Derive temporal metrics ---
        blink_rate       = self._compute_blink_rate(timestamp_ms)
        avg_blink_dur    = self._compute_avg_blink_duration()
        blink_trend      = self._compute_blink_trend(timestamp_ms)
        ear_mean, ear_std = self._compute_ear_stats()

        # Current streak durations
        if self._eye_state == "open":
            open_dur   = timestamp_ms - self._open_since_ms
            closed_dur = 0.0
        else:
            open_dur   = 0.0
            closed_dur = timestamp_ms - self._closed_since_ms

        return {
            # Pass-through
            "left_EAR"               : eye_features.get("left_EAR",   0.0),
            "right_EAR"              : eye_features.get("right_EAR",  0.0),
            "avg_EAR"                : avg_ear,
            "blink_detected"         : blink_frame,

            # Temporal
            "blink_rate"             : round(blink_rate,    2),
            "avg_blink_duration_ms"  : round(avg_blink_dur, 1),
            "eye_open_duration_ms"   : round(open_dur,      1),
            "eye_closure_duration_ms": round(closed_dur,    1),
            "blink_trend"            : round(blink_trend,   4),
            "ear_mean"               : round(ear_mean,      4),
            "ear_std"                : round(ear_std,       4),

            # Meta
            "window_seconds"         : self.window_ms / 1000.0,
            "confidence"             : ear_conf,
        }

    def reset(self) -> None:
        """Clear all internal state (call when face is lost)."""
        self._ear_buf.clear()
        self._blink_times.clear()
        self._blink_durations.clear()
        self._eye_state        = "open"
        self._initialized      = False

    # -----------------------------------------------------------------------
    # Private — state machine
    # -----------------------------------------------------------------------

    def _advance_state(
        self,
        blink_frame: bool,
        avg_ear:     float,
        ts:          float,
    ) -> None:
        """
        Two-state machine: open ↔ closed.

        A blink event is confirmed when:
          1. Eye transitions open → closed  (avg_ear drops below threshold)
          2. Eye transitions closed → open  (avg_ear rises above threshold)
          3. The closure duration is within [blink_min_dur, blink_max_dur]
          4. Time since last blink > min_blink_gap  (debounce)

        Closures longer than blink_max_dur are classified as prolonged closure,
        not blinks — they still update eye_closure_duration but don't add to
        blink_rate. (Useful downstream for drowsiness detection.)
        """
        if self._eye_state == "open":
            if blink_frame:
                # Transition to closing
                self._eye_state       = "closed"
                self._blink_start_ms  = ts
                self._closed_since_ms = ts
        else:
            # Currently closed
            if not blink_frame:
                # Transition back to open
                closure_dur = ts - self._blink_start_ms
                gap_ok      = (ts - self._last_blink_end_ms) > self.min_blink_gap_ms
                is_blink    = (
                    self.blink_min_dur_ms <= closure_dur <= self.blink_max_dur_ms
                    and gap_ok
                )
                if is_blink:
                    self._blink_times.append(ts)
                    self._blink_durations.append(closure_dur)
                    self._last_blink_end_ms = ts

                self._eye_state    = "open"
                self._open_since_ms = ts

    # -----------------------------------------------------------------------
    # Private — metric computation
    # -----------------------------------------------------------------------

    def _compute_blink_rate(self, now_ms: float) -> float:
        """
        Blinks per minute within the rolling window.

        Uses only the portion of the window that has actually elapsed,
        preventing artificially high rates at startup.
        """
        n = len(self._blink_times)
        if n == 0:
            return 0.0

        elapsed_s = min(now_ms - self._blink_times[0], self.window_ms) / 1000.0
        if elapsed_s < 0.5:
            # Not enough data yet
            return 0.0

        return (n / elapsed_s) * 60.0

    def _compute_avg_blink_duration(self) -> float:
        """Rolling mean blink closure duration in milliseconds."""
        if not self._blink_durations:
            return 0.0
        return float(np.mean(self._blink_durations))

    def _compute_blink_trend(self, now_ms: float) -> float:
        """
        Compare blink rate in the first vs second half of the window.

        Returns:
            Positive → rate increasing (possible fatigue onset)
            Negative → rate decreasing
            0.0      → insufficient data
        """
        if len(self._blink_times) < 4:
            return 0.0

        mid = now_ms - (self.window_ms / 2.0)
        first_half  = sum(1 for t in self._blink_times if t <  mid)
        second_half = sum(1 for t in self._blink_times if t >= mid)

        # Normalise to rate-per-minute per half-window
        half_min = (self.window_ms / 2.0) / 60000.0
        if half_min < 1e-6:
            return 0.0

        rate_first  = first_half  / half_min
        rate_second = second_half / half_min

        return round(rate_second - rate_first, 3)

    def _compute_ear_stats(self) -> tuple[float, float]:
        """Rolling mean and std of EAR buffer."""
        if not self._ear_buf:
            return 0.0, 0.0
        arr = np.array(self._ear_buf, dtype=np.float32)
        return float(arr.mean()), float(arr.std())
