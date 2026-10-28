"""Live preview window: input frame (left) and annotated output (right) while the pipeline runs.

The window has its own thread (OpenCV's window calls must all come from one thread). The writer hands
it frames through a small inbox that never blocks (a slow display skips frames instead of slowing the
writer down).

Frames leave the pipeline in bursts: output is strictly in order, so the frames after an analysed one
wait for its Model B result and are then released together. Showing each frame as it arrives looks
jerky, so the preview plays them back at the pace they were read: frame shown at t_ingest + delay,
where the delay follows the slowest recent frame (a steady, slightly later picture beats a stuttering
one). q / Esc stops the run (like Ctrl+C: every frame read so far is still written); the window's close
button hides the preview and lets the run continue.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

import cv2
import numpy as np

log = logging.getLogger(__name__)

WINDOW = "vpipe: input | output"


def fit_height(frame: np.ndarray, height: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if h == height:
        return frame.copy()
    width = round(w * height / h / 2) * 2  # even width: libx264 needs even dimensions
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def label(frame: np.ndarray, text: str) -> None:
    # bottom-left: the annotated output already draws its own overlay along the top
    h = frame.shape[0]
    cv2.rectangle(frame, (0, h - 44), (20 + 16 * len(text), h), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)


def side_by_side(original: np.ndarray, annotated: np.ndarray, height: int, frame_id: int) -> np.ndarray:
    left, right = fit_height(original, height), fit_height(annotated, height)
    label(left, "INPUT")
    label(right, f"OUTPUT  #{frame_id}")
    gap = np.full((height, 6, 3), 255, np.uint8)
    return np.hstack([left, gap, right])


class LivePreview:
    def __init__(self, height: int = 540, on_quit=None, max_delay_s: float = 3.0):
        self.height = height // 2 * 2
        self.on_quit = on_quit  # called when the user presses q / Esc
        self.max_delay_s = max_delay_s
        self._inbox: deque = deque(maxlen=16)  # (original, annotated, frame_id, t_ingest) not composed yet
        self._cond = threading.Condition()
        self._done = False
        self.enabled = True  # False once the user closed the window
        self._thread = threading.Thread(target=self._loop, name="vpipe-preview", daemon=True)
        self._thread.start()

    def show(self, original: np.ndarray | None, annotated: np.ndarray, frame_id: int,
             t_ingest: float | None = None) -> None:
        """Never blocks: if the window falls behind, the oldest uncomposed frames are skipped."""
        if not self.enabled:
            return
        if original is None:  # invalid frame: the writer drew on a black placeholder
            original = np.zeros_like(annotated)
        with self._cond:
            self._inbox.append((original, annotated, frame_id, t_ingest or time.perf_counter()))
            self._cond.notify()

    def close(self) -> None:
        with self._cond:
            self._done = True
            self._cond.notify()
        self._thread.join(timeout=5.0)

    def _loop(self) -> None:
        queued: deque = deque()          # (due time, canvas) in frame order
        latencies: deque = deque(maxlen=90)  # arrival - ingest of the last ~3 s of frames
        sized = False
        try:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            while True:
                with self._cond:
                    if not self._inbox and not self._done:
                        wait = queued[0][0] - time.perf_counter() if queued else 0.03
                        self._cond.wait(timeout=min(max(wait, 0.001), 0.03))
                    items = list(self._inbox)
                    self._inbox.clear()
                    if self._done:
                        return
                now = time.perf_counter()
                for original, annotated, frame_id, t_ingest in items:
                    latencies.append(now - t_ingest)
                    delay = min(max(latencies) + 0.05, self.max_delay_s)
                    queued.append((t_ingest + delay, side_by_side(original, annotated, self.height, frame_id)))
                # show the newest frame that is due; older due ones are skipped (only if running late)
                canvas = None
                now = time.perf_counter()
                while queued and queued[0][0] <= now:
                    canvas = queued.popleft()[1]
                while len(queued) > 60:  # bound memory if the delay jumps
                    canvas = queued.popleft()[1]
                if canvas is not None:
                    if not sized:
                        cv2.resizeWindow(WINDOW, canvas.shape[1], canvas.shape[0])
                        sized = True
                    cv2.imshow(WINDOW, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    log.warning("q pressed in the preview: stopping the run")
                    if self.on_quit is not None:
                        self.on_quit()
                    return
                if sized and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    log.info("preview window closed; the run continues")
                    return
        except cv2.error as e:  # e.g. no display / headless OpenCV build
            log.warning("live preview unavailable: %s", e)
        finally:
            self.enabled = False
            with self._cond:
                self._inbox.clear()
            try:
                cv2.destroyWindow(WINDOW)
                cv2.waitKey(1)
            except cv2.error:
                pass
