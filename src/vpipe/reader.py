"""Frame sources and the ingest (reader) thread."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .models import encode_frame
from .schema import STRIDE_REASON, AStatus, Stage, Task

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- sources
class VideoSource:
    """cv2 video reader. `repeat` > 1 replays the file back-to-back (benchmarks need more samples)."""

    def __init__(self, path: str, repeat: int = 1):
        self.path = path
        self.repeat = max(1, repeat)
        self._loop = 1
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            hint = " (run `python -m vpipe download` to fetch the default video)" if not Path(path).exists() else ""
            raise FileNotFoundError(f"cannot open video: {path}{hint}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) * self.repeat

    def read(self) -> tuple[bool, np.ndarray | None]:
        ok, frame = self.cap.read()
        if not ok and self._loop < self.repeat:
            self._loop += 1
            self.cap.release()
            self.cap = cv2.VideoCapture(str(self.path))
            ok, frame = self.cap.read()
        return ok, frame

    def close(self) -> None:
        self.cap.release()


class SyntheticSource:
    """Frames that encode their own id (see models.encode_frame) - for tests and dry runs."""

    def __init__(self, n: int, width: int = 160, height: int = 120, fps: float = 30.0):
        self.n, self.width, self.height, self.fps = n, width, height, fps
        self.frame_count = n
        self._i = 0

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._i >= self.n:
            return False, None
        img = encode_frame(self._i, self.height, self.width)
        self._i += 1
        return True, img

    def close(self) -> None:
        pass


def is_valid_frame(frame, width: int, height: int) -> bool:
    return (isinstance(frame, np.ndarray) and frame.dtype == np.uint8 and frame.ndim == 3
            and frame.shape == (height, width, 3))


# --------------------------------------------------------------------------- preprocessing
def prep_for_a(frame: np.ndarray, imgsz: int, enabled: bool) -> np.ndarray:
    """Shrink so the short side == imgsz (what the classifier's Resize+CenterCrop does anyway)."""
    if not enabled:
        return frame
    h, w = frame.shape[:2]
    s = imgsz / min(h, w)
    if s >= 1.0:
        return frame
    # INTER_LINEAR: ~4x cheaper than INTER_AREA on 1080p and what Ultralytics' own letterbox uses
    return cv2.resize(frame, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_LINEAR)


def prep_for_b(frame: np.ndarray, imgsz: int, enabled: bool) -> tuple[np.ndarray, float]:
    """Shrink so the long side == imgsz. Returns (image, scale back to original pixels)."""
    if not enabled:
        return frame, 1.0
    h, w = frame.shape[:2]
    s = imgsz / max(h, w)
    if s >= 1.0:
        return frame, 1.0
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR), w / nw


# --------------------------------------------------------------------------- reader thread
@dataclass
class IngestEvent:
    frame_id: int
    t_ingest: float
    t_enqueue: float | None
    status: AStatus = AStatus.PENDING
    reason: str = ""


@dataclass
class EndOfStream:
    total_frames: int


class FrameStore:
    """Original full-resolution frames, kept in the main process until written."""

    def __init__(self):
        self._d: dict[int, np.ndarray | None] = {}
        self._lock = threading.Lock()

    def put(self, fid: int, img) -> None:
        with self._lock:
            self._d[fid] = img

    def get(self, fid: int):
        with self._lock:
            return self._d.get(fid)

    def pop(self, fid: int):
        with self._lock:
            return self._d.pop(fid, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._d)


class Reader(threading.Thread):
    """Reads frames, assigns ids/timestamps and submits them to queue A.

    offline: blocking put  -> backpressure, never drops (may run slower than real time)
    live:    paced at fps, put_nowait -> frame is DROPPED (and recorded) if queue A is full
    Memory is bounded by `slots`: one slot per frame between ingest and write.
    """

    def __init__(self, source, task_q, events: queue.Queue, store: FrameStore, slots: threading.Semaphore,
                 stop: threading.Event, *, mode: str, fps: float, imgsz_a: int, resize: bool,
                 max_frames: int = 0, invalid_every: int = 0, infer_every: int = 1, gate: bool = True,
                 pace: bool = False):
        super().__init__(name="vpipe-reader", daemon=True)
        self.source, self.task_q, self.events, self.store, self.slots = source, task_q, events, store, slots
        self.stop_event = stop
        self.mode, self.fps, self.imgsz_a, self.resize = mode, fps, imgsz_a, resize
        self.max_frames, self.invalid_every = max_frames, invalid_every
        self.infer_every = max(1, infer_every)  # fast mode: only every Nth frame goes to the models
        self.gate = gate                         # False: inferred frames skip Model A, straight to Model B
        self.pace = pace                         # cap ingest at fps without dropping (offline + real-time cap)
        self.t_first: float | None = None
        self.t_last: float | None = None
        self.blocked_s = 0.0
        self.dropped = 0
        self.behind_schedule = 0
        self.error: BaseException | None = None

    def _acquire_slot(self) -> bool:
        t = time.perf_counter()
        while not self.slots.acquire(timeout=0.05):
            if self.stop_event.is_set():
                return False
        self.blocked_s += time.perf_counter() - t
        return True

    def run(self) -> None:
        fid = 0
        try:
            fid = self._loop()
        except BaseException as e:  # noqa: BLE001 - surfaced to the main loop
            self.error = e
            log.exception("reader failed")
        finally:
            self.events.put(EndOfStream(fid))

    def _loop(self) -> int:
        fid = 0
        live = self.mode == "live"
        interval = 1.0 / self.fps if self.fps > 0 else 0.0
        t0 = time.perf_counter()
        while not self.stop_event.is_set():
            if self.max_frames and fid >= self.max_frames:
                break
            if not self._acquire_slot():
                break
            if live or self.pace:
                delay = t0 + fid * interval - time.perf_counter()
                if delay > 0:
                    if self.stop_event.wait(delay):
                        self.slots.release()
                        break
                elif delay < -interval:
                    self.behind_schedule += 1
            ok, frame = self.source.read()
            if not ok:
                self.slots.release()
                break
            t_ingest = time.perf_counter()
            if self.t_first is None:
                self.t_first = t_ingest
            self.t_last = t_ingest

            if self.invalid_every and fid % self.invalid_every == self.invalid_every - 1:
                frame = None  # injected corrupt frame
            if not is_valid_frame(frame, self.source.width, self.source.height):
                self.store.put(fid, None)
                self.events.put(IngestEvent(fid, t_ingest, None, AStatus.INVALID, "invalid/corrupt frame"))
                fid += 1
                continue

            self.store.put(fid, frame)
            if fid % self.infer_every:
                # not inferred by design (fast mode): recorded and written like every other frame
                self.events.put(IngestEvent(fid, t_ingest, None, AStatus.HELD,
                                            f"{STRIDE_REASON}: models run every {self.infer_every} frames"))
                fid += 1
                continue
            if not self.gate:
                self.events.put(IngestEvent(fid, t_ingest, None, AStatus.NOT_GATED, "fast mode: no gate"))
                fid += 1
                continue
            task = Task(Stage.A, fid, prep_for_a(frame, self.imgsz_a, self.resize), t_ingest, 0.0)
            task.t_enqueue = time.perf_counter()
            if live:
                try:
                    self.task_q.put_nowait(task)
                    self.events.put(IngestEvent(fid, t_ingest, task.t_enqueue))
                except queue.Full:
                    self.dropped += 1
                    self.events.put(IngestEvent(fid, t_ingest, None, AStatus.DROPPED, "queue A full (live overload)"))
            else:
                if self._put_blocking(task):
                    self.events.put(IngestEvent(fid, t_ingest, task.t_enqueue))
                else:
                    self.events.put(IngestEvent(fid, t_ingest, None, AStatus.ABORTED, "stopped before submission"))
                    fid += 1
                    break
            fid += 1
        return fid

    def _put_blocking(self, task: Task) -> bool:
        t = time.perf_counter()
        while True:
            try:
                self.task_q.put(task, timeout=0.05)
                self.blocked_s += time.perf_counter() - t
                return True
            except queue.Full:
                if self.stop_event.is_set():
                    return False
