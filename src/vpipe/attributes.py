"""Person attributes (gender estimate, bag, backpack, hat, ...) for tracked people.

Model: OpenVINO person-attributes-recognition-crossroad-0230 (BGR crop 80x160 -> 8 probabilities).
It runs in the writer thread on crops of the full-resolution original frame, sampled a few
times per track, and the probabilities are averaged over the track. A single crop is noisy
(pose, occlusion, motion blur), so a person whose average stays near the middle is reported as
"unknown" instead of being forced into male/female.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2
import numpy as np

ATTRIBUTES = ("is_male", "has_bag", "has_backpack", "has_hat", "has_longsleeves", "has_longpants",
              "has_longhair", "has_coat_jacket")
DEFAULT_MODEL = "models/person-attributes-recognition-crossroad-0230.xml"
THUMB_W, THUMB_H = 80, 160


class PersonAttributeModel:
    def __init__(self, xml_path: str, threads: int = 2):
        import openvino as ov

        # few threads: this runs in the main process next to the inference worker pools
        self._compiled = ov.Core().compile_model(xml_path, "CPU", {"INFERENCE_NUM_THREADS": threads})
        self._out = self._compiled.output(0)
        self._local = threading.local()  # an infer request is not thread-safe: one per calling thread

    def predict(self, crop: np.ndarray) -> np.ndarray:
        inp = cv2.resize(crop, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)[None]
        req = getattr(self._local, "req", None)
        if req is None:
            req = self._local.req = self._compiled.create_infer_request()
        out = req.infer({0: inp.astype(np.float32)})[self._out]
        return np.asarray(out, dtype=np.float32).reshape(-1)[: len(ATTRIBUTES)]


@dataclass
class _TrackState:
    total: np.ndarray = field(default_factory=lambda: np.zeros(len(ATTRIBUTES), dtype=np.float64))
    n: int = 0
    queued: int = 0                       # samples submitted (n = samples finished)
    last_frame: int = -(10**9)
    thumb: np.ndarray | None = None
    thumb_h: float = 0.0


class TrackAttributes:
    """Samples crops per track, averages attribute probabilities, and decides the labels."""

    def __init__(self, model, size: tuple[int, int], every: int = 6, max_samples: int = 12, min_height: int = 80,
                 male_threshold: float = 0.5, female_threshold: float = 0.35, min_samples: int = 3,
                 attr_threshold: float = 0.5, background: bool = False):
        self.model = model
        self.w, self.h = size
        self.every = every
        self.max_samples = max_samples
        self.min_height = min_height
        self.male_threshold = male_threshold
        self.female_threshold = female_threshold
        self.min_samples = min_samples
        self.attr_threshold = attr_threshold
        self.tracks: dict[int, _TrackState] = {}
        # background: crops are classified on a small thread pool (OpenVINO releases the GIL) so the
        # ordered writer stage does not wait; labels then appear a few frames after a sample is taken
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vpipe-attr") if background else None
        self._futures: list = []

    def _usable(self, box) -> bool:
        x1, y1, x2, y2 = box
        edge = 4  # a box touching the frame border is a partial body: its attributes are guesses
        return (y2 - y1) >= self.min_height and x1 > edge and y1 > edge and x2 < self.w - edge and y2 < self.h - edge

    def observe(self, frame_id: int, gid: int, box, image: np.ndarray) -> None:
        st = self.tracks.setdefault(gid, _TrackState())
        if st.queued >= self.max_samples or frame_id - st.last_frame < self.every or not self._usable(box):
            return
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return
        st.queued += 1
        st.last_frame = frame_id
        if self._pool is None:
            self._classify(st, crop, y2 - y1)
        else:
            self._futures = [f for f in self._futures if not f.done()]
            self._futures.append(self._pool.submit(self._classify, st, crop.copy(), y2 - y1))

    def _classify(self, st: _TrackState, crop: np.ndarray, height: int) -> None:
        probs = self.model.predict(crop)
        thumb = cv2.resize(crop, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA) if height > st.thumb_h else None
        with self._lock:
            st.total += probs
            st.n += 1
            if thumb is not None and height > st.thumb_h:
                st.thumb, st.thumb_h = thumb, height

    def flush(self) -> None:
        """Wait for background classifications (call before reading final results)."""
        if self._pool is not None:
            for f in self._futures:
                f.result()
            self._futures = []
            self._pool.shutdown(wait=True)
            self._pool = None

    def probs(self, gid: int) -> dict | None:
        st = self.tracks.get(gid)
        if st is None or st.n == 0:
            return None
        return dict(zip(ATTRIBUTES, (st.total / st.n).tolist()))

    def gender(self, gid: int) -> str:
        st = self.tracks.get(gid)
        if st is None or st.n < self.min_samples:
            return "unknown"
        p_male = st.total[0] / st.n
        if p_male >= self.male_threshold:
            return "male"
        if p_male <= self.female_threshold:
            return "female"
        return "unknown"

    def summary(self, gid: int) -> dict:
        p = self.probs(gid)
        st = self.tracks.get(gid)
        out = {"gender": self.gender(gid), "samples": st.n if st else 0,
               "p_male": None if p is None else round(p["is_male"], 3)}
        for name in ATTRIBUTES[1:]:
            known = p is not None and st.n >= self.min_samples
            out[name] = (p[name] >= self.attr_threshold) if known else None
        return out

    def gallery(self, gids_by_gender: dict[str, list[int]], per_row: int = 16, rows: int = 2) -> np.ndarray | None:
        """Sample crops per gender class, so the labels can be checked by eye."""
        bands = []
        for gender, gids in gids_by_gender.items():
            thumbs = []
            for gid in gids:
                st = self.tracks.get(gid)
                if st is None or st.thumb is None:
                    continue
                t = st.thumb.copy()
                cv2.rectangle(t, (0, THUMB_H - 16), (THUMB_W, THUMB_H), (20, 20, 20), -1)
                p = st.total[0] / st.n
                cv2.putText(t, f"#{gid} {p:.2f}", (2, THUMB_H - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255),
                            1, cv2.LINE_AA)
                thumbs.append(t)
                if len(thumbs) == per_row * rows:
                    break
            head = np.full((28, per_row * THUMB_W, 3), 30, dtype=np.uint8)
            cv2.putText(head, f"{gender}: {len(gids)} people (showing {len(thumbs)}; number = mean p(male))",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            thumbs += [np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)] * (-len(thumbs) % per_row or 0)
            if not thumbs:
                thumbs = [np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)] * per_row
            grid = np.vstack([np.hstack(thumbs[i:i + per_row]) for i in range(0, len(thumbs), per_row)])
            bands += [head, grid]
        return np.vstack(bands) if bands else None
