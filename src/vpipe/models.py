"""Model backends. Each worker process builds exactly one of these.

Both real (Ultralytics YOLO11) and fake backends expose the same interface:
    names: dict[int, str]
    warmup() -> None
    predict(image: np.ndarray BGR) -> dict (classifier) | list[dict] (detector)

Fake backends decode the frame_id that `encode_frame` paints into synthetic frames,
which lets tests prove that every result is attached to the frame it was computed from.
"""

from __future__ import annotations

import time

import numpy as np

from .config import ModelSpec
from .schema import Stage


# --------------------------------------------------------------------------- real models
class YoloClassifier:
    def __init__(self, spec: ModelSpec):
        from ultralytics import YOLO

        self.model = YOLO(spec.weights, task="classify")
        self.imgsz = spec.imgsz
        self.names = {int(k): str(v) for k, v in self.model.names.items()}

    def warmup(self) -> None:
        self.predict(np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8))

    def predict(self, image: np.ndarray) -> dict:
        r = self.model.predict(image, imgsz=self.imgsz, device="cpu", verbose=False)[0]
        probs = r.probs
        top5 = [(self.names[int(i)], float(c)) for i, c in zip(probs.top5, probs.top5conf.tolist())]
        top1 = int(probs.top1)
        return {"label": self.names[top1], "conf": float(probs.top1conf), "top5": top5}


class YoloDetector:
    def __init__(self, spec: ModelSpec):
        from ultralytics import YOLO

        self.model = YOLO(spec.weights, task="detect")
        self.imgsz = spec.imgsz
        self.conf = spec.conf
        self.names = {int(k): str(v) for k, v in self.model.names.items()}

    def warmup(self) -> None:
        self.predict(np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8))

    def predict(self, image: np.ndarray) -> list[dict]:
        r = self.model.predict(image, imgsz=self.imgsz, conf=self.conf, device="cpu", verbose=False)[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.tolist()
        confs = boxes.conf.tolist()
        clss = boxes.cls.tolist()
        return [
            {"label": self.names[int(c)], "cls": int(c), "conf": float(p), "box": [float(v) for v in b]}
            for b, p, c in zip(xyxy, confs, clss)
        ]


class PersonGate:
    """Model A alternative: "does this frame contain a person?" from a small, low-res detector.

    ImageNet classifiers have no person class, so for people analytics the gate is a
    YOLO11n detector restricted to COCO class 0 at low resolution (fast). Its output has
    the same shape as the classifier's, so routing is unchanged: label "person" with
    conf = highest person confidence (>= spec.conf), otherwise "no_person".
    """

    names = {0: "no_person", 1: "person"}

    def __init__(self, spec: ModelSpec):
        from ultralytics import YOLO

        self.model = YOLO(spec.weights, task="detect")
        self.imgsz = spec.imgsz
        self.conf = spec.conf
        self.person_cls = next(int(k) for k, v in self.model.names.items() if v == "person")

    def warmup(self) -> None:
        self.predict(np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8))

    def predict(self, image: np.ndarray) -> dict:
        r = self.model.predict(image, imgsz=self.imgsz, conf=0.05, classes=[self.person_cls], device="cpu",
                               verbose=False)[0]
        confs = r.boxes.conf.tolist() if r.boxes is not None else []
        best = max(confs, default=0.0)
        n = sum(1 for c in confs if c >= self.conf)
        if best >= self.conf:
            return {"label": "person", "conf": best, "top5": [("person", best)], "people_estimate": n}
        return {"label": "no_person", "conf": 1.0 - best, "top5": [("no_person", 1.0 - best)], "people_estimate": 0}


# --------------------------------------------------------------------------- synthetic frames
def encode_frame(frame_id: int, height: int = 120, width: int = 160) -> np.ndarray:
    """Solid-colour frame whose pixels encode frame_id (survives resizing)."""
    img = np.empty((height, width, 3), dtype=np.uint8)
    img[..., 0] = frame_id % 256
    img[..., 1] = (frame_id // 256) % 256
    img[..., 2] = 128
    return img


def decode_frame_id(image: np.ndarray) -> int:
    h, w = image.shape[:2]
    px = image[h // 2, w // 2]
    return int(px[0]) + 256 * int(px[1])


def fake_classification(frame_id: int) -> tuple[str, float]:
    """Deterministic ground truth for tests: id%3==0 routable, id%3==2 below threshold."""
    r = frame_id % 3
    if r == 0:
        return "person", 0.90
    if r == 1:
        return "car", 0.80
    return "person", 0.20


def fake_detections(frame_id: int) -> list[dict]:
    if frame_id % 2 == 0:
        return [{"label": f"obj{frame_id}", "conf": 0.5, "box": [10.0, 10.0, 50.0, 50.0]}]
    return []


class FakeClassifier:
    names = {0: "person", 1: "car"}

    def __init__(self, spec: ModelSpec):
        self.infer_s = spec.fake_infer_s

    def warmup(self) -> None:
        pass

    def predict(self, image: np.ndarray) -> dict:
        if self.infer_s:
            time.sleep(self.infer_s)
        label, conf = fake_classification(decode_frame_id(image))
        return {"label": label, "conf": conf, "top5": [(label, conf)]}


class FakeDetector:
    names = {0: "obj"}

    def __init__(self, spec: ModelSpec):
        self.infer_s = spec.fake_infer_s

    def warmup(self) -> None:
        pass

    def predict(self, image: np.ndarray) -> list[dict]:
        if self.infer_s:
            time.sleep(self.infer_s)
        return fake_detections(decode_frame_id(image))


def build_model(stage: Stage, spec: ModelSpec):
    if spec.kind == "fake":
        return FakeClassifier(spec) if stage == Stage.A else FakeDetector(spec)
    if spec.kind == "yolo":
        return YoloClassifier(spec) if stage == Stage.A else YoloDetector(spec)
    if spec.kind == "yolo_gate":
        return PersonGate(spec)
    raise ValueError(f"unknown model kind: {spec.kind!r}")
