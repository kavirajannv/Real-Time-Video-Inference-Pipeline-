"""Fetch the input video and the pretrained weights (Ultralytics' documented loading process)."""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

from .config import DEFAULT_VIDEO_URL

log = logging.getLogger(__name__)


def download_video(dest: str = "data/solutions_ci_demo.mp4", url: str = DEFAULT_VIDEO_URL) -> Path:
    path = Path(dest)
    if path.exists() and path.stat().st_size > 0:
        log.info("video already present: %s", path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s -> %s", url, path)
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)
    return path


def download_weights(weights=("yolo11n-cls.pt", "yolo11n.pt")) -> None:
    """YOLO('<name>.pt') downloads official weights to the working directory on first use."""
    from ultralytics import YOLO

    for w in weights:
        m = YOLO(w)
        log.info("weights ready: %s (%d classes, task=%s)", w, len(m.names), m.task)
