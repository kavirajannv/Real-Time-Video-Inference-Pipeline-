import json
from dataclasses import replace
from pathlib import Path

import cv2
import pytest

from vpipe.config import FaultConfig, ModelSpec, PipelineConfig
from vpipe.pipeline import Pipeline
from vpipe.reader import SyntheticSource


def fake_config(tmp_path: Path, **overrides) -> PipelineConfig:
    cfg = PipelineConfig(
        output_dir=str(tmp_path),
        a_workers=2,
        b_workers=2,
        model_a=ModelSpec(kind="fake", fake_infer_s=0.002),
        model_b=ModelSpec(kind="fake", fake_infer_s=0.004),
        route_labels=("person",),
        route_threshold=0.30,
        timeout_a_s=10.0,
        timeout_b_s=10.0,
        startup_timeout_s=60.0,
        sample_interval_s=0.02,
    )
    return replace(cfg, **overrides)


def run_fake(tmp_path: Path, n: int = 40, **overrides):
    cfg = fake_config(tmp_path, **overrides)
    summary = Pipeline(cfg, source=SyntheticSource(n)).run()
    rows = json.loads((tmp_path / "frames_log.json").read_text())
    return summary, rows


def video_info(path: Path):
    cap = cv2.VideoCapture(str(path))
    n = 0
    shape = None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        shape = f.shape
        n += 1
    cap.release()
    return n, shape


@pytest.fixture
def fault():
    return FaultConfig
