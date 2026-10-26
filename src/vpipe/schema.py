"""Messages exchanged between pipeline stages and the per-frame record kept by the aggregator.

Everything that crosses a process boundary (Task, WorkerResult, WorkerReady, WorkerFailed)
is a small picklable dataclass. Every message carries the frame_id it belongs to so the
aggregator can verify association instead of trusting arrival order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class Stage(str, Enum):
    A = "A"
    B = "B"


class AStatus(str, Enum):
    PENDING = "PENDING"
    OK = "OK"
    ERROR = "ERROR"          # worker raised an exception
    TIMEOUT = "TIMEOUT"      # no result before the stage deadline
    CRASHED = "CRASHED"      # worker process died while holding this frame
    STALE = "STALE"          # live mode: frame too old when a worker picked it up
    DROPPED = "DROPPED"      # live mode: never submitted, input queue was full
    INVALID = "INVALID"      # unreadable / malformed frame from the source
    ABORTED = "ABORTED"      # application stopped before a result arrived
    HELD = "HELD"            # fast mode: by design not sent to the models; the tracker holds the boxes
    NOT_GATED = "NOT_GATED"  # fast mode without gate: sent straight to Model B (Model A not run)


class BStatus(str, Enum):
    PENDING = "PENDING"
    DETECTIONS = "DETECTIONS"
    NO_DETECTIONS = "NO_DETECTIONS"
    SKIPPED = "SKIPPED"      # routing condition not met (never submitted to Model B)
    NOT_RUN = "NOT_RUN"      # Model A did not produce a usable result
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"
    CRASHED = "CRASHED"
    ABORTED = "ABORTED"


A_FAILURES = {AStatus.ERROR, AStatus.TIMEOUT, AStatus.CRASHED, AStatus.STALE, AStatus.ABORTED}
B_FAILURES = {BStatus.ERROR, BStatus.TIMEOUT, BStatus.CRASHED, BStatus.ABORTED}

SKIP_REASON = "Routing condition not met"
STRIDE_REASON = "Detection stride"  # routed by Model A, but Model B only runs on every Nth frame


@dataclass
class Task:
    stage: Stage
    frame_id: int
    image: np.ndarray
    t_ingest: float
    t_enqueue: float


@dataclass
class WorkerResult:
    stage: Stage
    frame_id: int
    worker_id: str
    ok: bool
    output: Any = None             # A: {"label", "conf", "top5"}; B: list of detections
    error: str | None = None
    t_enqueue: float = 0.0
    t_start: float = 0.0
    t_end: float = 0.0
    stale: bool = False


@dataclass
class WorkerReady:
    stage: Stage
    worker_id: str
    pid: int
    names: dict[int, str]
    load_s: float


@dataclass
class WorkerFailed:
    """Model could not be loaded; the pipeline cannot run this stage."""

    stage: Stage
    worker_id: str
    error: str


@dataclass
class FrameRecord:
    """Everything known about one input frame. Lives only in the main process."""

    frame_id: int
    t_ingest: float
    a_status: AStatus = AStatus.PENDING
    b_status: BStatus = BStatus.PENDING
    final: bool = False

    a_label: str | None = None
    a_conf: float | None = None
    a_top5: list | None = None
    a_worker: str | None = None
    a_error: str | None = None
    t_enqueue_a: float | None = None
    t_start_a: float | None = None
    t_end_a: float | None = None
    t_result_a: float | None = None

    routed: bool = False
    route_reason: str = ""

    b_worker: str | None = None
    b_error: str | None = None
    b_scale: float = 1.0
    detections: list[dict] = field(default_factory=list)
    t_route: float | None = None
    t_enqueue_b: float | None = None
    t_start_b: float | None = None
    t_end_b: float | None = None
    t_result_b: float | None = None

    t_final: float | None = None
    t_written: float | None = None
    analytics: dict | None = None   # tracking / counting result (writer stage, in frame order)

    @property
    def is_failure(self) -> bool:
        return self.a_status in A_FAILURES or self.b_status in B_FAILURES

    @property
    def overall_status(self) -> str:
        if self.a_status == AStatus.INVALID:
            return "INVALID"
        if self.a_status == AStatus.DROPPED:
            return "DROPPED"
        if self.a_status == AStatus.ABORTED or self.b_status == BStatus.ABORTED:
            return "INCOMPLETE"
        if self.is_failure:
            return "ERROR"
        return "OK"
