"""Result aggregation: routing, frame association, deadlines and in-order release.

This class is deliberately single-threaded and free of I/O so it can be unit-tested
exhaustively. The pipeline's main loop feeds it events; it decides what each frame's
final state is and releases finished frames strictly in frame_id order.

Association guarantees:
  * a result is applied only to the record with the same frame_id and expected stage;
  * a result for a frame that already has a final state for that stage (e.g. it timed
    out) is discarded and counted as late - it can never overwrite another outcome;
  * results that arrive before the frame's ingest event (possible across processes)
    are parked and applied once the frame is known.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable, Iterable

from .routing import should_route
from .schema import AStatus, BStatus, FrameRecord, SKIP_REASON, STRIDE_REASON, Stage, WorkerResult

log = logging.getLogger(__name__)


class Aggregator:
    def __init__(self, route_labels: Iterable[str], threshold: float, timeout_a_s: float,
                 timeout_b_s: float, clock: Callable[[], float] = time.perf_counter, detect_every: int = 1):
        self.route_labels = tuple(route_labels)
        self.detect_every = max(1, detect_every)  # Model B on every Nth routed frame (the rest: SKIPPED/stride)
        self.threshold = threshold
        self.timeout_a_s = timeout_a_s
        self.timeout_b_s = timeout_b_s
        self.clock = clock

        self.records: dict[int, FrameRecord] = {}
        self.pending_b: deque[int] = deque()      # routed, waiting for space in queue B
        self.next_out = 0
        self.total_frames: int | None = None
        self._orphans: dict[int, list[WorkerResult]] = {}

        self.late_results = 0
        self.unexpected_results = 0
        self.released = 0

    # ------------------------------------------------------------------ ingest
    def on_ingest(self, frame_id: int, t_ingest: float, t_enqueue_a: float | None = None,
                  a_status: AStatus = AStatus.PENDING, reason: str = "") -> None:
        if frame_id in self.records:
            raise ValueError(f"duplicate frame_id {frame_id}")
        rec = FrameRecord(frame_id=frame_id, t_ingest=t_ingest, t_enqueue_a=t_enqueue_a)
        self.records[frame_id] = rec
        if a_status == AStatus.HELD:      # fast mode: not inferred by design, boxes held by the tracker
            rec.a_status, rec.b_status, rec.route_reason = a_status, BStatus.SKIPPED, reason
            self._finalize(rec)
        elif a_status == AStatus.NOT_GATED:  # fast mode without gate: routed to Model B directly
            rec.a_status, rec.routed, rec.route_reason = a_status, True, reason
            rec.t_route = self.clock()
            self.pending_b.append(frame_id)
        elif a_status != AStatus.PENDING:   # DROPPED / INVALID: final immediately, never routed
            rec.a_status = a_status
            rec.a_error = reason or None
            rec.b_status = BStatus.NOT_RUN
            rec.route_reason = reason
            self._finalize(rec)
        for res in self._orphans.pop(frame_id, []):
            self.on_result(res)

    def set_total(self, n: int) -> None:
        self.total_frames = n

    # ------------------------------------------------------------------ results
    def on_result(self, res: WorkerResult) -> bool:
        """Apply a worker result. Returns True if the frame was routed to Model B."""
        rec = self.records.get(res.frame_id)
        if rec is None:
            if self.total_frames is not None and res.frame_id >= self.total_frames:
                self.unexpected_results += 1
                log.error("result for unknown frame %d discarded", res.frame_id)
            else:
                self._orphans.setdefault(res.frame_id, []).append(res)
            return False
        now = self.clock()
        if res.stage == Stage.A:
            return self._on_a(rec, res, now)
        self._on_b(rec, res, now)
        return False

    def _on_a(self, rec: FrameRecord, res: WorkerResult, now: float) -> bool:
        if rec.a_status != AStatus.PENDING:
            self.late_results += 1
            log.warning("late Model A result for frame %d discarded (already %s)", rec.frame_id, rec.a_status.value)
            return False
        rec.a_worker, rec.t_start_a, rec.t_end_a, rec.t_result_a = res.worker_id, res.t_start, res.t_end, now
        if res.stale:
            rec.a_status, rec.a_error = AStatus.STALE, res.error
        elif not res.ok:
            rec.a_status, rec.a_error = AStatus.ERROR, res.error
        else:
            out = res.output or {}
            rec.a_status = AStatus.OK
            rec.a_label, rec.a_conf, rec.a_top5 = out.get("label"), out.get("conf"), out.get("top5")
        if rec.a_status != AStatus.OK:
            rec.b_status = BStatus.NOT_RUN
            rec.route_reason = f"Model A {rec.a_status.value}"
            self._finalize(rec)
            return False
        routed, reason = should_route(rec.a_label, rec.a_conf, self.route_labels, self.threshold)
        rec.routed, rec.route_reason = routed, reason
        if routed and rec.frame_id % self.detect_every:
            rec.routed = routed = False
            rec.route_reason = f"{STRIDE_REASON}: Model B runs every {self.detect_every} frames"
            rec.b_status = BStatus.SKIPPED
            self._finalize(rec)
        elif routed:
            rec.t_route = now
            self.pending_b.append(rec.frame_id)
        else:
            rec.b_status = BStatus.SKIPPED
            rec.route_reason = SKIP_REASON
            self._finalize(rec)
        return routed

    def _on_b(self, rec: FrameRecord, res: WorkerResult, now: float) -> None:
        if not rec.routed:
            self.unexpected_results += 1
            log.error("Model B result for non-routed frame %d discarded", rec.frame_id)
            return
        if rec.b_status != BStatus.PENDING:
            self.late_results += 1
            log.warning("late Model B result for frame %d discarded (already %s)", rec.frame_id, rec.b_status.value)
            return
        rec.b_worker, rec.t_start_b, rec.t_end_b, rec.t_result_b = res.worker_id, res.t_start, res.t_end, now
        if res.stale:
            rec.b_status, rec.b_error = BStatus.ERROR, res.error
        elif not res.ok:
            rec.b_status, rec.b_error = BStatus.ERROR, res.error
        else:
            dets = []
            s = rec.b_scale
            for d in res.output or []:
                x1, y1, x2, y2 = d["box"]
                dets.append({"label": d["label"], "cls": d.get("cls", 0), "conf": d["conf"],
                             "box": [x1 * s, y1 * s, x2 * s, y2 * s]})
            rec.detections = dets
            rec.b_status = BStatus.DETECTIONS if dets else BStatus.NO_DETECTIONS
        self._finalize(rec)

    def mark_b_dispatched(self, frame_id: int, t_enqueue: float, scale: float) -> None:
        rec = self.records[frame_id]
        rec.t_enqueue_b = t_enqueue
        rec.b_scale = scale

    # ------------------------------------------------------------------ failures
    def on_lost(self, stage: Stage, frame_id: int, reason: str) -> None:
        """A worker died while processing frame_id."""
        rec = self.records.get(frame_id)
        if rec is None or rec.final:
            return
        if stage == Stage.A and rec.a_status == AStatus.PENDING:
            rec.a_status, rec.a_error = AStatus.CRASHED, reason
            rec.b_status, rec.route_reason = BStatus.NOT_RUN, "Model A CRASHED"
            self._finalize(rec)
        elif stage == Stage.B and rec.routed and rec.b_status == BStatus.PENDING:
            rec.b_status, rec.b_error = BStatus.CRASHED, reason
            self._finalize(rec)

    def check_timeouts(self, now: float | None = None) -> int:
        now = self.clock() if now is None else now
        n = 0
        for rec in self.records.values():
            if rec.final:
                continue
            if rec.a_status == AStatus.PENDING and rec.t_enqueue_a is not None and now - rec.t_enqueue_a > self.timeout_a_s:
                rec.a_status = AStatus.TIMEOUT
                rec.a_error = f"no Model A result within {self.timeout_a_s:.1f}s"
                rec.b_status, rec.route_reason = BStatus.NOT_RUN, "Model A TIMEOUT"
                self._finalize(rec)
                n += 1
            elif rec.routed and rec.b_status == BStatus.PENDING and rec.t_enqueue_b is not None \
                    and now - rec.t_enqueue_b > self.timeout_b_s:
                rec.b_status = BStatus.TIMEOUT
                rec.b_error = f"no Model B result within {self.timeout_b_s:.1f}s"
                self._finalize(rec)
                n += 1
        return n

    def abort_pending(self, reason: str = "application stopped") -> int:
        """Shutdown: give every unfinished frame an explicit INCOMPLETE state."""
        n = 0
        for rec in self.records.values():
            if rec.final:
                continue
            if rec.a_status == AStatus.PENDING:
                rec.a_status, rec.a_error = AStatus.ABORTED, reason
                rec.b_status = BStatus.NOT_RUN
            else:
                rec.b_status, rec.b_error = BStatus.ABORTED, reason
            self._finalize(rec)
            n += 1
        self.pending_b.clear()
        if self.records:
            self.total_frames = max(self.records) + 1
        else:
            self.total_frames = 0
        return n

    # ------------------------------------------------------------------ output
    def _finalize(self, rec: FrameRecord) -> None:
        rec.final = True
        rec.t_final = self.clock()

    def pop_ready(self) -> list[FrameRecord]:
        """Release consecutive finished frames starting at next_out (reorder buffer)."""
        out = []
        while True:
            rec = self.records.get(self.next_out)
            if rec is None or not rec.final:
                break
            out.append(self.records.pop(self.next_out))
            self.next_out += 1
        self.released += len(out)
        return out

    @property
    def reorder_depth(self) -> int:
        """Finished frames waiting for an earlier frame (out-of-order completions)."""
        return sum(1 for r in self.records.values() if r.final)

    @property
    def in_flight(self) -> int:
        return len(self.records)

    def done(self) -> bool:
        return self.total_frames is not None and self.next_out >= self.total_frames
