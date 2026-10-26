"""Worker pool: N identical worker processes sharing one bounded input queue."""

from __future__ import annotations

import logging
import queue
import time

from .config import FaultConfig, ModelSpec
from .schema import Stage
from .workers import WorkerArgs, worker_main

log = logging.getLogger(__name__)


class WorkerPool:
    """Owns the processes of one stage.

    Each worker publishes the frame_id it is currently processing in a shared integer.
    If a worker process dies, `check_health` reports that frame as lost (so it is marked
    CRASHED instead of being silently dropped or mis-associated) and respawns the worker.
    """

    def __init__(self, ctx, stage: Stage, size: int, task_q, result_q, model_spec: ModelSpec,
                 num_threads: int, faults: FaultConfig, stale_after_s: float = 0.0):
        if size < 1:
            raise ValueError(f"{stage.value} pool needs at least one worker")
        self.ctx = ctx
        self.stage = stage
        self.size = size
        self.task_q = task_q
        self.result_q = result_q
        self.model_spec = model_spec
        self.num_threads = num_threads
        self.faults = faults
        self.stale_after_s = stale_after_s
        self._current = [ctx.Value("q", -1, lock=False) for _ in range(size)]
        self._generation = [0] * size
        self._procs = [None] * size
        self._stopping = False
        self.restarts = 0

    def worker_id(self, idx: int) -> str:
        return f"{self.stage.value}{idx}"

    def _spawn(self, idx: int) -> None:
        args = WorkerArgs(self.stage, self.worker_id(idx), self.model_spec, self.num_threads,
                          self.faults, self._generation[idx], self.stale_after_s)
        p = self.ctx.Process(target=worker_main, args=(args, self.task_q, self.result_q, self._current[idx]),
                             name=f"vpipe-{self.worker_id(idx)}", daemon=True)
        p.start()
        self._procs[idx] = p

    def start(self) -> None:
        for i in range(self.size):
            self._spawn(i)

    @property
    def pids(self) -> list[int]:
        return [p.pid for p in self._procs if p is not None and p.pid]

    def alive_count(self) -> int:
        return sum(1 for p in self._procs if p is not None and p.is_alive())

    def check_health(self) -> list[tuple[int, str]]:
        """Detect dead workers. Returns [(lost_frame_id, reason)] and respawns the workers."""
        lost: list[tuple[int, str]] = []
        if self._stopping:
            return lost
        for i, p in enumerate(self._procs):
            if p is None or p.is_alive():
                continue
            fid = self._current[i].value
            reason = f"worker {self.worker_id(i)} (pid {p.pid}) died with exit code {p.exitcode}"
            log.error("%s; respawning", reason)
            if fid >= 0:
                lost.append((fid, reason))
            self._current[i].value = -1
            self._generation[i] += 1
            self.restarts += 1
            self._spawn(i)
        return lost

    def stop(self, timeout: float = 5.0, drain=None) -> None:
        """Poison-pill shutdown, then terminate stragglers.

        `drain` is called while waiting so workers blocked on a full result pipe can exit.
        """
        self._stopping = True
        deadline = time.monotonic() + timeout
        for _ in self._procs:
            try:
                self.task_q.put(None, timeout=max(0.05, deadline - time.monotonic()))
            except queue.Full:
                break
        while time.monotonic() < deadline and any(p is not None and p.is_alive() for p in self._procs):
            if drain:
                drain()
            time.sleep(0.02)
        for p in self._procs:
            if p is not None and p.is_alive():
                log.warning("terminating %s (did not exit in %.1fs)", p.name, timeout)
                p.terminate()
        for p in self._procs:
            if p is not None:
                p.join(timeout=2.0)
