"""Worker process entry point (one model instance per process).

A worker never decides anything about routing or ordering: it pulls a Task, runs the
model, and returns a WorkerResult tagged with the same frame_id. Exceptions are caught
and returned as error results; they never kill the worker.
"""

from __future__ import annotations

import os
import random
import signal
import time
from dataclasses import dataclass

from .config import FaultConfig, ModelSpec
from .schema import Stage, WorkerFailed, WorkerReady, WorkerResult


class InjectedFault(RuntimeError):
    pass


@dataclass
class WorkerArgs:
    stage: Stage
    worker_id: str
    model_spec: ModelSpec
    num_threads: int
    faults: FaultConfig
    generation: int = 0       # 0 = first incarnation, >0 = respawned after a crash
    stale_after_s: float = 0.0


class FaultInjector:
    def __init__(self, cfg: FaultConfig, generation: int, worker_id: str):
        self.cfg = cfg
        self.generation = generation
        self._jitter = random.Random(f"{cfg.seed}-{worker_id}-{generation}")

    def before_predict(self, frame_id: int, processed: int) -> None:
        c = self.cfg
        if c.crash_after and self.generation == 0 and processed >= c.crash_after:
            time.sleep(0.1)  # crash part-way through inference, after earlier results were flushed
            os._exit(13)  # simulate a hard crash (segfault / OOM kill): no cleanup, no result
        if c.delay_every and c.delay_s and frame_id % c.delay_every == 0:
            time.sleep(c.delay_s)
        if c.jitter_s:
            time.sleep(self._jitter.uniform(0.0, c.jitter_s))
        if c.error_every and frame_id % c.error_every == 0:
            raise InjectedFault(f"injected failure on frame {frame_id}")
        if c.error_rate and random.Random(c.seed * 1_000_003 + frame_id).random() < c.error_rate:
            raise InjectedFault(f"injected random failure on frame {frame_id}")


def _configure_threads(n: int) -> None:
    # idle OpenMP threads sleep instead of spin-waiting: spinning workers kept every core at 100% while
    # their queues were empty, starving the main process (measured: tree CPU 587% -> 237%)
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    os.environ.setdefault("KMP_BLOCKTIME", "0")
    if n > 0:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[var] = str(n)
    try:
        import cv2

        cv2.setNumThreads(1)
    except Exception:
        pass
    if n > 0:
        try:
            import torch

            torch.set_num_threads(n)
            torch.set_num_interop_threads(1)
        except Exception:
            pass


def worker_main(args: WorkerArgs, task_q, result_q, current_frame) -> None:
    # Ctrl+C is delivered to every process in the console group; only the main process
    # handles it and shuts workers down via poison pills.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    t0 = time.perf_counter()
    try:
        _configure_threads(args.num_threads)
        from .models import build_model

        model = build_model(args.stage, args.model_spec)
        model.warmup()
    except Exception as e:  # model failed to load: report and exit
        result_q.put(WorkerFailed(args.stage, args.worker_id, f"{type(e).__name__}: {e}"))
        return
    result_q.put(WorkerReady(args.stage, args.worker_id, os.getpid(), dict(model.names), time.perf_counter() - t0))

    faults = FaultInjector(args.faults, args.generation, args.worker_id)
    processed = 0
    while True:
        task = task_q.get()
        if task is None:  # poison pill
            break
        current_frame.value = task.frame_id
        t_start = time.perf_counter()
        res = WorkerResult(args.stage, task.frame_id, args.worker_id, ok=False,
                           t_enqueue=task.t_enqueue, t_start=t_start)
        if args.stale_after_s and t_start - task.t_ingest > args.stale_after_s:
            res.stale = True
            res.error = f"stale: {t_start - task.t_ingest:.2f}s old > {args.stale_after_s:.2f}s"
        else:
            try:
                faults.before_predict(task.frame_id, processed)
                res.output = model.predict(task.image)
                res.ok = True
            except Exception as e:
                res.error = f"{type(e).__name__}: {e}"
        res.t_end = time.perf_counter()
        processed += 1
        result_q.put(res)
        current_frame.value = -1
