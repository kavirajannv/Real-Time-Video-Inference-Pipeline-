"""Pipeline orchestration.

    Reader thread ──► Queue A (bounded, mp) ──► Model A pool (N processes) ─┐
                                                                            ▼
    Writer thread ◄── reorder buffer ◄── Aggregator / router (main thread) ◄─ Results queue
                                              │  routed frames only
                                              ▼
                              Queue B (bounded, mp) ──► Model B pool (M processes) ──► Results queue

The main thread owns all mutable per-frame state (Aggregator), so no locks are needed
around routing / association logic.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import queue
import threading
import time
from pathlib import Path

import psutil

from .aggregator import Aggregator
from .config import PipelineConfig
from .metrics import Sampler, summarize
from .pools import WorkerPool
from .reader import EndOfStream, FrameStore, IngestEvent, Reader, VideoSource, prep_for_b
from .routing import verify_route_labels
from .schema import Stage, Task, WorkerFailed, WorkerReady, WorkerResult
from .writer import OutputWriter

log = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    pass


class Pipeline:
    def __init__(self, cfg: PipelineConfig, source=None):
        self.cfg = cfg
        self.source = source
        self.stop_event = threading.Event()
        self.summary: dict | None = None

    def request_stop(self) -> None:
        self.stop_event.set()

    # ------------------------------------------------------------------ setup helpers
    def _wait_ready(self, results, expected: int) -> dict:
        ready: dict[str, WorkerReady] = {}
        deadline = time.monotonic() + self.cfg.startup_timeout_s
        while len(ready) < expected:
            if self.stop_event.is_set():
                raise PipelineError("stopped during startup")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PipelineError(f"only {len(ready)}/{expected} workers ready after {self.cfg.startup_timeout_s}s")
            try:
                msg = results.get(timeout=min(0.5, remaining))
            except queue.Empty:
                for pool in (self.pool_a, self.pool_b):
                    if pool.alive_count() < pool.size and len(ready) < expected:
                        # a process that died before reporting is a startup failure
                        dead = [p for p in pool._procs if p is not None and not p.is_alive()]
                        if dead and all(p.exitcode not in (None, 0) for p in dead):
                            raise PipelineError(f"{pool.stage.value} worker died during startup")
                continue
            if isinstance(msg, WorkerFailed):
                raise PipelineError(f"worker {msg.worker_id} failed to load model: {msg.error}")
            if isinstance(msg, WorkerReady):
                ready[msg.worker_id] = msg
                log.info("worker %s ready (pid %d, %.1fs)", msg.worker_id, msg.pid, msg.load_s)
        return ready

    def _drain(self, q) -> int:
        n = 0
        while True:
            try:
                q.get_nowait()
                n += 1
            except (queue.Empty, OSError, ValueError):
                return n

    # ------------------------------------------------------------------ output-folder lock
    def _acquire_output_lock(self) -> Path:
        """Two runs writing the same folder interleave their video and log writes and corrupt both
        outputs, so a second run into a folder that is in use is refused."""
        out = Path(self.cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        lock = out / ".vpipe.lock"
        for _ in range(2):
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    pid = int(lock.read_text().strip() or 0)
                except (OSError, ValueError):
                    pid = 0
                if pid and pid != os.getpid() and psutil.pid_exists(pid):
                    raise PipelineError(f"another vpipe run (pid {pid}) is writing to {out}; wait for it to finish "
                                        "or use another --output-dir") from None
                lock.unlink(missing_ok=True)  # left behind by a run that was killed
                continue
            with os.fdopen(fd, "w") as f:
                f.write(str(os.getpid()))
            return lock
        raise PipelineError(f"cannot lock output folder {out}")

    # ------------------------------------------------------------------ main
    def _pick_free_video_name(self) -> None:
        """On Windows a video open in a player cannot be overwritten. Rather than fail (after the
        models have loaded), write next to it under a new name; summary.json records the path."""
        path = self.cfg.output_path
        if not self.cfg.write_video or not path.exists():
            return
        try:
            open(path, "r+b").close()
        except PermissionError:
            self.cfg.output_name = f"{path.stem}_{time.strftime('%Y%m%d_%H%M%S')}{path.suffix}"
            log.warning("%s is open in another program (video player?); writing %s instead", path,
                        self.cfg.output_path)

    def run(self) -> dict:
        lock = self._acquire_output_lock()
        self._pick_free_video_name()
        try:
            return self._run()
        finally:
            lock.unlink(missing_ok=True)

    def _run(self) -> dict:
        cfg = self.cfg
        # resizing 1080p frames (reader, B dispatch, writer) is the main process's biggest cost: give OpenCV
        # a pool the size of the physical cores (workers no longer spin idle threads, see workers.py)
        import cv2

        cv2.setNumThreads(int(os.environ.get("VPIPE_MAIN_CV_THREADS", "0")) or
                          (psutil.cpu_count(logical=False) or 4))
        own_source = self.source is None
        source = self.source or VideoSource(cfg.video, cfg.repeat)
        if cfg.max_seconds > 0:
            n = max(1, round(cfg.max_seconds * (source.fps or 30.0)))
            cfg.max_frames = min(cfg.max_frames, n) if cfg.max_frames else n
        fps_out = source.fps
        live_fps = cfg.live_fps or source.fps
        size = (source.width, source.height)
        log.info("input: %dx%d @ %.2f fps, ~%d frames", source.width, source.height, source.fps, source.frame_count)

        ctx = mp.get_context("spawn")
        q_a = ctx.Queue(maxsize=cfg.queue_a_size)
        q_b = ctx.Queue(maxsize=cfg.queue_b_size)
        results = ctx.Queue()  # bounded implicitly: <= max_buffered_frames * 2 messages in flight
        cores = psutil.cpu_count(logical=False) or psutil.cpu_count() or 1
        threads = cfg.resolved_threads(cores)
        stale = cfg.stale_after_s if cfg.mode == "live" else 0.0
        self.pool_a = WorkerPool(ctx, Stage.A, cfg.a_workers, q_a, results, cfg.model_a, threads, cfg.fault_a, stale)
        self.pool_b = WorkerPool(ctx, Stage.B, cfg.b_workers, q_b, results, cfg.model_b, threads, cfg.fault_b, stale)

        t_boot = time.perf_counter()
        sampler = writer = reader = writer_thread = None
        try:
            self.pool_a.start()
            self.pool_b.start()
            ready = self._wait_ready(results, cfg.a_workers + cfg.b_workers)
            startup_s = time.perf_counter() - t_boot
            names_a = next(r.names for r in ready.values() if r.stage == Stage.A)
            present, missing = verify_route_labels(cfg.route_labels, names_a)

            agg = Aggregator(cfg.route_labels, cfg.route_threshold, cfg.timeout_a_s, cfg.timeout_b_s,
                             detect_every=cfg.detect_every)
            store = FrameStore()
            slots = threading.BoundedSemaphore(cfg.max_buffered_frames)
            events: queue.Queue = queue.Queue()
            writer_q: queue.Queue = queue.Queue()

            t0 = time.perf_counter()
            analytics = None
            if cfg.analytics:
                from .analytics import Analytics

                attributes = None
                if cfg.attributes_model:
                    from .attributes import PersonAttributeModel

                    attributes = PersonAttributeModel(cfg.attributes_model)
                    log.info("person attributes: %s", cfg.attributes_model)
                analytics = Analytics(Path(cfg.output_dir), size, fps_out, cfg.count_line, cfg.min_track_frames,
                                      cfg.scene_cut_threshold, attributes,
                                      detect_every=cfg.detect_every * cfg.infer_every,
                                      background_attributes=True)
            writer = OutputWriter(cfg.output_path, fps_out, size, t0, cfg.write_video, analytics, cfg.show_classes,
                                  cfg.output_height, cfg.video_preset)
            writer_state = {"error": None}

            total = source.frame_count
            if cfg.max_frames:
                total = min(total, cfg.max_frames) if total > 0 else cfg.max_frames
            progress = {"next": time.monotonic() + 5.0, "start": None}

            def report_progress(done: int) -> None:
                # a long run is otherwise silent until the summary: show it is alive and how long is left
                now = time.monotonic()
                if progress["start"] is None:  # measure from the first written frame, not from setup
                    progress["start"] = now
                    progress["next"] = now + 5.0
                    return
                if now < progress["next"] and done != total:
                    return
                progress["next"] = now + 5.0
                fps = done / max(now - progress["start"], 1e-6)
                eta = (total - done) / fps if total > 0 and fps > 0 else 0.0
                pct = f" ({100.0 * done / total:.0f}%)" if total > 0 else ""
                log.info("progress: %d/%s frames%s  %.1f fps  ETA %d:%02d", done, total if total > 0 else "?", pct,
                         fps, eta // 60, eta % 60)

            def writer_loop():
                while True:
                    item = writer_q.get()
                    if item is None:
                        return
                    rec, img = item
                    try:
                        writer.write(rec, img)
                        report_progress(writer.frames_written)
                    except Exception as e:  # noqa: BLE001
                        writer_state["error"] = e
                        log.exception("writer failed")
                        self.stop_event.set()
                    finally:
                        slots.release()

            writer_thread = threading.Thread(target=writer_loop, name="vpipe-writer", daemon=True)
            writer_thread.start()

            def safe_qsize(q):
                try:
                    return q.qsize()
                except NotImplementedError:  # macOS
                    return -1

            sampler = Sampler(
                {
                    "queue_a_depth": lambda: safe_qsize(q_a),
                    "queue_b_depth": lambda: safe_qsize(q_b),
                    "pending_b_depth": lambda: len(agg.pending_b),
                    "results_depth": lambda: safe_qsize(results),
                    "reorder_buffer_depth": lambda: agg.reorder_depth,
                    "frames_in_flight": lambda: agg.in_flight,
                    "frames_buffered": lambda: len(store),
                    "writer_queue_depth": lambda: writer_q.qsize(),
                },
                pids=lambda: [psutil.Process().pid] + self.pool_a.pids + self.pool_b.pids,
                interval_s=cfg.sample_interval_s,
            )
            sampler.start()

            reader = Reader(source, q_a, events, store, slots, self.stop_event, mode=cfg.mode, fps=live_fps,
                            imgsz_a=cfg.model_a.imgsz, resize=cfg.resize_for_ipc, max_frames=cfg.max_frames,
                            invalid_every=cfg.invalid_every, infer_every=cfg.infer_every, gate=cfg.gate,
                            pace=cfg.pace)
            reader.start()

            aborted = self._main_loop(agg, results, q_b, events, store, writer_q, reader, t0)

            writer_q.put(None)
            writer_thread.join()
            t_done = time.perf_counter()
            if writer_state["error"]:
                raise PipelineError(f"writer failed: {writer_state['error']}")
            if reader.error:
                raise PipelineError(f"reader failed: {reader.error}")
        finally:
            if sampler:
                sampler.stop()
            # discard tasks nobody will process so pills get through, then stop workers
            self._drain(q_a)
            self._drain(q_b)
            for pool in (self.pool_a, self.pool_b):
                pool.stop(timeout=5.0, drain=lambda: self._drain(results))
            for q in (q_a, q_b, results):
                q.cancel_join_thread()
                q.close()
            if writer:
                writer.close()
            if own_source:
                source.close()

        ingest_span = (reader.t_last - reader.t_first) if reader.t_first and reader.t_last else 0.0
        if cfg.mode == "live" and ingest_span > 0:
            ingest_span = max(ingest_span, (len(writer.rows) - 1) / live_fps)
        summary = summarize(
            writer.rows, sampler.samples, wall_s=t_done - t0, startup_s=startup_s, source_fps=source.fps,
            ingest_span_s=ingest_span,
            extra={
                "aborted": aborted,
                "config": {
                    "mode": cfg.mode, "a_workers": cfg.a_workers, "b_workers": cfg.b_workers,
                    "torch_threads_per_worker": threads, "physical_cores": cores,
                    "queue_a_size": cfg.queue_a_size, "queue_b_size": cfg.queue_b_size,
                    "max_buffered_frames": cfg.max_buffered_frames, "resize_for_ipc": cfg.resize_for_ipc,
                    "route_labels": list(cfg.route_labels), "route_threshold": cfg.route_threshold,
                    "live_fps": live_fps if cfg.mode == "live" else None,
                    "model_a": cfg.model_a.weights if cfg.model_a.kind == "yolo" else "fake",
                    "model_b": cfg.model_b.weights if cfg.model_b.kind == "yolo" else "fake",
                },
                "route_labels_verified": present,
                "route_labels_missing": missing,
                "reader_blocked_s": round(reader.blocked_s, 3),
                "reader_behind_schedule": reader.behind_schedule,
                "worker_restarts": {"A": self.pool_a.restarts, "B": self.pool_b.restarts},
                "late_results_discarded": agg.late_results,
                "unexpected_results": agg.unexpected_results,
                "output_video": str(cfg.output_path) if cfg.write_video else None,
                "frame_size": list(size),
                "output_fps_setting": fps_out,
                "analytics": writer.analytics_summary,
            },
        )
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(cfg.output_dir) / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with open(Path(cfg.output_dir) / "samples.json", "w", encoding="utf-8") as f:
            t_ref = sampler.samples[0]["t"] if sampler.samples else 0.0
            json.dump([{**s, "t": round(s["t"] - t_ref, 3)} for s in sampler.samples], f)
        self.summary = summary
        return summary

    def _main_loop(self, agg: Aggregator, results, q_b, events, store, writer_q, reader, t0) -> bool:
        cfg = self.cfg
        aborted = False
        last_health = 0.0

        def drain_events():
            while True:
                try:
                    ev = events.get_nowait()
                except queue.Empty:
                    return
                if isinstance(ev, IngestEvent):
                    agg.on_ingest(ev.frame_id, ev.t_ingest, ev.t_enqueue, ev.status, ev.reason)
                elif isinstance(ev, EndOfStream):
                    agg.set_total(ev.total_frames)

        while True:
            try:
                if not aborted and self.stop_event.is_set():
                    log.warning("stop requested: finishing in-flight frames as INCOMPLETE")
                    reader.join(timeout=5.0)
                    drain_events()
                    n = agg.abort_pending()
                    log.warning("%d frames marked INCOMPLETE", n)
                    aborted = True

                drain_events()
                try:
                    msg = results.get(timeout=0.02)
                except queue.Empty:
                    msg = None
                while msg is not None:
                    drain_events()  # make sure the frame's ingest record exists first
                    self._handle(msg, agg)
                    try:
                        msg = results.get_nowait()
                    except queue.Empty:
                        msg = None

                now = time.perf_counter()
                if now - last_health > 0.2:
                    last_health = now
                    for pool in (self.pool_a, self.pool_b):
                        for fid, reason in pool.check_health():
                            agg.on_lost(pool.stage, fid, reason)
                agg.check_timeouts(now)
                if not aborted:
                    self._dispatch_b(agg, q_b, store)

                for rec in agg.pop_ready():
                    writer_q.put((rec, store.pop(rec.frame_id)))
                if agg.done():
                    return aborted
                if cfg.stop_after_s and now - t0 > cfg.stop_after_s:
                    self.stop_event.set()
            except KeyboardInterrupt:
                log.warning("Ctrl+C received")
                self.stop_event.set()

    def _handle(self, msg, agg: Aggregator) -> None:
        if isinstance(msg, WorkerResult):
            agg.on_result(msg)
        elif isinstance(msg, WorkerReady):
            log.info("worker %s (re)started, pid %d", msg.worker_id, msg.pid)
        elif isinstance(msg, WorkerFailed):
            log.error("worker %s failed: %s", msg.worker_id, msg.error)

    def _dispatch_b(self, agg: Aggregator, q_b, store: FrameStore) -> None:
        """Move routed frames into queue B without ever blocking the main loop."""
        while agg.pending_b:
            fid = agg.pending_b[0]
            rec = agg.records.get(fid)
            if rec is None or rec.final:
                agg.pending_b.popleft()
                continue
            if q_b.full():
                return
            img, scale = prep_for_b(store.get(fid), self.cfg.model_b.imgsz, self.cfg.resize_for_ipc)
            t_enq = time.perf_counter()
            try:
                q_b.put_nowait(Task(Stage.B, fid, img, rec.t_ingest, t_enq))
            except queue.Full:
                return
            agg.mark_b_dispatched(fid, t_enq, scale)
            agg.pending_b.popleft()
