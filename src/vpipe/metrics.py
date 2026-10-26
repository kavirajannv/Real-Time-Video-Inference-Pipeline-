"""Runtime sampling (queue depth, CPU, memory) and end-of-run summary statistics."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import numpy as np
import psutil


def pct(values, ps=(50, 95, 99)) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0}
    arr = np.asarray(vals, dtype=float)
    out = {"n": len(vals), "mean": round(float(arr.mean()), 2), "min": round(float(arr.min()), 2),
           "max": round(float(arr.max()), 2)}
    for p in ps:
        # p99 needs ~100 samples to mean anything; report it anyway but flag small n
        out[f"p{p}"] = round(float(np.percentile(arr, p)), 2)
    return out


class Sampler(threading.Thread):
    """Periodically records queue depths and process-tree CPU / memory."""

    def __init__(self, probes: dict[str, Callable[[], int]], pids: Callable[[], list[int]], interval_s: float = 0.1):
        super().__init__(name="vpipe-sampler", daemon=True)
        self.probes = probes
        self.pids = pids
        self.interval_s = interval_s
        self.samples: list[dict] = []
        self._halt = threading.Event()
        self._procs: dict[int, psutil.Process] = {}
        psutil.cpu_percent(None)

    def _proc(self, pid: int):
        p = self._procs.get(pid)
        if p is None:
            try:
                p = psutil.Process(pid)
                p.cpu_percent(None)  # prime
                self._procs[pid] = p
            except psutil.Error:
                return None
        return p

    def run(self) -> None:
        last_cpu = time.monotonic()
        cpu_tree, rss_tree = 0.0, 0.0
        while not self._halt.wait(self.interval_s):
            s = {"t": time.perf_counter()}
            for name, fn in self.probes.items():
                try:
                    s[name] = int(fn())
                except (NotImplementedError, OSError, ValueError):
                    s[name] = -1
            # CPU percent needs a reasonable window; refresh process stats every 0.5 s
            if time.monotonic() - last_cpu >= 0.5:
                last_cpu = time.monotonic()
                cpu_tree, rss_tree = 0.0, 0.0
                for pid in self.pids():
                    p = self._proc(pid)
                    if p is None:
                        continue
                    try:
                        cpu_tree += p.cpu_percent(None)
                        rss_tree += p.memory_info().rss
                    except psutil.Error:
                        self._procs.pop(pid, None)
                s["cpu_system_pct"] = psutil.cpu_percent(None)
                s["sys_mem_used_mb"] = (psutil.virtual_memory().total - psutil.virtual_memory().available) / 2**20
            s["cpu_tree_pct"] = cpu_tree
            s["rss_tree_mb"] = rss_tree / 2**20
            self.samples.append(s)

    def stop(self) -> None:
        self._halt.set()
        self.join(timeout=2.0)


def summarize(rows: list[dict], samples: list[dict], *, wall_s: float, startup_s: float, source_fps: float,
              ingest_span_s: float, extra: dict | None = None) -> dict:
    n = len(rows)
    count = lambda pred: sum(1 for r in rows if pred(r))  # noqa: E731
    routed = count(lambda r: r["routed"])
    dropped = count(lambda r: r["a_status"] == "DROPPED")
    a_ok = count(lambda r: r["a_status"] == "OK")
    summary = {
        "frames_total": n,
        "frames_classified_ok": a_ok,
        "frames_routed_to_b": routed,
        "frames_skipped": count(lambda r: r["b_status"] == "SKIPPED"),
        "frames_with_detections": count(lambda r: r["b_status"] == "DETECTIONS"),
        "frames_no_detections": count(lambda r: r["b_status"] == "NO_DETECTIONS"),
        "frames_dropped": dropped,
        "drop_rate": round(dropped / n, 4) if n else 0.0,
        "frames_invalid": count(lambda r: r["a_status"] == "INVALID"),
        "frames_held": count(lambda r: r["a_status"] == "HELD"),
        "frames_not_gated": count(lambda r: r["a_status"] == "NOT_GATED"),
        "a_errors": count(lambda r: r["a_status"] in ("ERROR", "CRASHED", "STALE")),
        "a_timeouts": count(lambda r: r["a_status"] == "TIMEOUT"),
        "b_errors": count(lambda r: r["b_status"] in ("ERROR", "CRASHED")),
        "b_timeouts": count(lambda r: r["b_status"] == "TIMEOUT"),
        "frames_incomplete": count(lambda r: r["status"] == "INCOMPLETE"),
        "source_fps": round(source_fps, 2),
        "input_fps": round(n / ingest_span_s, 2) if ingest_span_s > 0 else None,
        "output_fps": round(n / wall_s, 2) if wall_s > 0 else None,
        "a_throughput_fps": round(a_ok / wall_s, 2) if wall_s > 0 else None,
        "wall_time_s": round(wall_s, 3),
        "startup_s": round(startup_s, 3),
        "latency_ms": {
            "a_infer": pct(r["a_infer_ms"] for r in rows),
            "a_queue_wait": pct(r["a_queue_wait_ms"] for r in rows),
            "b_infer": pct(r["b_infer_ms"] for r in rows),
            "b_route_wait": pct(r["b_route_wait_ms"] for r in rows),
            "b_queue_wait": pct(r["b_queue_wait_ms"] for r in rows),
            "result": pct(r["result_latency_ms"] for r in rows if r["a_status"] not in ("DROPPED", "INVALID")),
            "end_to_end": pct(r["e2e_latency_ms"] for r in rows if r["a_status"] not in ("DROPPED", "INVALID")),
        },
    }
    qs = {}
    if samples:
        for key in samples[0]:
            if key == "t":
                continue
            vals = [s[key] for s in samples if key in s and s[key] is not None and s[key] >= 0]
            if vals:
                qs[key] = {"mean": round(float(np.mean(vals)), 2), "max": round(float(np.max(vals)), 2)}
    summary["runtime"] = qs
    if extra:
        summary.update(extra)
    return summary
