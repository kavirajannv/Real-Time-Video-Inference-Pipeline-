"""Benchmark worker configurations, the optimisation, and a 30 FPS live replay.

Each configuration runs in a fresh process (`python -m vpipe run ...`) so CPU / memory
numbers are not polluted by earlier runs. Results: outputs/bench/results.json and
outputs/bench/benchmark_tables.md.

    python scripts/benchmark.py                 # full suite (~10 min on a 4-core laptop)
    python scripts/benchmark.py --only 1A1B 2A2B
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "bench"

# yolo11n-cls never reaches 0.30 on any frame of this video and ImageNet has no "person"
# class, so the strict rule routes nothing. To load Model B realistically, benchmarks route
# on labels the classifier actually predicts for this scene (~50% of frames). See docs.
DEMO_ROUTING = ["--route-labels", "puck,ski", "--route-threshold", "0.10"]
STRICT_ROUTING = ["--route-labels", "person", "--route-threshold", "0.30"]

SUITE: list[tuple[str, str, list[str]]] = [
    # name, description, extra args
    ("1A1B", "1 Model A worker, 1 Model B worker", ["--a-workers", "1", "--b-workers", "1"]),
    ("2A1B", "2 Model A workers, 1 Model B worker", ["--a-workers", "2", "--b-workers", "1"]),
    ("1A2B", "1 Model A worker, 2 Model B workers", ["--a-workers", "1", "--b-workers", "2"]),
    ("2A2B", "2 Model A workers, 2 Model B workers", ["--a-workers", "2", "--b-workers", "2"]),
    ("1A3B", "1 Model A worker, 3 Model B workers", ["--a-workers", "1", "--b-workers", "3"]),
    ("3A3B", "3 Model A workers, 3 Model B workers", ["--a-workers", "3", "--b-workers", "3"]),
    # optimisation study (2A2B): baseline = every worker uses torch's default thread pool
    # (all cores -> oversubscribed) and receives full-resolution frames over IPC
    ("2A2B-baseline", "2A2B, torch default threads, full frames over IPC",
     ["--a-workers", "2", "--b-workers", "2", "--torch-threads", "-1", "--no-resize-ipc"]),
    ("2A2B-threads-only", "2A2B, tuned threads, full frames over IPC",
     ["--a-workers", "2", "--b-workers", "2", "--no-resize-ipc"]),
    # controlled thread study: identical inputs/routing (150 frames routed), only thread budget differs
    ("2A2B-default-threads", "2A2B, torch default threads, resized IPC (isolates thread tuning)",
     ["--a-workers", "2", "--b-workers", "2", "--torch-threads", "-1"]),
    ("1A1B-default-threads", "1A1B, torch default threads, resized IPC (isolates thread tuning)",
     ["--a-workers", "1", "--b-workers", "1", "--torch-threads", "-1"]),
    ("1A1B-baseline", "1A1B, torch default threads, full frames over IPC",
     ["--a-workers", "1", "--b-workers", "1", "--torch-threads", "-1", "--no-resize-ipc"]),
    # strict spec rule (routes nothing on this video): Model A only cost
    ("2A2B-strict", "2A2B with the strict spec rule (person >= 0.30)",
     ["--a-workers", "2", "--b-workers", "2", *STRICT_ROUTING]),
    # 30 FPS real-time replay: paced ingest, small queues, stale frames dropped
    ("live30-1A2B", "LIVE 30 FPS replay, 1A2B, queue A=2, stale>0.5s dropped",
     ["--a-workers", "1", "--b-workers", "2", "--mode", "live", "--live-fps", "30", "--queue-a", "2",
      "--queue-b", "2", "--stale-after", "0.5"]),
    ("live30-2A2B", "LIVE 30 FPS replay, 2A2B, queue A=2, stale>0.5s dropped",
     ["--a-workers", "2", "--b-workers", "2", "--mode", "live", "--live-fps", "30", "--queue-a", "2",
      "--queue-b", "2", "--stale-after", "0.5"]),
    ("live30-strict-2A1B", "LIVE 30 FPS replay, strict rule (Model A only), 2A1B",
     ["--a-workers", "2", "--b-workers", "1", "--mode", "live", "--live-fps", "30", "--queue-a", "2",
      "--stale-after", "0.5", *STRICT_ROUTING]),
]


def run_one(name: str, args: list[str], repeat: int) -> dict:
    out_dir = OUT / name
    routing = [] if any(a == "--route-labels" for a in args) else DEMO_ROUTING
    cmd = [sys.executable, "-m", "vpipe", "run", "--output-dir", str(out_dir), "--repeat", str(repeat),
           *routing, *args]
    print(f"\n=== {name}: {' '.join(cmd[3:])}", flush=True)
    t = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-3000:], proc.stderr[-3000:])
        raise SystemExit(f"{name} failed")
    summary = json.loads((out_dir / "summary.json").read_text())
    summary["process_total_s"] = round(time.perf_counter() - t, 2)
    s = summary
    print(f"    frames={s['frames_total']} routed={s['frames_routed_to_b']} output_fps={s['output_fps']} "
          f"wall={s['wall_time_s']}s e2e_p50={s['latency_ms']['end_to_end'].get('p50')}ms "
          f"dropped={s['frames_dropped']}", flush=True)
    return summary


def g(d: dict, *keys, default="-"):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def tables(results: dict[str, dict], descr: dict[str, str]) -> str:
    lines = []
    hdr = ("| config | frames | routed | det | dropped | wall s | output FPS | A infer p50/p95 ms | "
           "B infer p50/p95 ms | A q-wait p50 ms | B q-wait p50 ms | E2E p50 / p95 / p99 ms | CPU mean/max % | "
           "RSS max MB | threads/worker |")
    lines += ["### Throughput, latency and resources", "", hdr, "|" + "---|" * (hdr.count("|") - 1)]
    for name, s in results.items():
        L = s["latency_ms"]
        rt = s.get("runtime", {})
        lines.append(
            f"| **{name}** | {s['frames_total']} | {s['frames_routed_to_b']} | {s['frames_with_detections']} | "
            f"{s['frames_dropped']} ({s['drop_rate']:.0%}) | {s['wall_time_s']} | {s['output_fps']} | "
            f"{g(L, 'a_infer', 'p50')} / {g(L, 'a_infer', 'p95')} | {g(L, 'b_infer', 'p50')} / {g(L, 'b_infer', 'p95')} | "
            f"{g(L, 'a_queue_wait', 'p50')} | {g(L, 'b_queue_wait', 'p50')} | "
            f"{g(L, 'end_to_end', 'p50')} / {g(L, 'end_to_end', 'p95')} / {g(L, 'end_to_end', 'p99')} | "
            f"{g(rt, 'cpu_tree_pct', 'mean')} / {g(rt, 'cpu_tree_pct', 'max')} | {g(rt, 'rss_tree_mb', 'max')} | "
            f"{s['config']['torch_threads_per_worker']} |")
    lines += ["", "### Queue depth (mean / max) and reader backpressure", "",
              "| config | queue A | queue B | pending B (router) | reorder buffer | frames buffered | reader blocked s | "
              "startup s |", "|---|---|---|---|---|---|---|---|"]
    for name, s in results.items():
        rt = s.get("runtime", {})
        q = lambda k: f"{g(rt, k, 'mean')} / {g(rt, k, 'max')}"  # noqa: E731
        lines.append(f"| {name} | {q('queue_a_depth')} | {q('queue_b_depth')} | {q('pending_b_depth')} | "
                     f"{q('reorder_buffer_depth')} | {q('frames_buffered')} | {s['reader_blocked_s']} | "
                     f"{s['startup_s']} |")
    lines += ["", "### Configurations", ""] + [f"- **{k}**: {descr.get(k, '')}" for k in results]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5, help="replay the 62-frame video N times per run")
    ap.add_argument("--only", nargs="*", help="subset of config names")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    results_path = OUT / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() and a.only else {}
    descr = {n: d for n, d, _ in SUITE}
    for name, _, args in SUITE:
        if a.only and name not in a.only:
            continue
        results[name] = run_one(name, args, a.repeat)
        results_path.write_text(json.dumps(results, indent=2))
    ordered = {n: results[n] for n, _, _ in SUITE if n in results}
    (OUT / "benchmark_tables.md").write_text(tables(ordered, descr), encoding="utf-8")
    print(f"\nwrote {results_path} and {OUT / 'benchmark_tables.md'}")


if __name__ == "__main__":
    main()
