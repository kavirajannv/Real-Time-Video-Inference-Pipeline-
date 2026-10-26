"""Failure-behaviour demonstration on the real video with real models (or --fake).

Every scenario runs the full pipeline with fault injection and then verifies invariants:
  * the output video and the log contain every ingested frame, in frame order
  * queue depths never exceed their configured bounds
  * failed / incomplete frames carry an explicit status (never a fake success)

    python scripts/failure_demo.py            # real YOLO models (~3 min)
    python scripts/failure_demo.py --fake     # fake models (~1 min)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "failure_demo"
ROUTING = ["--route-labels", "puck,ski", "--route-threshold", "0.10"]

SCENARIOS = [
    ("model_a_slow", "Model A takes longer than expected (2 s on every 20th frame, timeout 1 s)",
     ["--a-workers", "2", "--inject-delay-a", "2.0", "--inject-delay-every-a", "20", "--timeout-a", "1.0"]),
    ("model_b_slow", "Model B takes longer than expected (2 s on every 4th frame id, timeout 1 s)",
     ["--b-workers", "2", "--inject-delay-b", "2.0", "--inject-delay-every-b", "4", "--timeout-b", "1.0"]),
    ("worker_exception", "Workers raise exceptions (A on every 15th frame, B on every 4th)",
     ["--inject-error-every-a", "15", "--inject-error-every-b", "4"]),
    ("worker_crash", "Model B worker process hard-crashes after 5 tasks (respawned)",
     ["--b-workers", "1", "--inject-crash-after-b", "5"]),
    ("queue_full_offline", "Queues at capacity, offline: queue A=1, queue B=1, 4 buffered frames -> backpressure",
     ["--a-workers", "1", "--b-workers", "1", "--queue-a", "1", "--queue-b", "1", "--max-buffered", "4"]),
    ("queue_full_live", "Queues at capacity, live 30 FPS with queue A=1 -> documented drop policy",
     ["--a-workers", "1", "--b-workers", "1", "--mode", "live", "--live-fps", "30", "--queue-a", "1",
      "--queue-b", "1"]),
    ("out_of_order", "Results complete out of order (random 0-0.4 s jitter on 3+3 workers)",
     ["--a-workers", "3", "--b-workers", "3", "--inject-jitter-a", "0.4", "--inject-jitter-b", "0.4"]),
    ("video_ends_in_flight", "Video ends while frames are in flight (every B call +0.5 s)",
     ["--b-workers", "1", "--inject-delay-b", "0.5"]),
    ("invalid_frames", "Corrupt frames in the source (every 25th)",
     ["--inject-invalid-every", "25"]),
    ("stopped_mid_run", "Application stopped during processing (operator stop after 4 s, 5x video)",
     ["--repeat", "5", "--stop-after", "4", "--b-workers", "1", "--inject-delay-b", "0.2"]),
]


def video_frames(path: Path) -> tuple[int, tuple | None]:
    cap = cv2.VideoCapture(str(path))
    n, shape = 0, None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        n, shape = n + 1, f.shape
    cap.release()
    return n, shape


def run(name: str, args: list[str], fake: bool) -> dict:
    out_dir = OUT / name
    cmd = [sys.executable, "-m", "vpipe", "run", "--output-dir", str(out_dir), *ROUTING, *args]
    if fake:
        cmd += ["--fake-models", "--fake-infer-ms", "30"]
    t = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
    elapsed = time.perf_counter() - t
    if proc.returncode != 0:
        print(proc.stderr[-3000:])
        return {"name": name, "ok": False, "error": f"exit code {proc.returncode}"}
    s = json.loads((out_dir / "summary.json").read_text())
    rows = json.loads((out_dir / "frames_log.json").read_text())
    n_video, shape = video_frames(out_dir / "output_annotated.mp4")
    ids = [r["frame_id"] for r in rows]
    qa, qb = int(s["config"]["queue_a_size"]), int(s["config"]["queue_b_size"])
    rt = s["runtime"]
    checks = {
        "log has every ingested frame, in order": ids == list(range(len(rows))),
        "video frame count == log rows": n_video == len(rows),
        "video size preserved (640x360)": shape is not None and shape[:2] == (360, 640),
        f"queue A depth <= {qa}": rt.get("queue_a_depth", {}).get("max", 0) <= qa,
        f"queue B depth <= {qb}": rt.get("queue_b_depth", {}).get("max", 0) <= qb,
        "buffered frames bounded": rt.get("frames_buffered", {}).get("max", 0) <= int(s["config"]["max_buffered_frames"]),
        "no result attached to a non-routed frame": all(r["routed"] or r["b_worker"] is None for r in rows),
        "every failure has an explicit status": all(
            r["status"] in ("OK", "DROPPED", "INVALID") or r["a_error"] or r["b_error"] for r in rows),
        "finished without hanging": elapsed < 300,
    }
    return {
        "name": name, "ok": all(checks.values()), "elapsed_s": round(elapsed, 1), "frames": len(rows),
        "status": dict(Counter(r["status"] for r in rows)), "a_status": dict(Counter(r["a_status"] for r in rows)),
        "b_status": dict(Counter(r["b_status"] for r in rows)), "checks": checks,
        "worker_restarts": s["worker_restarts"], "late_results_discarded": s["late_results_discarded"],
        "dropped": s["frames_dropped"], "aborted": s["aborted"],
        "max_queue_a": rt.get("queue_a_depth", {}).get("max"), "max_queue_b": rt.get("queue_b_depth", {}).get("max"),
        "reader_blocked_s": s["reader_blocked_s"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true", help="use fake models (fast)")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for name, desc, args in SCENARIOS:
        if a.only and name not in a.only:
            continue
        print(f"\n=== {name}: {desc}", flush=True)
        r = run(name, args, a.fake)
        r["description"] = desc
        results.append(r)
        print(f"    {'PASS' if r['ok'] else 'FAIL'}  frames={r.get('frames')}  status={r.get('status')}  "
              f"A={r.get('a_status')}  B={r.get('b_status')}  restarts={r.get('worker_restarts')}  "
              f"late={r.get('late_results_discarded')}  maxQ(A,B)=({r.get('max_queue_a')},{r.get('max_queue_b')})",
              flush=True)
        for k, v in r.get("checks", {}).items():
            if not v:
                print(f"      FAILED CHECK: {k}")
    (OUT / "report.json").write_text(json.dumps(results, indent=2))

    md = ["| scenario | result | frames | overall status | Model A status | Model B status | notes |",
          "|---|---|---|---|---|---|---|"]
    for r in results:
        notes = []
        if r.get("worker_restarts") and any(r["worker_restarts"].values()):
            notes.append(f"restarts {r['worker_restarts']}")
        if r.get("late_results_discarded"):
            notes.append(f"{r['late_results_discarded']} late results discarded")
        if r.get("dropped"):
            notes.append(f"{r['dropped']} dropped")
        if r.get("aborted"):
            notes.append("aborted -> in-flight frames INCOMPLETE")
        notes.append(f"max queue A/B {r.get('max_queue_a')}/{r.get('max_queue_b')}")
        md.append(f"| {r['name']} | {'PASS' if r['ok'] else 'FAIL'} | {r.get('frames')} | {r.get('status')} | "
                  f"{r.get('a_status')} | {r.get('b_status')} | {'; '.join(notes)} |")
    (OUT / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nall passed: {all(r['ok'] for r in results)}  ->  {OUT / 'report.md'}")
    sys.exit(0 if all(r["ok"] for r in results) else 1)


if __name__ == "__main__":
    main()
