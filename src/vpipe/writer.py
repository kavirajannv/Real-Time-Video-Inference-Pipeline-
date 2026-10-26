"""Frame annotation, output video and per-frame processing log."""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from .schema import AStatus, BStatus, FrameRecord, SKIP_REASON, STRIDE_REASON

log = logging.getLogger(__name__)

GREEN = (80, 200, 80)
YELLOW = (0, 215, 255)
ORANGE = (0, 140, 255)
RED = (40, 40, 230)
WHITE = (255, 255, 255)
GREY = (170, 170, 170)
GENDER_STYLE = {"male": ("man", (230, 150, 40)), "female": ("woman", (190, 70, 235))}  # BGR

_PALETTE = [(255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29), (207, 210, 49), (72, 249, 10),
            (146, 204, 23), (61, 219, 134), (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
            (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255), (82, 0, 133), (203, 56, 255)]


def _box_colour(label: str) -> tuple[int, int, int]:
    r, g, b = _PALETTE[sum(map(ord, label)) % len(_PALETTE)]
    return b, g, r


def _ms(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else round((b - a) * 1000.0, 3)


def status_lines(rec: FrameRecord) -> list[tuple[str, tuple[int, int, int]]]:
    """Text overlay lines (and colours) for a frame - one source of truth for every frame state."""
    head_colour = GREEN if rec.overall_status == "OK" else (YELLOW if rec.overall_status == "DROPPED" else RED)
    lines = [(f"Frame {rec.frame_id}   Status: {rec.overall_status}", head_colour)]

    a = rec.a_status
    if a == AStatus.OK:
        lines.append((f"Model A: {rec.a_label}  conf {rec.a_conf:.2f}", WHITE))
    elif a == AStatus.HELD:
        lines.append(("Model A: not run on this frame (fast mode)", GREY))
    elif a == AStatus.NOT_GATED:
        lines.append(("Model A: gate off (fast mode) - sent straight to Model B", GREY))
    elif a == AStatus.DROPPED:
        lines.append(("Model A: NOT PROCESSED (dropped - live overload)", YELLOW))
    elif a == AStatus.INVALID:
        lines.append(("Model A: NOT PROCESSED (invalid frame)", RED))
    else:
        lines.append((f"Model A: {a.value} - {rec.a_error or ''}"[:90], RED))

    b = rec.b_status
    if b == BStatus.DETECTIONS:
        lines.append((f"Model B: DETECTIONS ({len(rec.detections)})", GREEN))
    elif b == BStatus.NO_DETECTIONS:
        lines.append(("Model B: NO DETECTIONS", ORANGE))
    elif b == BStatus.SKIPPED and (rec.route_reason or "").startswith(STRIDE_REASON):
        lines.append(("Model B: not run on this frame - boxes held by the tracker", GREY))
    elif b == BStatus.SKIPPED:
        lines.append((f"Model B: SKIPPED - {SKIP_REASON}", GREY))
    elif b == BStatus.NOT_RUN:
        lines.append((f"Model B: NOT RUN - {rec.route_reason or 'no Model A result'}", RED if rec.is_failure else YELLOW))
    else:
        lines.append((f"Model B: {b.value} - {rec.b_error or ''}"[:90], RED))
    return lines


def _draw_box(out, box, text, col, scale, thick, k: float = 1.0) -> None:
    x1, y1, x2, y2 = (int(round(v * k)) for v in box)  # k: source -> output image coordinates
    cv2.rectangle(out, (x1, y1), (x2, y2), col, thick + 1)
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale * 0.8, thick)
    ty = max(th + bl, y1)
    cv2.rectangle(out, (x1, ty - th - bl), (x1 + tw + 4, ty), col, -1)
    cv2.putText(out, text, (x1 + 2, ty - bl), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.8, WHITE, thick, cv2.LINE_AA)


def _dim(roi: np.ndarray) -> None:
    roi[:] = cv2.convertScaleAbs(roi, alpha=0.35)  # translucent dark panel (uint8 math, no float copy)


def _draw_analytics(out, an: dict, line, scale, thick, pad, show_classes=(), k: float = 1.0) -> None:
    w = out.shape[1]
    shown = (lambda label: label in show_classes) if show_classes else (lambda label: True)
    for t in (t for t in an["tracks"] if shown(t["label"])):
        tag = f"#{t['id']} " if t["id"] is not None else ""
        name, col = GENDER_STYLE.get(t.get("gender"), (t["label"], _box_colour(t["label"])))
        _draw_box(out, t["box"], f"{tag}{name} {t['conf']:.2f}", col, scale, thick, k)
    if line is not None:
        (ax, ay), (bx, by) = ((x * k, y * k) for x, y in line)
        cv2.line(out, (int(ax), int(ay)), (int(bx), int(by)), (0, 255, 255), thick + 1, cv2.LINE_AA)
        cv2.putText(out, f"IN {an['line_in']}  OUT {an['line_out']}", (int(ax) + pad, int(ay) - pad),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 255), thick + 1, cv2.LINE_AA)
    if an["status"] == "OK":
        others = sorted(((k, v) for k, v in an["objects"].items() if k != "person" and shown(k)),
                        key=lambda kv: -kv[1])[:5]
        lines = [f"People in frame: {an['people']}", f"Unique people: {an['unique_people']}"]
        if an.get("gender_in_frame") is not None:
            g, u = an["gender_in_frame"], an["unique_by_gender"]
            lines += [f"  in frame  men {g['male']}  women {g['female']}  ? {g['unknown']}",
                      f"  unique    men {u.get('male', 0)}  women {u.get('female', 0)}  ? {u.get('unknown', 0)}"]
        lines.append(f"Crossed line  in: {an['line_in']}  out: {an['line_out']}")
        lines += [f"{k}: {v}" for k, v in others]
    elif an["status"] == "TRANSITION":
        lines = ["Scene transition: counting paused", f"Unique people: {an['unique_people']}",
                 f"Crossed line  in: {an['line_in']}  out: {an['line_out']}"]
    else:
        lines = ["Counts: UNAVAILABLE (inference failed)", f"Unique people: {an['unique_people']}"]
    line_h = int(34 * scale)
    widths = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0] for t in lines]
    pw = max(widths) + 2 * pad
    ph = line_h * len(lines) + pad
    _dim(out[0:ph, w - pw:w])
    for i, text in enumerate(lines):
        col = (0, 255, 255) if i == 0 else WHITE
        cv2.putText(out, text, (w - pw + pad, (i + 1) * line_h), cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick,
                    cv2.LINE_AA)


def annotate(image: np.ndarray, rec: FrameRecord, count_line=None, show_classes=(), k: float = 1.0) -> np.ndarray:
    """Draw overlays on a copy of the frame (original content preserved).

    k scales source-frame coordinates (boxes, counting line) when `image` is a resized copy.
    """
    out = image.copy()
    h, w = out.shape[:2]
    scale = max(0.45, w / 1600.0)
    thick = max(1, int(round(scale * 2)))

    pad = int(10 * scale)
    if rec.analytics is not None:
        _draw_analytics(out, rec.analytics, count_line, scale, thick, pad, show_classes, k)
    elif rec.b_status == BStatus.DETECTIONS:
        for d in rec.detections:
            _draw_box(out, d["box"], f"{d['label']} {d['conf']:.2f}", _box_colour(d["label"]), scale, thick, k)

    lines = status_lines(rec)
    line_h = int(34 * scale)
    widths = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0] for t, _ in lines]
    panel_w = min(w, max(widths) + 2 * pad)
    panel_h = line_h * len(lines) + pad
    _dim(out[0:panel_h, 0:panel_w])
    for i, (text, col) in enumerate(lines):
        cv2.putText(out, text, (pad, (i + 1) * line_h), cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)

    if rec.is_failure or rec.a_status == AStatus.INVALID:
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), RED, max(4, thick * 3))
        banner = "INCOMPLETE / ERROR RESULT"
        (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, scale * 1.1, thick + 1)
        cv2.putText(out, banner, (w - tw - pad * 2, h - pad * 2), cv2.FONT_HERSHEY_SIMPLEX, scale * 1.1, RED,
                    thick + 1, cv2.LINE_AA)
    return out


def record_to_dict(rec: FrameRecord, t0: float) -> dict:
    rel = (lambda t: None if t is None else round(t - t0, 6))
    return {
        "frame_id": rec.frame_id,
        "status": rec.overall_status,
        "a_status": rec.a_status.value,
        "a_label": rec.a_label,
        "a_conf": None if rec.a_conf is None else round(rec.a_conf, 4),
        "a_top5": rec.a_top5,
        "a_worker": rec.a_worker,
        "a_error": rec.a_error,
        "routed": rec.routed,
        "route_reason": rec.route_reason,
        "b_status": rec.b_status.value,
        "b_worker": rec.b_worker,
        "b_error": rec.b_error,
        "num_detections": len(rec.detections),
        "detections": [{"label": d["label"], "conf": round(d["conf"], 4), "box": [round(v, 1) for v in d["box"]]}
                       for d in rec.detections],
        "t_ingest_s": rel(rec.t_ingest),
        "a_queue_wait_ms": _ms(rec.t_enqueue_a, rec.t_start_a),
        "a_infer_ms": _ms(rec.t_start_a, rec.t_end_a),
        "b_route_wait_ms": _ms(rec.t_route, rec.t_enqueue_b),
        "b_queue_wait_ms": _ms(rec.t_enqueue_b, rec.t_start_b),
        "b_infer_ms": _ms(rec.t_start_b, rec.t_end_b),
        "result_latency_ms": _ms(rec.t_ingest, rec.t_final),
        "e2e_latency_ms": _ms(rec.t_ingest, rec.t_written),
        **({} if rec.analytics is None else {
            "people_count": rec.analytics["people"],
            "object_counts": rec.analytics["objects"],
            "unique_people_so_far": rec.analytics["unique_people"],
            "line_in": rec.analytics["line_in"],
            "line_out": rec.analytics["line_out"],
            "scene": rec.analytics["scene"],
            "track_ids": [t["id"] for t in rec.analytics["tracks"]],
            "gender_in_frame": rec.analytics.get("gender_in_frame"),
        }),
    }


CSV_FIELDS = ["frame_id", "status", "a_status", "a_label", "a_conf", "a_worker", "a_error", "routed", "route_reason",
              "b_status", "b_worker", "b_error", "num_detections", "detections", "t_ingest_s", "a_queue_wait_ms",
              "a_infer_ms", "b_route_wait_ms", "b_queue_wait_ms", "b_infer_ms", "result_latency_ms", "e2e_latency_ms",
              "people_count", "unique_people_so_far", "line_in", "line_out", "scene", "object_counts"]


class _FfmpegWriter:
    """Pipes raw BGR frames into ffmpeg/libx264 (CRF 23): H.264 at a sane size on every OS."""

    def __init__(self, exe: str, path: Path, fps: float, size: tuple[int, int], preset: str = "veryfast"):
        import subprocess

        w, h = size
        cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
               "-r", f"{fps:.3f}", "-i", "-", "-c:v", "libx264", "-preset", preset, "-crf", "23",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
        self._p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def isOpened(self) -> bool:  # noqa: N802 - mirrors cv2.VideoWriter
        return self._p.poll() is None

    def write(self, frame: np.ndarray) -> None:
        try:
            self._p.stdin.write(memoryview(np.ascontiguousarray(frame)))  # no extra bytes copy
        except OSError as e:  # ffmpeg exited: surface its own error message instead of "broken pipe"
            self._p.wait()
            err = self._p.stderr.read().decode(errors="replace").strip()
            raise RuntimeError(f"ffmpeg stopped: {err or e}") from e

    def release(self) -> None:
        self._p.stdin.close()
        err = self._p.stderr.read().decode(errors="replace").strip()
        if self._p.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {err}")


def _open_video_writer(path: Path, fps: float, size: tuple[int, int], preset: str = "veryfast"):
    """H.264 plays in browsers and Windows players; mp4v (MPEG-4 Part 2) is the last resort.

    Preferred: ffmpeg from the imageio-ffmpeg package (libx264, CRF 23). Fallback: OpenCV. On Windows
    the pip OpenCV build's FFmpeg backend needs an external openh264 DLL, so H.264 is requested from
    Media Foundation directly; it ignores quality settings and writes ~40 Mbit/s at 1080p.
    """
    if path.exists():
        try:
            open(path, "r+b").close()
        except PermissionError as e:
            raise RuntimeError(f"cannot overwrite {path}: the file is open in another program "
                               "(close the video player) or choose another --output-dir") from e
    try:
        import imageio_ffmpeg

        return _FfmpegWriter(imageio_ffmpeg.get_ffmpeg_exe(), path, fps, size, preset)
    except (ImportError, RuntimeError, OSError) as e:
        log.warning("ffmpeg unavailable (%s); falling back to OpenCV's video writer", e)
    h264_backend = cv2.CAP_MSMF if sys.platform == "win32" else cv2.CAP_ANY
    for backend, codec in ((h264_backend, "avc1"), (cv2.CAP_ANY, "mp4v")):
        vw = cv2.VideoWriter(str(path), backend, cv2.VideoWriter_fourcc(*codec), fps, size)
        if vw.isOpened():
            return vw
        log.warning("video codec %s unavailable, trying next", codec)
    raise RuntimeError(f"cannot open video writer for {path}")


class OutputWriter:
    """Writes annotated frames (in the order given) plus the JSON and CSV logs."""

    def __init__(self, out_path: Path, fps: float, size: tuple[int, int], t0: float, write_video: bool = True,
                 analytics=None, show_classes: tuple[str, ...] = (), output_height: int = 0,
                 video_preset: str = "veryfast"):
        self.show_classes = tuple(show_classes)  # classes drawn in the video (empty = all)
        # output_height: write the video smaller than the source (encoding 1080p costs ~2x 720p);
        # analytics and person-attribute crops still use the full-resolution frame
        w, h = size
        self.video_size = size if not output_height or output_height >= h else             (int(round(w * output_height / h / 2)) * 2, int(output_height) // 2 * 2)
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps, self.size, self.t0 = fps, size, t0
        self.analytics = analytics
        self.analytics_summary: dict | None = None
        self.json_path = self.out_path.with_name("frames_log.json")
        self.csv_path = self.out_path.with_name("frames_log.csv")
        self._vw = None
        self._render_pool = None
        self._rendering: deque = deque()  # futures of frames being drawn, in frame order
        if write_video:
            self._vw = _open_video_writer(self.out_path, fps, self.video_size, video_preset)
            # resize + draw run on a few threads (OpenCV releases the GIL), so several frames are drawn at
            # once; frames still reach the encoder strictly in order. On a slow core a 1080p->720p resize
            # alone takes ~19 ms, so doing it one frame at a time capped the writer at ~11 fps.
            self._render_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="vpipe-render")
        self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._csv = csv.DictWriter(self._csv_file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        self._csv.writeheader()
        self.rows: list[dict] = []
        self.frames_written = 0
        self._last_id = -1
        self.stage_s = {"analytics": 0.0, "render+encode": 0.0, "log": 0.0}

    def write(self, rec: FrameRecord, image: np.ndarray | None) -> dict:
        if rec.frame_id != self._last_id + 1:
            raise RuntimeError(f"output order violated: got frame {rec.frame_id} after {self._last_id}")
        self._last_id = rec.frame_id
        w, h = self.size
        if image is None or image.shape[:2] != (h, w):
            image = np.zeros((h, w, 3), dtype=np.uint8)  # invalid frame placeholder, same dimensions
            usable = False
        else:
            usable = True
        clock = time.perf_counter
        t = clock()
        if self.analytics is not None:
            rec.analytics = self.analytics.process(rec, image if usable else None)
        t, self.stage_s["analytics"] = clock(), self.stage_s["analytics"] + clock() - t
        if self._vw is not None:
            self._rendering.append(self._render_pool.submit(self._render, image, rec))
            # keep a few frames in flight; hand finished ones to the encoder in order
            while self._rendering and (len(self._rendering) > 6 or self._rendering[0].done()):
                self._vw.write(self._rendering.popleft().result())
            t, self.stage_s["render+encode"] = clock(), self.stage_s["render+encode"] + clock() - t
        rec.t_written = time.perf_counter()
        row = record_to_dict(rec, self.t0)
        self.rows.append(row)
        self._csv.writerow({**row, "detections": json.dumps(row["detections"]),
                            "object_counts": json.dumps(row.get("object_counts", {}))})
        self.frames_written += 1
        self.stage_s["log"] += clock() - t
        return row

    def _render(self, image: np.ndarray, rec: FrameRecord) -> np.ndarray:
        # shrink first, then draw: fewer pixels to copy and draw on than drawing at source size
        k = 1.0
        if self.video_size != self.size:
            image = cv2.resize(image, self.video_size, interpolation=cv2.INTER_LINEAR)
            k = self.video_size[0] / self.size[0]
        return annotate(image, rec, self.analytics.line if self.analytics else None, self.show_classes, k)

    def close(self) -> None:
        if self._render_pool is not None:
            while self._rendering:
                self._vw.write(self._rendering.popleft().result())
            self._render_pool.shutdown(wait=True)
            self._render_pool = None
        if self.frames_written:
            # where the writer thread's time goes (it runs every frame, in order, in the main process)
            log.info("writer ms/frame: %s", "  ".join(f"{k} {1000 * v / self.frames_written:.1f}"
                                                    for k, v in self.stage_s.items()))
        if self._vw is not None:
            self._vw.release()
            self._vw = None
        if not self._csv_file.closed:
            self._csv_file.close()
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.rows, f, indent=1)
        if self.analytics is not None and self.analytics_summary is None:
            self.analytics_summary = self.analytics.finalize()
