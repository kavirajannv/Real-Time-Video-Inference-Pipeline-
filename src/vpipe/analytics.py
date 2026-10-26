"""People / object analytics on top of Model B detections.

Runs in the writer thread, i.e. strictly in frame order, which is what a tracker needs
(Model B workers finish out of order; tracking them there would be wrong).

Per frame:  scene-cut detection -> ByteTrack -> counts per class -> line crossing -> heatmap
            (+ optional person attributes per track: gender estimate, bag, backpack, ...)
End of run: analytics_summary.json, counts.csv, people_over_time.png, heatmaps.png
            (+ people_tracks.csv, people_gallery.png with attributes)
"""

from __future__ import annotations

import csv
import json
import logging
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from .attributes import ATTRIBUTES, TrackAttributes
from .schema import STRIDE_REASON, BStatus, FrameRecord

log = logging.getLogger(__name__)

PERSON = "person"


# --------------------------------------------------------------------------- scene cuts
class SceneCutDetector:
    """Scene-change detector for fixed-camera footage joined by cuts *or* dissolves.

    Signal: median absolute difference between the current grayscale thumbnail and the
    one `lag` frames earlier. With a fixed camera the background is identical, so the
    median stays near 0 even when people or a cyclist fill part of the view (they move
    a minority of pixels). A new view changes most pixels, so the median jumps. Comparing
    against a frame ~1 s back also catches slow dissolves, whose frame-to-frame change is
    tiny. Measured on the demo video: p50 = 2, transitions > 20.
    """

    def __init__(self, threshold: float = 20.0, lag: int = 30, min_scene_frames: int = 60):
        from collections import deque

        self.threshold = threshold
        self.min_scene_frames = min_scene_frames
        self._hist = deque(maxlen=lag + 1)
        self._since_cut = 0
        self.in_transition = False

    def signal(self, image: np.ndarray) -> float:
        # pick every k-th pixel instead of area-averaging: a 1080p frame strided by 12 is exactly 160x90,
        # and the median-difference signal is robust to the extra pixel noise (single-thread resize = 19 ms)
        h, w = image.shape[:2]
        small = image[:: max(1, h // 90), :: max(1, w // 160)]
        if small.shape[:2] != (90, 160):
            small = cv2.resize(small, (160, 90), interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(np.ascontiguousarray(small), cv2.COLOR_BGR2GRAY).astype(np.float32)
        sig = float(np.median(np.abs(gray - self._hist[0]))) if self._hist else 0.0
        self._hist.append(gray)
        return sig

    def is_cut(self, image: np.ndarray) -> bool:
        """True on the first frame of a scene change (used for hard cuts / simple callers)."""
        return self.update(image) == "start"

    def update(self, image: np.ndarray) -> str:
        """Returns "start" (a change begins), "transition" (still changing), "end" (new view
        has settled: start a new scene) or "steady"."""
        sig = self.signal(image)
        self._since_cut += 1
        if not self.in_transition:
            if self._since_cut >= self.min_scene_frames and sig > self.threshold:
                self.in_transition = True
                self._since_cut = 0
                return "start"
            return "steady"
        if sig > self.threshold:
            return "transition"
        self.in_transition = False
        self._since_cut = 0
        return "end"


# --------------------------------------------------------------------------- tracker
def _make_tracker(fps: float):
    import torch
    from ultralytics.trackers.byte_tracker import BYTETracker

    # the main process only needs torch for box bookkeeping; a full-size thread pool here
    # would fight the inference workers for cores
    torch.set_num_threads(1)

    args = SimpleNamespace(tracker_type="bytetrack", track_high_thresh=0.25, track_low_thresh=0.1,
                           new_track_thresh=0.25, track_buffer=int(round(2 * fps)), match_thresh=0.8,
                           fuse_score=True)  # keep lost tracks 2 s: people hidden in a crowd keep their ID
    try:
        return BYTETracker(args, frame_rate=int(round(fps)))
    except TypeError:
        return BYTETracker(args)


def _boxes(dets: list[dict], shape: tuple[int, int]):
    from ultralytics.engine.results import Boxes

    arr = np.array([[*d["box"], d["conf"], d.get("cls", 0)] for d in dets], dtype=np.float32).reshape(-1, 6)
    return Boxes(arr, shape)


def _side(p, a, b) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _segments_cross(p1, p2, a, b) -> bool:
    d1, d2 = _side(p1, a, b), _side(p2, a, b)
    d3, d4 = _side(a, p1, p2), _side(b, p1, p2)
    return (d1 > 0) != (d2 > 0) and d1 != 0 and d2 != 0 and (d3 > 0) != (d4 > 0)


class _Scene:
    def __init__(self, index: int, start_frame: int, first_image: np.ndarray, heat_shape):
        self.index = index
        self.start_frame = start_frame
        self.end_frame = start_frame
        self.thumb = first_image.copy()
        self.frames = 0
        self.heat = np.zeros(heat_shape, dtype=np.float32)
        self.people_ids: set[int] = set()


class Analytics:
    def __init__(self, out_dir: Path, size: tuple[int, int], fps: float,
                 count_line: tuple[float, float, float, float] = (0.0, 0.75, 1.0, 0.75),
                 min_track_frames: int = 15, scene_cut_threshold: float = 20.0, attributes=None,
                 detect_every: int = 1, background_attributes: bool = False):
        self.out_dir = Path(out_dir)
        self.w, self.h = size
        # attributes: a model with predict(bgr_crop) -> probabilities in ATTRIBUTES order (or None)
        self.attrs = TrackAttributes(attributes, size, background=background_attributes)             if attributes is not None else None
        self.fps = fps
        self.detect_every = max(1, detect_every)
        self._held: list[dict] = []              # last tracked boxes, shown on frames Model B skips (stride)
        self.stage_s = {"scene": 0.0, "tracker": 0.0, "attributes": 0.0}
        self.line = ((count_line[0] * self.w, count_line[1] * self.h), (count_line[2] * self.w, count_line[3] * self.h))
        self.min_track_frames = min_track_frames
        self.cuts = SceneCutDetector(scene_cut_threshold)
        self.heat_div = 4
        self.tracker = None
        self.scenes: list[_Scene] = []
        self._next_gid = 1
        self._gid: dict[int, int] = {}           # tracker id (current scene) -> global id
        self.tracks: dict[int, dict] = {}       # global id -> {"label", "frames", "last", "counted"}
        self.line_in = 0                         # crossings toward the camera / downward
        self.line_out = 0
        self.line_in_by_class: Counter = Counter()
        self.line_out_by_class: Counter = Counter()
        self.rows: list[dict] = []
        self.transitions: list[tuple[int, int]] = []   # (first, last) frame of each scene transition
        self._transition_start: int | None = None

    # ------------------------------------------------------------------ per frame
    def _new_scene(self, frame_id: int, image: np.ndarray) -> None:
        # the tracker only sees every Nth frame: give it that rate so its 2 s memory stays 2 s
        self.tracker = _make_tracker(self.fps / self.detect_every)
        self._gid = {}
        self._held = []
        self.scenes.append(_Scene(len(self.scenes), frame_id, image,
                                  (self.h // self.heat_div + 1, self.w // self.heat_div + 1)))

    def _global_id(self, tid: int) -> int:
        if tid not in self._gid:
            self._gid[tid] = self._next_gid
            self._next_gid += 1
        return self._gid[tid]

    def unique_count(self, label: str = PERSON) -> int:
        return sum(1 for t in self.tracks.values() if t["label"] == label and t["frames"] >= self.min_track_frames)

    def unique_people_ids(self) -> list[int]:
        return [i for i, t in self.tracks.items() if t["label"] == PERSON and t["frames"] >= self.min_track_frames]

    def unique_by_gender(self) -> Counter:
        if self.attrs is None:
            return Counter()
        return Counter(self.attrs.gender(i) for i in self.unique_people_ids())

    def process(self, rec: FrameRecord, image: np.ndarray | None) -> dict:
        usable = image is not None and image.shape[:2] == (self.h, self.w)
        phase = "steady"
        t0 = time.perf_counter()
        if usable:
            phase = self.cuts.update(image)
            if self.tracker is None or phase == "end":
                self._new_scene(rec.frame_id, image)
            if phase == "start":
                self._transition_start = rec.frame_id
            elif phase == "end" and self._transition_start is not None:
                self.transitions.append((self._transition_start, rec.frame_id - 1))
                self._transition_start = None
        scene = self.scenes[-1] if self.scenes else None
        if scene is not None and usable and phase not in ("start", "transition"):
            scene.frames += 1
            if scene.frames == 15:  # representative background once the new view has settled
                scene.thumb = image.copy()

        self.stage_s["scene"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        status = "OK"
        tracked: list[dict] = []
        held = rec.b_status == BStatus.SKIPPED and (rec.route_reason or "").startswith(STRIDE_REASON)
        if scene is None or not usable:
            status = "UNAVAILABLE"
        elif phase in ("start", "transition"):
            # two scenes are blended on screen: detections are "ghosts" of both views, so
            # tracking / counting pause instead of inventing identities and crossings
            status = "TRANSITION"
            if rec.b_status == BStatus.DETECTIONS:
                tracked = [{"id": None, "label": d["label"], "conf": d["conf"], "box": d["box"]}
                           for d in rec.detections if d["conf"] >= 0.25]
        elif held:
            # Model B did not run on this frame: keep the last tracked boxes (no tracker step, no new IDs)
            tracked = [dict(t) for t in self._held]
        elif rec.b_status in (BStatus.DETECTIONS, BStatus.NO_DETECTIONS, BStatus.SKIPPED):
            dets = rec.detections if rec.b_status == BStatus.DETECTIONS else []
            out = self.tracker.update(_boxes(dets, (self.h, self.w)), image)
            for row in out:
                x1, y1, x2, y2, tid, score, cls, idx = row[:8]
                label = dets[int(idx)]["label"] if 0 <= int(idx) < len(dets) else str(int(cls))
                tracked.append({"id": self._global_id(int(tid)), "label": label, "conf": float(score),
                                "box": [float(x1), float(y1), float(x2), float(y2)]})
        else:
            status = "UNAVAILABLE"  # Model A/B failed for this frame: no counts rather than wrong counts

        self.stage_s["tracker"] += time.perf_counter() - t0
        a, b = self.line
        for t in (tracked if status == "OK" else []):
            x1, y1, x2, y2 = t["box"]
            foot = ((x1 + x2) / 2.0, y2)
            st = self.tracks.setdefault(t["id"], {"label": t["label"], "frames": 0, "last": None, "counted": set()})
            st["frames"] += 1
            if st["last"] is not None and _segments_cross(st["last"], foot, a, b):
                direction = "in" if _side(foot, a, b) > 0 else "out"
                if direction not in st["counted"]:
                    st["counted"].add(direction)
                    if direction == "in":
                        self.line_in += t["label"] == PERSON
                        self.line_in_by_class[t["label"]] += 1
                    else:
                        self.line_out += t["label"] == PERSON
                        self.line_out_by_class[t["label"]] += 1
            st["last"] = foot
            if t["label"] == PERSON and self.attrs is not None and not held:  # held boxes lag the image
                t0 = time.perf_counter()
                self.attrs.observe(rec.frame_id, t["id"], t["box"], image)
                self.stage_s["attributes"] += time.perf_counter() - t0
                t["gender"] = self.attrs.gender(t["id"])
            if t["label"] == PERSON and scene is not None:
                scene.people_ids.add(t["id"])
                hx, hy = int(foot[0] / self.heat_div), int(min(foot[1], self.h - 1) / self.heat_div)
                cv2.circle(scene.heat, (hx, hy), max(2, int((x2 - x1) / self.heat_div / 6)), 1.0, -1)
        if scene is not None:
            scene.end_frame = rec.frame_id
        if status == "OK" and not held:
            self._held = tracked
        elif status != "OK":
            self._held = []

        counts = Counter(t["label"] for t in tracked)
        in_frame_gender = Counter(t.get("gender", "unknown") for t in tracked if t["label"] == PERSON)
        result = {
            "status": status,
            "scene": scene.index if scene else None,
            "people": counts.get(PERSON, 0) if status == "OK" else None,
            "objects": dict(counts) if status == "OK" else {},
            "unique_people": self.unique_count(),
            "line_in": self.line_in,
            "line_out": self.line_out,
            "tracks": tracked,
        }
        if self.attrs is not None:
            ok = status == "OK"
            result["gender_in_frame"] = {g: in_frame_gender.get(g, 0) for g in ("male", "female", "unknown")} if ok else None
            result["unique_by_gender"] = dict(self.unique_by_gender())
        self.rows.append({"frame_id": rec.frame_id, "time_s": round(rec.frame_id / self.fps, 3), **result})
        return result

    # ------------------------------------------------------------------ end of run
    def finalize(self) -> dict:
        if self.attrs is not None:
            self.attrs.flush()
        if self.rows:
            log.info("analytics ms/frame: %s", "  ".join(f"{k} {1000 * v / len(self.rows):.1f}"
                                                       for k, v in self.stage_s.items()))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        ok_rows = [r for r in self.rows if r["status"] == "OK"]
        people = [r["people"] for r in ok_rows]
        classes = sorted({c for r in ok_rows for c in r["objects"]})
        unique_by_class = Counter(t["label"] for t in self.tracks.values() if t["frames"] >= self.min_track_frames)
        peak_row = max(ok_rows, key=lambda r: r["people"]) if ok_rows else None
        summary = {
            "frames": len(self.rows),
            "frames_with_counts": len(ok_rows),
            "frames_unavailable": sum(1 for r in self.rows if r["status"] == "UNAVAILABLE"),
            "frames_in_transition": sum(1 for r in self.rows if r["status"] == "TRANSITION"),
            "transitions": [{"start_frame": s, "end_frame": e, "start_s": round(s / self.fps, 2),
                             "end_s": round(e / self.fps, 2)} for s, e in self.transitions],
            "duration_s": round(len(self.rows) / self.fps, 2),
            "scenes": [{"scene": s.index, "start_frame": s.start_frame, "end_frame": s.end_frame,
                        "start_s": round(s.start_frame / self.fps, 2), "end_s": round(s.end_frame / self.fps, 2),
                        "unique_people": sum(1 for i in s.people_ids
                                             if self.tracks[i]["frames"] >= self.min_track_frames)}
                       for s in self.scenes],
            "people": {
                "unique_total": self.unique_count(),
                "peak_in_frame": peak_row["people"] if peak_row else 0,
                "peak_frame_id": peak_row["frame_id"] if peak_row else None,
                "peak_time_s": peak_row["time_s"] if peak_row else None,
                "mean_per_frame": round(float(np.mean(people)), 2) if people else 0.0,
                "median_per_frame": float(np.median(people)) if people else 0.0,
                "crossed_line_in": self.line_in,
                "crossed_line_out": self.line_out,
            },
            **self._attribute_summary(),
            "objects": {
                c: {"unique_tracked": unique_by_class.get(c, 0),
                    "peak_in_frame": max((r["objects"].get(c, 0) for r in ok_rows), default=0),
                    "mean_per_frame": round(float(np.mean([r["objects"].get(c, 0) for r in ok_rows])), 3),
                    "crossed_in": self.line_in_by_class.get(c, 0), "crossed_out": self.line_out_by_class.get(c, 0)}
                for c in classes
            },
            "method": {
                "tracker": "ByteTrack (Ultralytics), 2 s track buffer, reset at every scene change",
                "transitions": "counting paused while two scenes are blended (dissolve); boxes still drawn",
                "unique_rule": f"track seen in >= {self.min_track_frames} frames",
                "count_line_px": [list(map(round, self.line[0])), list(map(round, self.line[1]))],
                "line_in": "foot point crosses the line moving down (toward the camera)",
                "line_out": "foot point crosses the line moving up (away from the camera)",
            },
        }
        if self.attrs is not None:
            summary["method"]["attributes"] = (
                "person-attributes-recognition-crossroad-0230 (OpenVINO) on full-resolution crops, sampled every "
                f"{self.attrs.every} frames (max {self.attrs.max_samples} per track, boxes >= {self.attrs.min_height} px "
                "tall and not touching the frame edge), probabilities averaged per track. Gender is an appearance-based "
                f"estimate: male if mean p(male) >= {self.attrs.male_threshold}, female if <= "
                f"{self.attrs.female_threshold}, otherwise or with < {self.attrs.min_samples} samples: unknown.")
            self._write_tracks()
        with open(self.out_dir / "analytics_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with open(self.out_dir / "counts.csv", "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            genders = ("male", "female", "unknown") if self.attrs is not None else ()
            wr.writerow(["frame_id", "time_s", "scene", "status", "people", "unique_people_so_far",
                         "line_in", "line_out", *[f"{g}_in_frame" for g in genders], *classes])
            for r in self.rows:
                gf = r.get("gender_in_frame") or {}
                wr.writerow([r["frame_id"], r["time_s"], r["scene"], r["status"], r["people"], r["unique_people"],
                             r["line_in"], r["line_out"], *[gf.get(g, "") for g in genders],
                             *[r["objects"].get(c, 0) for c in classes]])
        try:
            self._chart()
        except Exception:  # noqa: BLE001 - a missing plotting backend must not lose the run
            log.exception("chart rendering failed")
        self._heatmaps()
        return summary

    def _attribute_summary(self) -> dict:
        if self.attrs is None:
            return {}
        ids = self.unique_people_ids()
        per = [self.attrs.summary(i) for i in ids]
        by_gender = Counter(p["gender"] for p in per)
        return {"people_attributes": {
            "unique_people": len(ids),
            "male": by_gender.get("male", 0),
            "female": by_gender.get("female", 0),
            "unknown": by_gender.get("unknown", 0),
            "attributes_unknown": sum(1 for p in per if p["has_bag"] is None),
            **{name: sum(1 for p in per if p[name]) for name in ATTRIBUTES[1:]},
        }}

    def _write_tracks(self) -> None:
        scene_of = {i: s.index for s in self.scenes for i in s.people_ids}
        ids = self.unique_people_ids()
        with open(self.out_dir / "people_tracks.csv", "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["person_id", "scene", "frames_tracked", "gender", "p_male", "attribute_samples",
                         *ATTRIBUTES[1:]])
            for i in ids:
                a = self.attrs.summary(i)
                wr.writerow([i, scene_of.get(i), self.tracks[i]["frames"], a["gender"], a["p_male"], a["samples"],
                             *["" if a[n] is None else int(a[n]) for n in ATTRIBUTES[1:]]])
        groups = {g: [i for i in ids if self.attrs.gender(i) == g] for g in ("male", "female", "unknown")}
        img = self.attrs.gallery(groups)
        if img is not None:
            cv2.imwrite(str(self.out_dir / "people_gallery.png"), img)

    def _chart(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
        blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
        t = [r["time_s"] for r in self.rows]
        ppl = [r["people"] if r["people"] is not None else np.nan for r in self.rows]
        # 1-second rolling mean on top of the raw per-frame count
        k = max(1, int(round(self.fps)))
        arr = np.array(ppl, dtype=float)
        smooth = np.array([np.nanmean(arr[max(0, i - k + 1): i + 1]) if not np.all(np.isnan(arr[max(0, i - k + 1): i + 1]))
                           else np.nan for i in range(len(arr))])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, facecolor=surface,
                                       gridspec_kw={"height_ratios": [3, 2], "hspace": 0.28})
        for ax in (ax1, ax2):
            ax.set_facecolor(surface)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(grid)
            ax.tick_params(colors=ink2, labelsize=9)
            ax.grid(axis="y", color=grid, linewidth=0.8)
            for s, e in self.transitions:
                ax.axvspan(s / self.fps, (e + 1) / self.fps, color="#eeede9", linewidth=0, zorder=0)

        ax1.plot(t, ppl, color=blue, linewidth=0.8, alpha=0.35)
        ax1.plot(t, smooth, color=blue, linewidth=2)
        ax1.set_title("People visible per frame", loc="left", color=ink, fontsize=12, fontweight="bold", pad=22)
        ax1.text(0, 1.02, "thin line = every frame, bold = 1 s average; grey bands = scene transitions "
                          "(counting paused)", transform=ax1.transAxes, color=ink2, fontsize=9, va="bottom")
        ax1.set_ylabel("people", color=ink2)
        ax1.set_ylim(bottom=0)

        uniq = [r["unique_people"] for r in self.rows]
        lin = [r["line_in"] for r in self.rows]
        lout = [r["line_out"] for r in self.rows]
        series = ((uniq, blue, "unique people"), (lin, orange, "crossed line: in"), (lout, aqua, "crossed line: out"))
        # end labels: nudge apart vertically (in points) when final values are close
        order = sorted(range(3), key=lambda i: series[i][0][-1])
        nudge = {order[0]: -6, order[1]: 6, order[2]: 0} if abs(lin[-1] - lout[-1]) < 0.06 * max(uniq[-1], 1) \
            else {0: 0, 1: 0, 2: 0}
        for i, (ys, col, name) in enumerate(series):
            ax2.plot(t, ys, color=col, linewidth=2, label=name)
            ax2.annotate(f"{name}: {ys[-1]}", (t[-1], ys[-1]), xytext=(6, nudge[i]), textcoords="offset points",
                         color=ink, fontsize=9, va="center")
        ax2.set_title("Cumulative counts", loc="left", color=ink, fontsize=12, fontweight="bold", pad=10)
        ax2.set_ylabel("people", color=ink2)
        ax2.set_xlabel("video time (s)", color=ink2)
        ax2.set_ylim(bottom=0)
        ax2.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=ink)
        ax2.set_xlim(0, t[-1] * 1.16 if t else 1)
        fig.savefig(self.out_dir / "people_over_time.png", dpi=130, bbox_inches="tight", facecolor=surface)
        plt.close(fig)

    def _heatmaps(self) -> None:
        tiles = []
        tw, th = 640, 360
        for s in self.scenes:
            heat = cv2.GaussianBlur(s.heat, (0, 0), sigmaX=4)
            base = cv2.resize(s.thumb, (tw, th), interpolation=cv2.INTER_AREA)
            nz = heat[heat > 0]
            if nz.size:
                # scale to the 99th percentile so one busy spot does not wash out the rest
                top = float(np.percentile(nz, 99))
                rel = np.clip(heat / max(top, 1e-6), 0, 1) ** 0.6
                norm = cv2.resize((rel * 255).astype(np.uint8), (tw, th))
                colour = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
                alpha = np.clip(norm.astype(np.float32) / 255.0 * 1.2, 0, 0.8)[..., None]
                base = (base * (1 - alpha) + colour * alpha).astype(np.uint8)
            label = f"scene {s.index}  {s.start_frame / self.fps:.0f}-{s.end_frame / self.fps:.0f}s  " \
                    f"unique people {sum(1 for i in s.people_ids if self.tracks[i]['frames'] >= self.min_track_frames)}"
            cv2.rectangle(base, (0, 0), (tw, 30), (20, 20, 20), -1)
            cv2.putText(base, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(base)
        if not tiles:
            return
        cols = 2 if len(tiles) > 1 else 1
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
        cv2.imwrite(str(self.out_dir / "heatmaps.png"), grid)
