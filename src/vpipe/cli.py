"""Command line interface: `python -m vpipe {download,labels,run}`."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import PERSON_PROXY_LABELS, FaultConfig, ModelSpec, PipelineConfig


def _add_fault_args(p: argparse.ArgumentParser, stage: str) -> None:
    s = stage.lower()
    g = p.add_argument_group(f"fault injection (Model {stage})")
    g.add_argument(f"--inject-delay-{s}", type=float, default=0.0, metavar="SEC", help="extra delay for selected frames")
    g.add_argument(f"--inject-delay-every-{s}", type=int, default=1, metavar="N",
                   help="apply the delay to frames with id %% N == 0 (default every frame)")
    g.add_argument(f"--inject-jitter-{s}", type=float, default=0.0, metavar="SEC",
                   help="random 0..SEC delay per task (forces out-of-order completion)")
    g.add_argument(f"--inject-error-every-{s}", type=int, default=0, metavar="N", help="raise on frames id %% N == 0")
    g.add_argument(f"--inject-error-rate-{s}", type=float, default=0.0, metavar="P", help="raise with probability P")
    g.add_argument(f"--inject-crash-after-{s}", type=int, default=0, metavar="N",
                   help="hard-kill each worker's first incarnation after N tasks")


def _fault_from_args(a: argparse.Namespace, stage: str) -> FaultConfig:
    s = stage.lower()
    g = lambda k: getattr(a, f"inject_{k}_{s}")  # noqa: E731
    delay = g("delay")
    return FaultConfig(delay_s=delay, delay_every=g("delay_every") if delay else 0, jitter_s=g("jitter"),
                       error_every=g("error_every"), error_rate=g("error_rate"), crash_after=g("crash_after"),
                       seed=a.seed)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vpipe", description="Real-time video inference pipeline (CPU)")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="download the YOLO11 weights (and, on request, the brief's sample clip)")
    d.add_argument("--sample-video", action="store_true",
                   help="also fetch the brief's solutions_ci_demo.mp4 (not used by default: input is people_walking.mp4)")
    d.add_argument("--video-dest", default="data/solutions_ci_demo.mp4")

    lb = sub.add_parser("labels", help="print Model A's real class labels (verify routing targets)")
    lb.add_argument("--weights", default="yolo11n-cls.pt")
    lb.add_argument("--grep", default="", help="substring filter")

    r = sub.add_parser("run", help="run the pipeline")
    r.add_argument("--video", default="data/people_walking.mp4")
    r.add_argument("--output-dir", default="outputs")
    r.add_argument("--output-name", default="output_annotated.mp4")
    r.add_argument("--a-workers", type=int, default=2)
    r.add_argument("--b-workers", type=int, default=2)
    r.add_argument("--weights-a", default="yolo11n-cls.pt")
    r.add_argument("--weights-b", default="yolo11n.pt")
    r.add_argument("--imgsz-a", type=int, default=224)
    r.add_argument("--imgsz-b", type=int, default=640)
    r.add_argument("--det-conf", type=float, default=0.25, help="Model B detection confidence threshold")
    r.add_argument("--fake-models", action="store_true", help="use fake models (no YOLO) - dry run / demos")
    r.add_argument("--fake-infer-ms", type=float, default=20.0)
    r.add_argument("--torch-threads", type=int, default=0, help="per worker; 0=auto split cores, -1=torch default")
    r.add_argument("--route-labels", default="person",
                   help="comma-separated top-1 labels that route to Model B (default: person)")
    r.add_argument("--person-proxy", action="store_true",
                   help=f"route on person-like ImageNet labels: {','.join(PERSON_PROXY_LABELS)}")
    r.add_argument("--route-threshold", type=float, default=0.30)
    r.add_argument("--mode", choices=("offline", "live"), default="offline")
    r.add_argument("--live-fps", type=float, default=0.0, help="live mode ingest rate (0 = source fps)")
    r.add_argument("--stale-after", type=float, default=0.0, help="live mode: drop tasks older than SEC at the worker")
    r.add_argument("--queue-a", type=int, default=8)
    r.add_argument("--queue-b", type=int, default=8)
    r.add_argument("--max-buffered", type=int, default=64, help="max frames held in memory between ingest and write")
    r.add_argument("--timeout-a", type=float, default=30.0)
    r.add_argument("--timeout-b", type=float, default=30.0)
    r.add_argument("--no-resize-ipc", action="store_true", help="baseline: send full-resolution frames to workers")
    r.add_argument("--max-frames", type=int, default=0)
    r.add_argument("--max-seconds", type=float, default=0.0, metavar="SEC",
                   help="process only the first SEC seconds of the video (e.g. 40)")
    r.add_argument("--repeat", type=int, default=1, help="replay the video N times (more samples for benchmarks)")
    r.add_argument("--stop-after", type=float, default=0.0, help="simulate an operator stop after SEC seconds")
    r.add_argument("--inject-invalid-every", type=int, default=0, metavar="N", help="corrupt every Nth frame")
    r.add_argument("--no-video", action="store_true", help="skip writing the mp4 (benchmark inference only)")
    r.add_argument("--min-conf-shown", type=float, default=0.0, metavar="P",
                   help="draw only boxes with detector confidence >= P (people preset default 0.80; "
                        "0 = all). Tracking and counts still use every detection")
    r.add_argument("--preview", action="store_true",
                   help="show a live window while running: input on the left, annotated output on the right")
    r.add_argument("--seed", type=int, default=0)
    g = r.add_argument_group("people / object analytics")
    g.add_argument("--preset", choices=("assignment", "people"), default="assignment",
                   help="people = Model A person gate (YOLO11n @320) + Model B YOLO11s + tracking/counting")
    g.add_argument("--analytics", action="store_true", help="track objects and count people (ByteTrack)")
    g.add_argument("--count-line", default="0,0.75,1,0.75",
                   help="counting line x1,y1,x2,y2 as fractions of width/height (default: horizontal at 75%%)")
    g.add_argument("--min-track-frames", type=int, default=15, help="frames a track must live to count as unique")
    g.add_argument("--attributes-model", default="",
                   help="OpenVINO person-attributes .xml (gender estimate, bag, backpack, hat, ...); the people "
                        "preset uses models/person-attributes-recognition-crossroad-0230.xml if present; 'none' = off")
    g.add_argument("--detect-every", type=int, default=1, metavar="N",
                   help="run Model B on every Nth frame Model A routes; the tracker holds boxes in between "
                        "(faster, slightly less precise). Default 1; people preset: 2")
    g.add_argument("--infer-every", type=int, default=1, metavar="N",
                   help="send only every Nth frame to the models; the tracker holds boxes on the others (HELD)")
    g.add_argument("--output-height", type=int, default=0, metavar="PX",
                   help="write the video at this height, e.g. 720 (default: source size)")
    g.add_argument("--video-preset", default="veryfast",
                   choices=("ultrafast", "superfast", "veryfast", "faster", "fast", "medium"),
                   help="libx264 speed preset for the output video (faster preset = less CPU, bigger file)")
    g.add_argument("--no-gate", action="store_true",
                   help="skip Model A: inferred frames go straight to Model B (fast mode; logged as NOT_GATED)")
    g.add_argument("--fast", action="store_true",
                   help="people preset tuned for speed: models on every --infer-every frame (default 3), "
                        "720p output, superfast encoding, worker counts from the CPU core count")
    g.add_argument("--show-classes", default="",
                   help="comma-separated classes to draw in the video (default: all; people preset: people, bags, "
                        "vehicles; 'all' = everything). Summaries always count every class")
    _add_fault_args(r, "A")
    _add_fault_args(r, "B")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def config_from_args(a: argparse.Namespace) -> PipelineConfig:
    if a.preset == "people":
        cfg = _people_preset(a)
        if cfg is not None:
            return cfg
    labels = tuple(x.strip() for x in a.route_labels.split(",") if x.strip())
    if a.person_proxy:
        labels = tuple(dict.fromkeys(labels + PERSON_PROXY_LABELS))
    kind = "fake" if a.fake_models else "yolo"
    fake_s = a.fake_infer_ms / 1000.0
    return PipelineConfig(
        video=a.video, output_dir=a.output_dir, output_name=a.output_name,
        a_workers=a.a_workers, b_workers=a.b_workers,
        model_a=ModelSpec(kind, a.weights_a, a.imgsz_a, fake_infer_s=fake_s),
        model_b=ModelSpec(kind, a.weights_b, a.imgsz_b, conf=a.det_conf, fake_infer_s=fake_s * 3),
        torch_threads=a.torch_threads, route_labels=labels, route_threshold=a.route_threshold,
        queue_a_size=a.queue_a, queue_b_size=a.queue_b, max_buffered_frames=a.max_buffered,
        mode=a.mode, live_fps=a.live_fps, stale_after_s=a.stale_after,
        timeout_a_s=a.timeout_a, timeout_b_s=a.timeout_b, resize_for_ipc=not a.no_resize_ipc,
        analytics=a.analytics, count_line=tuple(float(v) for v in a.count_line.split(",")),
        min_track_frames=a.min_track_frames,
        attributes_model="" if a.attributes_model == "none" else a.attributes_model,
        show_classes=_split_classes(a.show_classes),
        detect_every=a.detect_every, infer_every=a.infer_every, gate=not a.no_gate, output_height=a.output_height,
        video_preset=a.video_preset,
        max_frames=a.max_frames, max_seconds=a.max_seconds, repeat=a.repeat, stop_after_s=a.stop_after, invalid_every=a.inject_invalid_every,
        fault_a=_fault_from_args(a, "A"), fault_b=_fault_from_args(a, "B"), write_video=not a.no_video, preview=a.preview, min_conf_shown=a.min_conf_shown,
    )


PEOPLE_SHOW_CLASSES = ("person", "handbag", "backpack", "suitcase", "umbrella", "car", "bus", "truck", "bicycle",
                       "motorcycle")


def _split_classes(value: str) -> tuple[str, ...]:
    if value in ("", "all"):
        return ()
    return tuple(c.strip() for c in value.split(",") if c.strip())


def _people_preset(a: argparse.Namespace) -> PipelineConfig:
    """Person gate as Model A, YOLO11s as Model B, analytics on. Explicit flags still win."""
    cfg = config_from_args(argparse.Namespace(**{**vars(a), "preset": "assignment"}))
    defaults = build_parser().parse_args(["run"])
    argv = getattr(a, "argv", None)

    def given(name: str) -> bool:
        # an option typed on the command line wins even when its value equals the parser default
        # (e.g. --weights-b yolo11n.pt); without argv, fall back to "differs from the default"
        if argv is None:
            return getattr(a, name) != getattr(defaults, name)
        flag = "--" + name.replace("_", "-")
        return any(t == flag or t.startswith(flag + "=") for t in argv)

    pick = lambda name, preset: getattr(a, name) if given(name) else preset  # noqa: E731
    cfg.model_a = ModelSpec("yolo_gate", pick("weights_a", "yolo11n.pt"), pick("imgsz_a", 320),
                            conf=pick("route_threshold", 0.30))
    cfg.model_b = ModelSpec("yolo", pick("weights_b", "yolo11s.pt"), pick("imgsz_b", 640), conf=pick("det_conf", 0.10))
    cfg.route_labels = ("person",)
    cfg.route_threshold = pick("route_threshold", 0.30)
    cfg.analytics = True
    # Model B on every 2nd frame (the tracker holds boxes in between): ~1.5x faster on a 4-core CPU for
    # ~4% fewer people / ~13% fewer handbags seen per frame; --detect-every 1 = detect on every frame
    cfg.detect_every = pick("detect_every", 2)
    if a.fast:
        _fast_mode(cfg, pick)
    if a.show_classes == "":
        cfg.show_classes = PEOPLE_SHOW_CLASSES
    # boxes drawn only when the detector is 80-100 % sure; weaker ones still feed the tracker and counts
    cfg.min_conf_shown = pick("min_conf_shown", 0.80)
    if a.attributes_model == "":
        from pathlib import Path

        from .attributes import DEFAULT_MODEL

        if Path(DEFAULT_MODEL).exists():
            cfg.attributes_model = DEFAULT_MODEL
    return cfg


def fast_settings(physical_cores: int) -> tuple[int, int]:
    """(infer_every, detector workers) for --fast from the CPU size.

    Measured on a 4-core i5-8265U held at 1.6 GHz, 40 s of 1080p (gate off, 720p, superfast, live
    preview on): every 8th frame with 2 detector workers ingested 27.4 fps (Model B was the bottleneck:
    150 x ~550 ms / 2 workers = 41 s of work for 40 s of video); every 10th keeps up at 30.0 fps.
    Bigger CPUs can afford to look at more frames. Explicit --infer-every / --b-workers override these.
    """
    if physical_cores <= 4:
        infer_every = 10
    elif physical_cores <= 6:
        infer_every = 6
    else:
        infer_every = 4
    return infer_every, max(1, physical_cores // 2)


# OpenVINO export of the people detector (`yolo export model=yolo11s.pt format=openvino`). Measured on the
# 4-core test laptop: identical detections to yolo11s.pt (128/128 boxes, IoU > 0.9, conf diff 0.000) and
# ~4x faster per frame as ONE worker (OpenVINO already uses every core; two workers fought: p95 1.9 s).
# Costs ~25 s more startup (model compile).
OPENVINO_DETECTOR = "yolo11s_openvino_model"


def _fast_mode(cfg: PipelineConfig, pick) -> None:
    """--fast: aim for real time (the source fps, ~30) on a laptop CPU, never faster, never dropping a frame.

    Only every Nth frame is analysed (others HELD: the tracker moves the boxes along each track), analysed
    frames skip the gate (on street video it says "person" on every frame, so it only cost time: +18%
    measured), the video is written at the source size with x264 superfast, and ingest is paced to the
    source fps. With the OpenVINO detector present, Model B is one OpenVINO worker on every 5th frame.
    """
    import psutil

    from dataclasses import replace
    from pathlib import Path

    cores = psutil.cpu_count(logical=False) or 4
    infer_every, b_workers = fast_settings(cores)
    if Path(OPENVINO_DETECTOR).is_dir() and pick("weights_b", None) is None:
        # same detector, ~4x faster: detect twice as often and still keep up
        cfg.model_b = replace(cfg.model_b, weights=OPENVINO_DETECTOR)
        infer_every, b_workers = max(1, infer_every // 2), 1
    cfg.infer_every = pick("infer_every", infer_every)
    cfg.detect_every = pick("detect_every", 1)
    cfg.gate = False  # --fast implies --no-gate
    cfg.a_workers = pick("a_workers", 1)        # idle with the gate off, but the pool needs one
    cfg.b_workers = pick("b_workers", b_workers)
    cfg.output_height = pick("output_height", 0)  # full source size: no quality lost (1080p ~ -3 fps)
    cfg.video_preset = pick("video_preset", "superfast")
    cfg.pace = True
    # live stream: keep the delay at ~one inference instead of letting Model B's queue grow
    cfg.b_latest_only = cfg.mode == "live"


def _ensure_weights(cfg: PipelineConfig) -> None:
    """Download missing official weights once, before N workers would race to fetch the same file."""
    from pathlib import Path

    for spec in (cfg.model_a, cfg.model_b):
        if spec.kind.startswith("yolo") and not Path(spec.weights).exists():
            from ultralytics.utils.downloads import attempt_download_asset

            attempt_download_asset(spec.weights)


def print_summary(s: dict) -> None:
    lat = s["latency_ms"]
    f = lambda d, k: d.get(k, "-")  # noqa: E731
    print("\n================ pipeline summary ================")
    print(f"frames: {s['frames_total']}  routed->B: {s['frames_routed_to_b']}  skipped: {s['frames_skipped']}  "
          f"with detections: {s['frames_with_detections']}  no detections: {s['frames_no_detections']}")
    print(f"dropped: {s['frames_dropped']} ({s['drop_rate']:.1%})  invalid: {s['frames_invalid']}  "
          f"A err/timeout: {s['a_errors']}/{s['a_timeouts']}  B err/timeout: {s['b_errors']}/{s['b_timeouts']}  "
          f"incomplete: {s['frames_incomplete']}  stale A/B: {s.get('a_stale', 0)}/{s.get('b_stale', 0)}")
    print(f"source fps: {s['source_fps']}  input fps: {s['input_fps']}  output fps: {s['output_fps']} "
          f"(steady {s.get('output_fps_steady')})  wall: {s['wall_time_s']}s  (startup {s['startup_s']}s)")
    for k in ("a_infer", "b_infer", "a_queue_wait", "b_queue_wait", "end_to_end"):
        d = lat[k]
        print(f"  {k:<13} n={d.get('n', 0):<5} p50={f(d, 'p50')}  p95={f(d, 'p95')}  p99={f(d, 'p99')}  ms")
    rt = s.get("runtime", {})
    for k in ("queue_a_depth", "queue_b_depth", "cpu_tree_pct", "cpu_system_pct", "rss_tree_mb"):
        if k in rt:
            print(f"  {k:<16} mean={rt[k]['mean']}  max={rt[k]['max']}")
    if s.get("route_labels_missing"):
        print(f"WARNING: routing labels not in Model A's label set: {s['route_labels_missing']}")
    an = s.get("analytics")
    if an:
        p = an["people"]
        print("---------------- people analytics ----------------")
        print(f"unique people: {p['unique_total']}   peak in one frame: {p['peak_in_frame']} (t={p['peak_time_s']}s)   "
              f"mean per frame: {p['mean_per_frame']}")
        print(f"crossed counting line: in {p['crossed_line_in']}  out {p['crossed_line_out']}   "
              f"scenes detected: {len(an['scenes'])}")
        top = sorted(an["objects"].items(), key=lambda kv: -kv[1]["unique_tracked"])[:10]
        print("objects (unique tracked): " + ", ".join(f"{k} {v['unique_tracked']}" for k, v in top))
        pa = an.get("people_attributes")
        if pa:
            print(f"people by gender (estimated): men {pa['male']}  women {pa['female']}  unknown {pa['unknown']}"
                  f"   (of {pa['unique_people']} unique people)")
            print(f"carrying: bag {pa['has_bag']}  backpack {pa['has_backpack']}   hat {pa['has_hat']}   "
                  f"attributes unknown {pa['attributes_unknown']}")
    if s.get("output_video"):
        print(f"output: {s['output_video']}")


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    a.argv = list(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    if a.cmd == "download":
        from .download import download_video, download_weights

        if a.sample_video:
            download_video(a.video_dest)
        download_weights()
        return 0
    if a.cmd == "labels":
        from ultralytics import YOLO

        names = YOLO(a.weights).names
        for i, n in names.items():
            if a.grep.lower() in n.lower():
                print(f"{i:4d}  {n}")
        print(f"'person' present: {'person' in names.values()}  ({len(names)} classes)")
        return 0

    from .pipeline import Pipeline

    cfg = config_from_args(a)
    _ensure_weights(cfg)
    logging.getLogger("vpipe").info("config: %s", json.dumps(cfg.to_dict(), default=str))
    summary = Pipeline(cfg).run()
    print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
