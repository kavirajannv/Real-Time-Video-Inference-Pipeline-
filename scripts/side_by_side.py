"""Play the input video and the annotated output side by side in one window.

Left panel: the original input. Right panel: the pipeline's annotated result. Both are
played in sync by frame number (the pipeline writes exactly one output frame per input frame).

    python scripts/side_by_side.py                         # play outputs/people (run it first)
    python scripts/side_by_side.py --run test              # run the first 10 s, then play
    python scripts/side_by_side.py --run fast              # run 40 s in fast mode, then play
    python scripts/side_by_side.py --run full              # run the whole video, then play
    python scripts/side_by_side.py --run fast --live       # show the window live while it runs
    python scripts/side_by_side.py --run stream            # assignment pipeline as a live 30 FPS stream
    python scripts/side_by_side.py --run stream-demo       # same, routing on clothing labels so Model B runs
    python scripts/side_by_side.py --run people-stream     # people preset (fast) as a live stream
    python scripts/side_by_side.py --run-dir outputs/fast --save outputs/fast/side_by_side.mp4

--live (or `python -m vpipe run ... --preview`) shows the window while the pipeline is still running.

Keys: space = pause/resume, . = step one frame while paused, q / Esc = quit.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2

from vpipe.preview import WINDOW, side_by_side
from vpipe.writer import _open_video_writer  # ffmpeg/libx264, same as the pipeline's output

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "data" / "people_walking.mp4"
PEOPLE = ["--preset", "people"]
# the brief's live replay: Model A classifies every frame, input paced at 30 FPS, a frame is DROPPED
# when queue A is full and stale tasks are skipped, so the picture stays current (offline mode never drops).
# 1080p on 4 cores, 20 s measured with the preview on: queue A 2 + veryfast x264 dropped 57% of frames;
# ultrafast x264 (same dimensions, bigger file) + queue A 4 dropped 30% with lower latency (p99 1.7 -> 0.9 s)
ASSIGNMENT_LIVE = ["--mode", "live", "--live-fps", "30", "--queue-a", "4", "--stale-after", "0.5",
                   "--a-workers", "2", "--b-workers", "2", "--video-preset", "ultrafast"]
# yolo11n-cls has no "person" class; on people_walking.mp4 its top labels are clothing such as
# miniskirt / swimming_trunks / poncho (median confidence ~0.2), so the demo routes on those
DEMO_ROUTING = ["--route-labels", "miniskirt,swimming_trunks,poncho,sombrero", "--route-threshold", "0.20"]

# name: (output folder, input video, pipeline options, always shown live)
RUNS = {
    # same settings as predict.bat
    "test": ("outputs/test/people", VIDEO, PEOPLE + ["--a-workers", "1", "--b-workers", "3", "--max-frames", "300"], False),
    "fast": ("outputs/fast", VIDEO, PEOPLE + ["--fast", "--max-seconds", "40"], False),
    "full": ("outputs/people", VIDEO, PEOPLE + ["--a-workers", "1", "--b-workers", "3"], False),
    # live streams: read at the source fps until the end of the video or q in the window
    "stream": ("outputs/stream", VIDEO, ASSIGNMENT_LIVE, True),
    # the brief's rule (person >= 0.30) never routes: yolo11n-cls has no "person" class (see README)
    "stream-demo": ("outputs/stream_demo", VIDEO, ASSIGNMENT_LIVE + DEMO_ROUTING, True),
    # first 60 s, paced at the source fps but never drops or skips a frame (--fast: offline + pace), full size
    "people-stream": ("outputs/people_stream", VIDEO, PEOPLE + ["--fast", "--max-seconds", "60"], True),
}


def run_pipeline(mode: str, video: str | None = None, live: bool = False) -> Path:
    out_dir, default_video, extra, _ = RUNS[mode]
    cmd = [sys.executable, "-m", "vpipe", "run", "--video", str(video or default_video),
           *extra, "--output-dir", out_dir, *(["--preview"] if live else [])]
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    return ROOT / out_dir


def output_video(run_dir: Path) -> Path:
    """The video a run actually wrote (a timestamped name is used if the default one was locked)."""
    summary = run_dir / "summary.json"
    if summary.exists():
        path = json.loads(summary.read_text()).get("output_video")
        if path and (ROOT / path).exists():  # relative paths are relative to the project root
            return ROOT / path
    return run_dir / "output_annotated.mp4"


def play(input_path: str | Path, output_path: Path, height: int, save: Path | None, show: bool) -> None:
    src, res = cv2.VideoCapture(str(input_path)), cv2.VideoCapture(str(output_path))
    if not src.isOpened():
        sys.exit(f"cannot open input video: {input_path}")
    if not res.isOpened():
        sys.exit(f"cannot open output video: {output_path} (run the pipeline first, or pass --run)")
    fps = res.get(cv2.CAP_PROP_FPS) or src.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(min(src.get(cv2.CAP_PROP_FRAME_COUNT), res.get(cv2.CAP_PROP_FRAME_COUNT)))
    print(f"input : {input_path}\noutput: {output_path}\n~{n} frames @ {fps:.2f} fps", flush=True)

    writer = None
    if show:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    paused, step, frame_no, period = False, False, 0, 1.0 / fps
    next_t = time.perf_counter()
    try:
        while True:
            if not paused or step or frame_no == 0:
                step = False
                ok_a, a = src.read()
                ok_b, b = res.read()
                if not (ok_a and ok_b):  # the output covers only the frames that were run
                    break
                canvas = side_by_side(a, b, height, frame_no)
                frame_no += 1
                if save is not None:
                    if writer is None:
                        save.parent.mkdir(parents=True, exist_ok=True)
                        writer = _open_video_writer(save, fps, (canvas.shape[1], canvas.shape[0]))
                    writer.write(canvas)
                if show:
                    if frame_no == 1:
                        cv2.resizeWindow(WINDOW, canvas.shape[1], canvas.shape[0])
                    cv2.imshow(WINDOW, canvas)
            if not show:
                continue
            # keep the source frame rate; never wait less than 1 ms (imshow needs the event loop)
            next_t += period
            key = cv2.waitKey(max(1, int((next_t - time.perf_counter()) * 1000)) if not paused else 30) & 0xFF
            if paused or time.perf_counter() > next_t + 0.5:  # resync after a pause or a slow frame
                next_t = time.perf_counter()
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord(" "):
                paused = not paused
            elif key == ord(".") and paused:
                step = True
    finally:
        src.release()
        res.release()
        if writer is not None:
            writer.release()
            print(f"saved: {save}")
        if show:
            cv2.destroyAllWindows()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", help=f"input video (default: the run's own video, else {VIDEO.name})")
    p.add_argument("--run", choices=list(RUNS), help="run the pipeline first (test/fast/full: same as predict.bat)")
    p.add_argument("--live", action="store_true",
                   help="with --run: show the window live while the pipeline runs (no replay afterwards)")
    p.add_argument("--run-dir", type=Path, default=ROOT / "outputs" / "people",
                   help="output folder of an earlier run (default: %(default)s)")
    p.add_argument("--output", type=Path, help="annotated video to show (overrides --run-dir)")
    p.add_argument("--height", type=int, default=540, help="height (even) of each panel in pixels (default: %(default)s)")
    p.add_argument("--save", type=Path, help="also write the side-by-side video to this .mp4")
    p.add_argument("--no-window", action="store_true", help="don't open a window (use with --save)")
    a = p.parse_args()

    if a.run and RUNS[a.run][3]:
        a.live = True  # streams are always shown live
    if a.live:
        if not a.run:
            p.error("--live needs --run")
        run_pipeline(a.run, a.video, live=True)
        return
    run_dir = run_pipeline(a.run, a.video) if a.run else a.run_dir
    video = a.video or (RUNS[a.run][1] if a.run else VIDEO)
    play(video, a.output or output_video(run_dir), a.height // 2 * 2, a.save, show=not a.no_window)


if __name__ == "__main__":
    main()
