"""Pipeline configuration (plain dataclasses so they pickle into worker processes)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_VIDEO_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/solutions_ci_demo.mp4"

# ImageNet-1k (what yolo11n-cls is trained on) has no "person" class. These are the
# ImageNet labels whose images are dominated by a person; used only when the user
# explicitly opts in with --route-labels / --person-proxy. See docs/technical_summary.md.
PERSON_PROXY_LABELS = (
    "person",
    "groom",
    "scuba_diver",
    "ballplayer",
    "lab_coat",
    "jersey",
    "sweatshirt",
    "suit",
    "cardigan",
    "trench_coat",
    "jean",
    "miniskirt",
    "abaya",
    "academic_gown",
    "kimono",
    "military_uniform",
    "pajama",
    "poncho",
    "fur_coat",
    "bulletproof_vest",
    "cowboy_hat",
    "sombrero",
    "wig",
)


@dataclass
class FaultConfig:
    """Fault injection applied inside a worker process (all disabled by default)."""

    delay_s: float = 0.0       # extra sleep for selected frames
    delay_every: int = 0       # 0 = never, N = frames with frame_id % N == 0 (1 = every frame)
    jitter_s: float = 0.0      # random 0..jitter_s sleep per task -> out-of-order completion
    error_every: int = 0       # raise for frame_id % N == 0
    error_rate: float = 0.0    # raise with this probability (deterministic per frame_id)
    crash_after: int = 0       # hard-kill the first incarnation of each worker after N tasks
    seed: int = 0

    @property
    def active(self) -> bool:
        return any((self.delay_every, self.jitter_s, self.error_every, self.error_rate, self.crash_after))


@dataclass
class ModelSpec:
    kind: str = "yolo"             # "yolo" or "fake" (tests / dry runs)
    weights: str = ""
    imgsz: int = 640
    conf: float = 0.25             # detection confidence threshold (Model B only)
    fake_infer_s: float = 0.0      # simulated inference time for fake models


@dataclass
class PipelineConfig:
    video: str = "data/solutions_ci_demo.mp4"
    output_dir: str = "outputs"
    output_name: str = "output_annotated.mp4"

    # worker pools
    a_workers: int = 1
    b_workers: int = 1
    model_a: ModelSpec = field(default_factory=lambda: ModelSpec(weights="yolo11n-cls.pt", imgsz=224))
    model_b: ModelSpec = field(default_factory=lambda: ModelSpec(weights="yolo11n.pt", imgsz=640))
    torch_threads: int = 0         # 0 = auto (split physical cores across workers), -1 = torch default

    # routing
    route_labels: tuple[str, ...] = ("person",)
    route_threshold: float = 0.30

    # queues / backpressure
    queue_a_size: int = 8
    queue_b_size: int = 8
    max_buffered_frames: int = 64  # hard bound on frames held in memory (ingested, not yet written)
    mode: str = "offline"          # "offline" = block, never drop; "live" = pace at source FPS, drop when full
    live_fps: float = 0.0          # 0 = use the source FPS
    stale_after_s: float = 0.0     # live mode: workers discard tasks older than this (0 = off)

    # timeouts (seconds, measured from enqueue into that stage's queue)
    timeout_a_s: float = 30.0
    timeout_b_s: float = 30.0
    startup_timeout_s: float = 180.0

    # optimisation switch: send pre-resized images to workers instead of full frames
    resize_for_ipc: bool = True

    # misc
    max_frames: int = 0            # 0 = whole video
    max_seconds: float = 0.0       # 0 = whole video; else only the first N seconds (converted with the source fps)
    repeat: int = 1                # replay the video N times back-to-back (benchmarks)
    stop_after_s: float = 0.0      # simulate an operator stop after N seconds (0 = off)
    sample_interval_s: float = 0.1
    fault_a: FaultConfig = field(default_factory=FaultConfig)
    fault_b: FaultConfig = field(default_factory=FaultConfig)
    invalid_every: int = 0         # source fault: emit a corrupt frame every N frames

    # people / object analytics (tracking + counting in the ordered writer stage)
    analytics: bool = False
    count_line: tuple[float, float, float, float] = (0.0, 0.75, 1.0, 0.75)  # x1,y1,x2,y2 as fractions
    min_track_frames: int = 15     # a track must live this many frames to count as a unique object
    scene_cut_threshold: float = 20.0   # median grayscale change vs 1 s earlier (see SceneCutDetector)
    attributes_model: str = ""     # OpenVINO person-attributes model (.xml): gender estimate, bag, ... ("" = off)
    detect_every: int = 1          # Model B on every Nth frame Model A routes; the tracker holds boxes in between
    infer_every: int = 1           # fast mode: only every Nth frame goes to the models at all (others: HELD)
    gate: bool = True              # False (fast mode): inferred frames skip Model A and go straight to Model B
    pace: bool = False             # offline mode but never faster than the source fps (reader waits, never drops)
    output_height: int = 0         # write the video at this height (0 = source size); analytics use full frames
    video_preset: str = "veryfast" # libx264 preset for the output video (superfast/ultrafast = less CPU, bigger file)
    show_classes: tuple[str, ...] = ()  # classes drawn in the video (empty = all); summaries still count every class
    write_video: bool = True

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir) / self.output_name

    def resolved_threads(self, cpu_count: int) -> int:
        if self.torch_threads != 0:
            return self.torch_threads
        return max(1, cpu_count // max(1, self.a_workers + self.b_workers))

    def to_dict(self) -> dict:
        return asdict(self)
