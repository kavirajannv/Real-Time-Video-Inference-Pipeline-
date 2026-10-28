vpipe: CPU Video Analytics Pipeline
A Python-based video analytics pipeline for detecting and tracking people, bags, vehicles, and other objects in video. It generates an annotated H.264 video, frame-level logs, analytics summaries, charts, and heatmaps.

Repository: https://github.com/kavirajannv/Real-Time-Video-Inference-Pipeline-
Features
- People analytics: per-frame counts, unique tracked people, and directional counting-line crossings.
- Person attributes: appearance-based man/woman estimates, with unknown when evidence is weak.
- Bag analytics: detects handbags, backpacks, and suitcases; estimates whether a person is carrying a bag/backpack or wearing a hat.
- Object detection: cars, buses, trucks, bicycles, motorcycles, and all 80 COCO classes in summaries.
- CPU-only inference: runs without a GPU using PyTorch CPU; OpenVINO can accelerate supported inference paths.
- Two-stage pipeline: Model A screens frames; selected frames are sent to Model B.
- Tracking: ByteTrack carries detections between detector runs.
- Reliability: bounded queues, backpressure handling, timeouts, worker crash recovery, graceful shutdown, and ordered output.
- Observability: frame logs, queue wait/inference timings, latency percentiles, CPU/memory metrics, and run summaries.
- Testing: automated tests for routing, frame ordering, failures, timeouts, shutdown, tracking, line counting, scene changes, and attributes.
Person-attribute and gender labels are appearance-based estimates, not verified facts. They can be inaccurate, so treat them as estimates.

Architecture
Video Reader
    |
  Queue A
    |
Model A Worker Pool
    |
Aggregator / Router  <---- Results Queue
    |
  Queue B (routed frames only)
    |
Model B Worker Pool
    |
Results Queue
    |
Ordered Aggregator / Reorder Buffer
    |
ByteTrack -> Person Attributes -> Counting
    |
Annotated Video + Logs + Summary + Charts
The pipeline uses worker processes for CPU-bound inference. Bounded queues limit memory growth. In offline mode, the reader waits when downstream stages are full to preserve all frames. In live mode, the configured overload policy may mark frames as DROPPED or tasks as STALE. Every input frame receives an output frame and a log record.
Technology Stack
- Language: Python 3.10+ (developed with Python 3.12)
- Inference: PyTorch (CPU), Ultralytics YOLO11
- Person attributes: OpenVINO person-attributes-recognition-crossroad-0230
- Tracking: ByteTrack, lap
- Video processing: OpenCV, FFmpeg/libx264 via imageio-ffmpeg, NumPy
- Charts: Matplotlib
- Monitoring: psutil
- Concurrency: Python multiprocessing (spawn) and threading
- Tests: pytest
Quick Start — Windows
Open Command Prompt or PowerShell and run:
git clone https://github.com/kavirajannv/Real-Time-Video-Inference-Pipeline-.git data_pipeline
cd data_pipeline
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Place an input video at data\people_walking.mp4. See Input Video.
Run the pipeline
.\predict
Runs a quick test on the first 10 seconds.
.\predict fast
Runs up to the first 40 seconds in fast mode, targeting up to the video's own frame rate (capped at 30 FPS).
.\predict full
Processes the full video in accurate mode.

You can also use the CLI directly:
.\.venv\Scripts\python.exe -m vpipe run --preset people --video data/people_walking.mp4 --a-workers 1 --b-workers 3 --output-dir outputs/people
To view the generated annotated video:
start outputs\people\output_annotated.mp4
The first run downloads YOLO weights automatically. The optional person-attributes model must be downloaded separately to enable appearance-based attribute estimates.

Linux / macOS
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
python -m vpipe run --preset people --video data/people_walking.mp4 --a-workers 1 --b-workers 3 --output-dir outputs/people
Input Video
The default demo video is royalty-free street footage at 1920×1080, 30 FPS, approximately 117 seconds long. Download it with yt-dlp:
.\.venv\Scripts\python.exe -m pip install yt-dlp
.\.venv\Scripts\python.exe -m yt_dlp -f "bv*[vcodec^=avc1][height<=1080][ext=mp4]" -o data/people_walking.mp4 "https://youtu.be/YzcawvDGe4Y"
Alternatively, pass another video with --video path/to/video.mp4.
Useful Options
Option	Description
--max-frames 300	Process only the first 300 frames
--max-seconds 40	Process only the first 40 seconds
--output-dir DIR	Set the output directory
--a-workers N / --b-workers M	Configure Model A and Model B worker processes
--show-classes all	Draw all detected classes
--show-classes person,handbag,car	Draw only selected classes
--attributes-model none	Disable person-attribute estimates
--count-line x1,y1,x2,y2	Set the counting line as frame-width/height fractions
--min-track-frames N	Minimum track length to count as a unique person
--detect-every N	Run Model B every Nth selected frame; tracker holds boxes between detections
--fast	Enable fast mode
--infer-every N	Send only every Nth frame to the models
--no-gate	Bypass Model A
--output-height PX	Resize output video height
--video-preset P	Set the video-encoding preset
--min-conf-shown P	Draw only boxes above the selected confidence; tracking/counts still use all detections
--preview	Show the input and annotated output in a live preview window


See all options with:
python -m vpipe run --help
How the Two-Stage Pipeline Works
People preset
- Model A: YOLO11n detector at 320 px, used as a person gate.
- Model B: YOLO11s detector at 640 px, detecting all 80 COCO classes.
- Detection stride: Model B runs on every second frame by default. ByteTrack holds boxes between detection frames.
- Tracking and counting: ByteTrack tracks objects; analytics include unique people, scene-level counts, and line crossings.
- Person attributes: when the optional OpenVINO model is available, attributes are estimated from eligible person crops.
Assignment mode
The original assignment mode uses YOLO11n-cls as Model A and YOLO11n detection as Model B. Model A's top-1 classification label and confidence determine whether a frame is routed to Model B.

ImageNet does not contain a person class. Therefore, a default routing rule such as person >= 0.30 will not route frames as expected in assignment mode. Use appropriate routing labels, a person proxy, or the people preset for street-person analytics.
Fast Mode and Performance
Fast mode prioritizes near-real-time output by processing fewer frames and holding the last tracked boxes between inference frames. This improves speed but may miss small or short-lived objects. Use accurate mode when object totals and tracking results matter more than speed.

Example performance measurements from the project's documented test environment (Intel Core i5-8265U, 4-core laptop; results vary by machine and settings):
Configuration	Approx. speed	Notes
Accurate mode, detector every frame	4.3 FPS	More detections; slower
Accurate mode, detector every 2nd frame	6.3 FPS	Default balance of speed and accuracy
Fast mode, earlier benchmark settings	26.6 FPS	Near-real-time target; reduced analysis


A video's source FPS and the pipeline's processing FPS are different. A 30 FPS source contains 30 frames per second of video, but a CPU pipeline processing at 6 FPS needs about five seconds to process one second of footage. Performance depends on model size, inference stride, input/output resolution, CPU capacity, encoding, and worker configuration.
Output Files
Every run writes to the selected output directory:
File	Description
output_annotated.mp4	Annotated H.264 video with detections, labels, and an analytics panel
frames_log.json / frames_log.csv	Per-frame status, detections, worker IDs, errors, queue waits, inference times, and latency
summary.json	FPS, latency percentiles, drops/stale tasks, errors, CPU, and memory metrics


The people preset also produces:
File	Description
analytics_summary.json	Unique people, peak/mean counts, line crossings, scene counts, attributes, and object statistics
people_tracks.csv	Per-person track information and estimated attributes
people_gallery.png	Sample person crops grouped by estimated attribute
counts.csv	Per-frame people and object counts
people_over_time.png	Per-frame and cumulative counts
heatmaps.png	Walking heatmaps by scene


If inference fails or times out for a frame, its counts are recorded as UNAVAILABLE, not as zero.
Tests and Benchmarks
.\.venv\Scripts\python.exe -m pytest -o addopts= -q
.\.venv\Scripts\python.exe -m pytest tests/test_analytics.py -q
.\.venv\Scripts\python.exe scripts/failure_demo.py
.\.venv\Scripts\python.exe scripts/benchmark.py
The tests use fake models for repeatable checks of frame association and pipeline behavior without downloading model weights.
Project Structure
src/vpipe/
  cli.py          # CLI commands and presets
  config.py       # Pipeline/model/fault configuration
  pipeline.py     # Pipeline orchestration and shutdown
  reader.py       # Video sources, ingest, frame store
  pools.py        # Worker pools, crash detection, respawn
  workers.py      # Worker process loop and fault injection
  models.py       # YOLO wrappers and fake models
  routing.py      # Routing rules and label checks
  aggregator.py   # Routing, frame association, deadlines, reordering
  analytics.py    # Scene changes, tracking, counting, charts
  attributes.py   # Person attributes
  writer.py       # Annotated video and logs
  preview.py      # Live preview
  metrics.py      # Queue, CPU, memory, and latency metrics
tests/            # Automated tests
scripts/          # Benchmark and failure-demo scripts
docs/             # Architecture, performance report, technical summary
models/           # Optional person-attributes model files
data/             # Input videos (not committed)
outputs/          # Generated output (not committed)
Troubleshooting
Problem	Suggested fix
No module named vpipe	Use the project's .venv Python or activate the virtual environment
run opens another tool on Windows	Use .\run.bat or .\predict in PowerShell
Many DROPPED / STALE frames in live mode	Close other apps, use a faster encoding preset, or use offline mode to preserve all frames
Video cannot be opened	Check the file path and run the command from the project folder
Another run is writing to the output directory	Wait for it to finish or select another --output-dir


Output video does not play	Install project dependencies, including imageio-ffmpeg
No progress appears immediately	Model loading can take 10–30 seconds on the first run


Documentation
- docs/architecture.md — architecture, queues, worker pools, routing, failures, and shutdown
- docs/performance_report.md — worker configuration, live replay, and optimization measurements
- docs/technical_summary.md — design decisions, bottlenecks, optimization, limitations, and GPU deployment
- docs/arch_diagram.svg — architecture diagram
- docs/vpipe_learning_guide.pdf — pipeline walkthrough
Contributing
1. Create a branch:git checkout -b my-change
2. Make and test your changes.
3. Commit:git add -A
   git commit -m "Describe the change"
4. Push the branch:git push -u origin my-change
5. Open a pull request into main.
License
No license information was provided in the project notes. Add a LICENSE file and update this section if the repository has a license.
