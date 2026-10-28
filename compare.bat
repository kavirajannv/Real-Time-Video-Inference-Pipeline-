@echo off
rem Show the input video (left) and the annotated output (right) in one window.
rem   compare               play the last full people run    (outputs\people)
rem   compare test          run the first 10 s, then play    (outputs\test\people)
rem   compare fast          run 40 s in fast mode, then play (outputs\fast)
rem   compare full          run the whole video, then play   (outputs\people)
rem   compare live          LIVE while running: 40 s people fast mode (outputs\fast)
rem   compare live full     LIVE while running: whole people video, accurate mode (slower than real time)
rem   compare stream        LIVE STREAM, assignment pipeline: people_walking.mp4 at 30 FPS,
rem                         Model A on every frame, rule person >= 0.30          (outputs\stream)
rem   compare stream-demo   same, routed on clothing labels so Model B runs      (outputs\stream_demo)
rem   compare people-stream LIVE STREAM, people preset in fast mode, first 60 s  (outputs\people_stream)
rem Keys (replay): space = pause, . = step a frame while paused, q / Esc = quit.
rem Keys (live):   q / Esc = stop the run (the video so far is saved); the close button hides the window only.
cd /d "%~dp0"
if "%~1"=="" (
    .venv\Scripts\python.exe scripts\side_by_side.py
) else if /i "%~1"=="live" (
    if "%~2"=="" (
        .venv\Scripts\python.exe scripts\side_by_side.py --run fast --live
    ) else (
        .venv\Scripts\python.exe scripts\side_by_side.py --run %2 --live
    )
) else (
    .venv\Scripts\python.exe scripts\side_by_side.py --run %1
)
