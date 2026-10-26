@echo off
rem Run the people analytics on data\people_walking.mp4 and open the output video.
rem   predict full    whole video, accurate mode (about 9-10 minutes)   -> outputs\people
rem   predict fast    first 40 s, fast mode at up to 30 fps (about 1 minute) -> outputs\fast
rem   predict         first 10 s, accurate mode (about 1 minute)         -> outputs\test\people
cd /d "%~dp0"

set "MODE=--a-workers 1 --b-workers 3"
if /i "%~1"=="full" (
    set "OUT=outputs\people"
    set "LIMIT="
) else if /i "%~1"=="fast" (
    set "OUT=outputs\fast"
    set "LIMIT=--max-seconds 40"
    rem --fast picks workers and frame interval from the CPU core count
    set "MODE=--fast"
) else (
    set "OUT=outputs\test\people"
    set "LIMIT=--max-frames 300"
)

.venv\Scripts\python.exe -m vpipe run --preset people --video data\people_walking.mp4 %MODE% %LIMIT% --output-dir %OUT%
if errorlevel 1 (
    echo.
    echo Run failed - see the messages above.
    exit /b 1
)
rem open the video this run wrote (a new name is used if the old one was open in a player)
for /f "usebackq delims=" %%v in (`.venv\Scripts\python.exe -c "import json; print(json.load(open(r'%OUT%\summary.json'))['output_video'])"`) do start "" "%%v"
