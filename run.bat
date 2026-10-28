@echo off
rem vpipe launcher: pick what to run from a menu, or pass the number directly (run 1).
rem   From PowerShell:  .\run.bat      or  .\run.bat 1
rem   From cmd:          run            or  run 1
setlocal
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo Python environment not found: %PY%
    echo Set it up first - see README.md, section Installation.
    exit /b 1
)

set "CHOICE=%~1"
if not "%CHOICE%"=="" goto run

echo.
echo  vpipe - real-time video inference pipeline
echo  ==========================================
echo   1  LIVE stream, first 60 s, input ^| output window (30 FPS)   -^> outputs\people_stream
echo   2  Replay the last live stream side by side
echo   3  Quick test: first 10 s, accurate mode, then replay          -^> outputs\test\people
echo   4  Whole video, accurate mode (~10 min), then replay           -^> outputs\people
echo   5  Assignment pipeline, live 30 FPS (brief's rule)             -^> outputs\stream
echo   6  Assignment pipeline, live, Model B active (boxes shown)     -^> outputs\stream_demo
echo   7  Open the output folder
echo   8  Run the tests
echo   9  Save the last live stream as one input ^| output video   -^> outputs\people_stream\side_by_side.mp4
echo   0  Exit
echo.
echo  Tip: close Edge, Teams and video players for full speed.
echo  In the live window: q = stop (everything so far is saved).
echo.
set /p "CHOICE=Choose 0-9: "

:run
if "%CHOICE%"=="1" (
    "%PY%" scripts\side_by_side.py --run people-stream
) else if "%CHOICE%"=="2" (
    "%PY%" scripts\side_by_side.py --run-dir outputs\people_stream
) else if "%CHOICE%"=="3" (
    "%PY%" scripts\side_by_side.py --run test
) else if "%CHOICE%"=="4" (
    "%PY%" scripts\side_by_side.py --run full
) else if "%CHOICE%"=="5" (
    "%PY%" scripts\side_by_side.py --run stream
) else if "%CHOICE%"=="6" (
    "%PY%" scripts\side_by_side.py --run stream-demo
) else if "%CHOICE%"=="7" (
    if exist outputs\people_stream (start "" outputs\people_stream) else (start "" outputs)
) else if "%CHOICE%"=="8" (
    "%PY%" -m pytest
) else if "%CHOICE%"=="9" (
    "%PY%" scripts\side_by_side.py --run-dir outputs\people_stream --save outputs\people_stream\side_by_side.mp4 --no-window
) else if "%CHOICE%"=="0" (
    exit /b 0
) else (
    echo Unknown choice "%CHOICE%". Use a number from 0 to 9.
    exit /b 1
)
if errorlevel 1 (
    echo.
    echo The run failed - see the messages above.
    exit /b 1
)
echo.
echo Done. Results are in outputs\ (output_annotated.mp4, frames_log.csv, summary.json).
