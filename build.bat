@echo off
rem Build the solver into bin\. Needs the CUDA Toolkit, CMake 3.18+ and
rem Visual Studio 2022 (or its Build Tools) with the C++ workload.
cd /d "%~dp0engine"
cmake -S . -B build %* || goto :fail
cmake --build build --config Release || goto :fail
echo built: %~dp0bin\rckangaroo.exe
pause
exit /b 0
:fail
echo build failed
pause
exit /b 1
