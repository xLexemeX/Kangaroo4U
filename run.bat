@echo off
rem Start Kangaroo4U: the GPU solver plus the live dashboard.
rem   run.bat              search and dashboard
rem   run.bat dashboard    dashboard only
rem   run.bat bench        quick GPU self-test
cd /d "%~dp0"
where py >nul 2>nul && (py -3 kangaroo4u.py %* & goto :done)
where python >nul 2>nul && (python kangaroo4u.py %* & goto :done)
echo Python 3 is required: https://www.python.org/downloads/
:done
pause
