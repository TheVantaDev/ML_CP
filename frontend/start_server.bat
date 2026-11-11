@echo off
echo Starting Sonobuoy Sound Detective Backend...
cd /d "%~dp0.."
"%LOCALAPPDATA%\Microsoft\WindowsApps\python3.exe" frontend/app.py
pause
