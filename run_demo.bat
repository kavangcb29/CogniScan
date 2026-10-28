@echo off
title CogniScan Final - Complete Merged Project
echo =====================================================================
echo   COGNISCAN FINAL - MERGED ACADEMIC & CLINICAL SUITE
echo =====================================================================
echo.
echo Starting CogniScan on Port 5002...
echo.
echo Home Page:    http://localhost:5002/
echo Patient Page: http://localhost:5002/patient
echo Doctor Page:  http://localhost:5002/doctor
echo Records Page: http://localhost:5002/records
echo.
echo Press Ctrl+C to stop the server.
echo =====================================================================
set PORT=5002
python app.py
pause
