@echo off
title Opus Pro - Unique Batch Engine
cd /d "C:\Users\DELL\auto-editor"
powershell.exe -ExecutionPolicy Bypass -Command "streamlit run clipper_dashboard.py"
pause