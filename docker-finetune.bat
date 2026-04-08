@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

rem XTTS fine-tuning (Gradio): http://localhost:5003
rem   docker-finetune.bat
rem   docker-finetune.bat nobuild

call "%~dp0run-docker.bat" finetune %*
exit /b !ERRORLEVEL!
