@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem Segue logs de forma continua (nao deve fechar sozinho):
rem   auto  = ficheiro data\xtts-work\ui_console.log (aguarda criacao; reconecta se o stream cair)
rem   docker = docker compose logs -f (aguarda container; reconecta se cair)
rem   file  = so ficheiro
rem
rem Ctrl+C para sair.

set "MODE=auto"
if /i "%~1"=="docker" set "MODE=docker"
if /i "%~1"=="file" set "MODE=file"

echo [coqui-tts] Modo: %MODE%  ^|  docker-finetune-logs.bat [docker^|file^|auto]
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0docker-finetune-logs.ps1" -Mode %MODE%
set "EC=%ERRORLEVEL%"

echo.
if not "%EC%"=="0" (
  echo [coqui-tts] Script terminou com codigo %EC%.
  pause
)
exit /b %EC%
