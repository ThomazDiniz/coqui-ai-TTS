@echo off
setlocal
rem Acompanha um .log em tempo real em outra janela (Get-Content -Wait).
rem Opcional: mesma saida em outra janela (o .bat principal ja mostra + grava no log).
rem
rem Uso:
rem   xtts-log-tail.bat "E:\git\coqui-ai-TTS\data\xtts-experiments\logs\conda-paraiba-staged.log"
rem   xtts-log-tail.bat "E:\...\paraiba-...\experiment_console.log"
rem
if "%~1"=="" (
  echo Uso: %~nx0 ^<arquivo.log^>
  echo Arraste o arquivo para esta janela ou cole o caminho completo entre aspas.
  exit /b 1
)
set "LOGFILE=%~f1"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\xtts_log_tail.ps1" -Path "%LOGFILE%"
set "EC=!ERRORLEVEL!"
exit /b %EC%
