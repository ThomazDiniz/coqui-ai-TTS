@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

rem Roda o Gradio do fine-tuning XTTS via Conda (sem Docker).
rem
rem Uso:
rem   conda-finetune.bat
rem   conda-finetune.bat <env_name>
rem   conda-finetune.bat <env_name> <port>
rem   conda-finetune.bat <env_name> <port> <out_path>
rem
rem Defaults:
rem   env_name = coqui-xtts
rem   port     = 5003
rem   out_path = .\data\xtts-work

set "ENV_NAME=%~1"
if "%ENV_NAME%"=="" set "ENV_NAME=coqui-xtts"

set "PORT=%~2"
if "%PORT%"=="" set "PORT=5003"

set "OUT_PATH=%~3"
if "%OUT_PATH%"=="" set "OUT_PATH=%~dp0data\xtts-work"

set "LOG_DIR=%~dp0data\xtts-work\logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%" >nul 2>nul
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"') do set "TS=%%i"
set "LOG_FILE=%LOG_DIR%\conda-finetune-%TS%.log"

echo.
echo [coqui-tts] Conda env: %ENV_NAME%
echo [coqui-tts] URL:       http://localhost:%PORT%
echo [coqui-tts] Out path:  %OUT_PATH%
echo [coqui-tts] Log file:  %LOG_FILE%
echo.

rem Desativa telemetria do Gradio (menos ruido/requests).
set "GRADIO_ANALYTICS_ENABLED=False"

rem Usa "conda run" para evitar problemas de activate em .bat.
rem Mantem o processo em primeiro plano: pare com Ctrl+C.
rem Mostra no console E grava no arquivo (Tee-Object).
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Continue'; " ^
  "$log='%LOG_FILE%'; " ^
  "Write-Host ('[coqui-tts] Streaming logs to: ' + $log); " ^
  "& conda run -n '%ENV_NAME%' --no-capture-output python -m TTS.demos.xtts_ft_demo.xtts_demo --port '%PORT%' --out_path '%OUT_PATH%' 2>&1 | Tee-Object -FilePath $log -Append"
set "EC=%ERRORLEVEL%"

echo.
if not "%EC%"=="0" (
  echo [coqui-tts] ERRO (codigo %EC%). Veja o log:
  echo   %LOG_FILE%
  echo [coqui-tts] Dica: no env "%ENV_NAME%", instale deps do demo:
  echo   pip install -e ".[cuda,xtts_ft]"
  echo.
  echo [coqui-tts] Pressione qualquer tecla para fechar (Ctrl+C tambem funciona)...
  pause >nul
) else (
  echo [coqui-tts] Encerrado com sucesso.
)
exit /b %EC%

