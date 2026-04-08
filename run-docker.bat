@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

rem Usage:
rem   run-docker.bat              ^> build + tts-server XTTS v2 (port 5002)
rem   run-docker.bat server       ^> same
rem   run-docker.bat finetune     ^> build + Gradio fine-tune (port 5003), primeiro plano
rem   run-docker.bat nobuild      ^> skip build, server only
rem   run-docker.bat finetune nobuild
rem
rem Fine-tune em background + logs:
rem   docker-finetune-detached.bat
rem   docker-finetune-logs.bat

set "MODE=server"
set "SKIP_BUILD=0"

if /i "%~1"=="finetune" set "MODE=finetune"
if /i "%~1"=="server" set "MODE=server"
if /i "%~1"=="nobuild" set "SKIP_BUILD=1"
if /i "%~2"=="nobuild" set "SKIP_BUILD=1"

if "!SKIP_BUILD!"=="0" (
  echo.
  echo [coqui-tts] Building image (coqui-tts:local^)...
  docker compose build
  if errorlevel 1 (
    echo.
    echo [coqui-tts] Build falhou — veja as mensagens acima.
    set "EC=1"
    goto :end_pause
  )
) else (
  echo.
  echo [coqui-tts] Skipping build ^(nobuild^).
)

if "!MODE!"=="finetune" (
  echo.
  echo [coqui-tts] Fine-tuning Gradio: http://localhost:5003
  echo Saida em .\data\xtts-work  ^|  dataset em .\data\brPB22_g1bF01_char -^> /dataset/brPB22
  echo.
  docker compose --profile finetune up xtts-finetune
  set "EC=!ERRORLEVEL!"
  goto :end_pause
)

echo.
echo [coqui-tts] tts-server XTTS v2: http://localhost:5002
echo Para fine-tuning: run-docker.bat finetune   ^|   docker-finetune.bat
echo.
docker compose up xtts-server
set "EC=!ERRORLEVEL!"
goto :end_pause

:end_pause
if not defined EC set "EC=0"
echo.
if not "!EC!"=="0" (
  echo [coqui-tts] Codigo de saida: !EC! ^(verifique mensagens acima se houve erro^)
) else (
  echo [coqui-tts] Codigo de saida: 0 ^(ok^).
)
echo.
echo Pressione qualquer tecla para fechar esta janela...
pause
exit /b !EC!
