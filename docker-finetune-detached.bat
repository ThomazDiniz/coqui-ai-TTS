@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

rem Sobe o Gradio de fine-tuning XTTS em segundo plano (nao trava ao fechar o terminal).
rem Uso:
rem   docker-finetune-detached.bat           ^> build + up -d
rem   docker-finetune-detached.bat nobuild   ^> so up -d (sem rebuild)

set "SKIP_BUILD=0"
if /i "%~1"=="nobuild" set "SKIP_BUILD=1"

if "!SKIP_BUILD!"=="0" (
  echo.
  echo [coqui-tts] Build da imagem coqui-tts:local...
  docker compose build
  if errorlevel 1 (
    echo.
    echo [coqui-tts] Build falhou.
    exit /b 1
  )
) else (
  echo.
  echo [coqui-tts] Pulando build ^(nobuild^).
)

echo.
echo [coqui-tts] Subindo xtts-finetune em background ^(-d^)...
docker compose --profile finetune up -d xtts-finetune
set "EC=!ERRORLEVEL!"
if not "!EC!"=="0" (
  echo [coqui-tts] Falha ao subir o servico. Codigo: !EC!
  exit /b !EC!
)

echo.
echo [coqui-tts] OK — container em execucao.
echo     Gradio: http://localhost:5003
echo     Dados:  .\data\xtts-work
echo.
echo     Ver logs ao vivo: docker-finetune-logs.bat  ^(ou data\xtts-work\ui_console.log^)
echo     Parar:          docker compose --profile finetune stop xtts-finetune
echo.
exit /b 0
