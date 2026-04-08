@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

rem Gera/regenara o training_loss.png a partir dos eventos do TensorBoard.
rem
rem Uso:
rem   conda-plot-loss.bat "<run_dir>"
rem   conda-plot-loss.bat "<run_dir>" "<env_name>"
rem
rem Exemplo:
rem   conda-plot-loss.bat ".\data\xtts-work\run\training\GPT_XTTS_FT-..."

set "RUN_DIR=%~1"
if "%RUN_DIR%"=="" (
  echo [coqui-tts] ERRO: informe a pasta do run.
  echo   Ex: conda-plot-loss.bat ".\data\xtts-work\run\training\GPT_XTTS_FT-..."
  exit /b 1
)

set "ENV_NAME=%~2"
if "%ENV_NAME%"=="" set "ENV_NAME=coqui-xtts"

echo.
echo [coqui-tts] Conda env: %ENV_NAME%
echo [coqui-tts] Run dir:   %RUN_DIR%
echo.

conda run -n "%ENV_NAME%" --no-capture-output python "%~dp0scripts\xtts_plot_loss.py" --run_dir "%RUN_DIR%"
set "EC=%ERRORLEVEL%"
echo.
if not "%EC%"=="0" (
  echo [coqui-tts] Falhou com codigo %EC%.
  exit /b %EC%
)
echo [coqui-tts] OK.
exit /b 0

