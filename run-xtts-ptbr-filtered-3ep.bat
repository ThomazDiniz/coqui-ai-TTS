@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Launcher para fine-tune PT-BR usando data\filtered com 3 epocas.
rem Chama o runner principal mantendo o mesmo pipeline (conda/pip/validate/prep/train).

cd /d "%~dp0"

set "DATA_ROOT=%~dp0data\filtered"
if not exist "%DATA_ROOT%\metadata.csv" (
  echo [ERRO] metadata.csv nao encontrado em %DATA_ROOT%
  pause
  exit /b 1
)
if not exist "%DATA_ROOT%\wavs" (
  echo [ERRO] pasta wavs nao encontrada em %DATA_ROOT%
  pause
  exit /b 1
)

if "%ENV_NAME%"=="" set "ENV_NAME=coqui-xtts"
set "EPOCHS=3"
if "%BATCH_SIZE%"=="" set "BATCH_SIZE=2"
if "%GRAD_ACUMM%"=="" set "GRAD_ACUMM=2"
if "%CHECKPOINT_EVERY_EPOCHS%"=="" set "CHECKPOINT_EVERY_EPOCHS=1"
if "%SAVE_N_CHECKPOINTS%"=="" set "SAVE_N_CHECKPOINTS=20"
if "%INFERENCE_SAMPLES%"=="" set "INFERENCE_SAMPLES=5"
if "%AUDIO_SUBDIR%"=="" set "AUDIO_SUBDIR=wavs"
if "%SPEAKER_NAME%"=="" set "SPEAKER_NAME=ptbr_filtered_ft"

echo.
echo ================================================================
echo   XTTS PT-BR (filtered) - Fine-tune rapido 3 epocas
echo ================================================================
echo   DATA_ROOT: %DATA_ROOT%
echo   ENV_NAME:  %ENV_NAME%
echo   EPOCHS:    %EPOCHS%
echo ================================================================
echo.

call "%~dp0run-xtts-ptbr-experiment.bat" "%DATA_ROOT%" "%ENV_NAME%"
exit /b %ERRORLEVEL%

