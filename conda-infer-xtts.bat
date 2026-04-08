@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

rem Inferencia XTTS fine-tuned (sem Gradio), usando Conda env coqui-xtts.
rem
rem Uso:
rem   conda-infer-xtts.bat
rem   conda-infer-xtts.bat <env_name>
rem
rem Saida: .\data\xtts-work\xtts_infer_YYYYMMDD-HHMMSS.wav

set "ENV_NAME=%~1"
if "%ENV_NAME%"=="" set "ENV_NAME=coqui-xtts"

set "TRAIN_CSV=/data/xtts_ft/dataset/metadata_train.csv"
set "EVAL_CSV=/data/xtts_ft/dataset/metadata_eval.csv"

set "CKPT=E:\git\coqui-ai-TTS\data\xtts-work\run\training\GPT_XTTS_FT-April-08-2026_08+49AM-a4b54402\best_model.pth"
set "CFG=E:\git\coqui-ai-TTS\data\xtts-work\run\training\GPT_XTTS_FT-April-08-2026_08+49AM-a4b54402\config.json"
set "VOCAB=E:\git\coqui-ai-TTS\data\xtts-work\run\training\XTTS_v2.0_original_model_files\vocab.json"

set "SPEAKER_WAV=E:\git\coqui-ai-TTS\PB_0003.wav"
set "TEXT=A cidade de campina grande realmente é grande"
set "LANG=pt"

echo.
echo [coqui-tts] (info) train_csv: %TRAIN_CSV%
echo [coqui-tts] (info) eval_csv : %EVAL_CSV%
echo.
echo [coqui-tts] env   : %ENV_NAME%
echo [coqui-tts] ckpt  : %CKPT%
echo [coqui-tts] cfg   : %CFG%
echo [coqui-tts] vocab : %VOCAB%
echo [coqui-tts] ref   : %SPEAKER_WAV%
echo [coqui-tts] lang  : %LANG%
echo [coqui-tts] text  : %TEXT%
echo.

conda run -n "%ENV_NAME%" --no-capture-output python "%~dp0scripts\xtts_infer_cli.py" ^
  --checkpoint "%CKPT%" ^
  --config "%CFG%" ^
  --vocab "%VOCAB%" ^
  --speaker_wav "%SPEAKER_WAV%" ^
  --language "%LANG%" ^
  --text "%TEXT%"

set "EC=%ERRORLEVEL%"
echo.
if not "%EC%"=="0" (
  echo [coqui-tts] ERRO: codigo %EC%.
  echo Pressione qualquer tecla para fechar...
  pause >nul
)
exit /b %EC%

