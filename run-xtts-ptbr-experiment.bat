@echo off
setlocal EnableExtensions EnableDelayedExpansion
rem ============================================================================
rem  Fine-tune XTTS para portugues brasileiro (PT-BR)
rem  Log por execucao: xtts-ptbr-AAAAMMDD-HHMMSS.log (evita "arquivo em uso")
rem  Indice: data\xtts-experiments\logs\xtts-ptbr-ULTIMO.txt = caminho do ultimo log
rem ============================================================================

chcp 65001 >nul 2>&1
color 07
set "NO_COLOR=1"
set "PYTHONCOLORS=0"
set "PYTHONUNBUFFERED=1"
set "PYTHONIOENCODING=utf-8"

cd /d "%~dp0"
title XTTS PT-BR - experimento

if not exist "%~dp0scripts\xtts_finetune_paraiba_experiment.py" (
  echo [ERRO] Execute na raiz do repositorio coqui-ai-TTS.
  pause
  exit /b 1
)
if not exist "%~dp0scripts\xtts_conda_tee.ps1" (
  echo [ERRO] Falta scripts\xtts_conda_tee.ps1
  pause
  exit /b 1
)

set "LOG_DIR=%~dp0data\xtts-experiments\logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%" >nul 2>nul
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"') do set "TS=%%i"
rem Um arquivo NOVO por run — nao use append no mesmo ficheiro com Notepad/tail aberto
set "LOG_FILE=%LOG_DIR%\xtts-ptbr-%TS%.log"
set "EXP_DIR=%~dp0data\xtts-experiments\ptbr-%TS%"
if not exist "%EXP_DIR%" mkdir "%EXP_DIR%" >nul 2>nul

echo %LOG_FILE%> "%LOG_DIR%\xtts-ptbr-ULTIMO.txt"
type nul > "%LOG_FILE%"

set "STAGE_FAIL="
set "ENV_NAME=%~2"
if "!ENV_NAME!"=="" set "ENV_NAME=coqui-xtts"

if "%~1"=="" (
  set "DATA_ROOT=%~dp0data\all_char"
) else (
  set "DATA_ROOT=%~f1"
)

if not defined AUDIO_SUBDIR set "AUDIO_SUBDIR=wavs"
if "%SKIP_INSTALL%"=="" set "SKIP_INSTALL=0"
if "%EPOCHS%"=="" set "EPOCHS=10"
if "%BATCH_SIZE%"=="" set "BATCH_SIZE=2"
if "%GRAD_ACUMM%"=="" set "GRAD_ACUMM=2"
if "%INFERENCE_SAMPLES%"=="" set "INFERENCE_SAMPLES=5"
if "%CHECKPOINT_EVERY_EPOCHS%"=="" set "CHECKPOINT_EVERY_EPOCHS=2"
if "%SAVE_N_CHECKPOINTS%"=="" set "SAVE_N_CHECKPOINTS=10"
if "%SPEAKER_NAME%"=="" set "SPEAKER_NAME=paraiba_ptbr"
set "PREP_LANGUAGE=pt"
set "TRAIN_LANGUAGE=pt"

echo.
echo ================================================================================
echo   XTTS fine-tune - Portugues do Brasil (PT-BR)
echo ================================================================================
echo   Log desta execucao:  %LOG_FILE%
echo   Indice ultimo log:   %LOG_DIR%\xtts-ptbr-ULTIMO.txt
echo   Experimento:         %EXP_DIR%
echo   DATA_ROOT:           %DATA_ROOT%
echo   Audios:              %DATA_ROOT%\%AUDIO_SUBDIR%\
echo   Conda env:           %ENV_NAME%
echo   Dica: feche Notepad/VS Code no log antigo se usava o ficheiro unico antigo.
echo ================================================================================
echo.

call :log "=============================================================================="
call :log "[RUN %TS%] XTTS PT-BR - inicio"
call :log "PWD=%CD%"
call :log "LOG_FILE=%LOG_FILE%"
call :log "EXP_DIR=%EXP_DIR%"
call :log "DATA_ROOT=%DATA_ROOT%  AUDIO_SUBDIR=%AUDIO_SUBDIR%"
call :log "ENV=%ENV_NAME%  EPOCHS=%EPOCHS%  BATCH=%BATCH_SIZE%  GRAD_ACUMM=%GRAD_ACUMM%"
call :log "INFERENCE_SAMPLES=%INFERENCE_SAMPLES%  CHECKPOINT_EVERY=%CHECKPOINT_EVERY_EPOCHS%  SAVE_N=%SAVE_N_CHECKPOINTS%"
call :log "SPEAKER_NAME=%SPEAKER_NAME%"
call :log "PREP_LANGUAGE=%PREP_LANGUAGE% TRAIN_LANGUAGE=%TRAIN_LANGUAGE% (PT-BR pelos dados)"
call :log "SKIP_INSTALL=%SKIP_INSTALL%"
call :log "=============================================================================="

rem ---------------------------------------------------------------------------
rem FASE 1: Conda
rem ---------------------------------------------------------------------------
call :phase 1 "Conda e ambiente %ENV_NAME%"
call :log "[FASE 1/6] Testando: conda --version"
rem OBRIGATORIO "call" — conda e um .bat; sem call o script atual termina e nao volta
call conda --version >"%TEMP%\xtts_cv_%TS%.txt" 2>&1
set "CERR=!ERRORLEVEL!"
call :log "Exit code conda --version: !CERR!"
type "%TEMP%\xtts_cv_%TS%.txt"
call :log_tail "%TEMP%\xtts_cv_%TS%.txt"
if not "!CERR!"=="0" (
  call :log "ERRO: conda --version falhou."
  set "STAGE_FAIL=1"
  goto FINAL
)

call :log "Rodando: conda info --base"
for /f "delims=" %%B in ('conda info --base 2^>^&1') do set "CONDA_BASE=%%B"
call :log "CONDA_BASE=!CONDA_BASE!"

call :log "Testando: conda run -n %ENV_NAME% python --version"
call conda run -n "%ENV_NAME%" --no-capture-output python --version >"%TEMP%\xtts_py_%TS%.txt" 2>&1
set "PVERR=!ERRORLEVEL!"
call :log "Exit code python --version: !PVERR!"
type "%TEMP%\xtts_py_%TS%.txt"
call :log_tail "%TEMP%\xtts_py_%TS%.txt"
if not "!PVERR!"=="0" (
  call :log "ERRO: ambiente conda invalido ou inexistente: %ENV_NAME%"
  set "STAGE_FAIL=1"
  goto FINAL
)
call :log "[FASE 1/6] OK"

rem ---------------------------------------------------------------------------
rem FASE 2: pip
rem ---------------------------------------------------------------------------
call :phase 2 "pip install -e .[cuda,xtts_ft]"
if "%SKIP_INSTALL%"=="1" (
  call :log "SKIP_INSTALL=1 - pulando pip."
  goto after_pip
)
set "CONDA_ENV_PY=!CONDA_BASE!\envs\!ENV_NAME!\python.exe"
call :log "Python do env: !CONDA_ENV_PY!"
if not exist "!CONDA_ENV_PY!" (
  call :log "ERRO: python do env nao encontrado."
  set "STAGE_FAIL=1"
  goto FINAL
)
call :log "Diretorio repo (pushd): %~dp0"
pushd "%~dp0"
call :log "Executando: python -m pip install -e .[cuda,xtts_ft]"
"!CONDA_ENV_PY!" -m pip install -e ".[cuda,xtts_ft]" >"%TEMP%\xtts_pip_%TS%.txt" 2>&1
set "PIPERR=!ERRORLEVEL!"
call :log "Exit code pip: !PIPERR!"
type "%TEMP%\xtts_pip_%TS%.txt"
call :log_tail "%TEMP%\xtts_pip_%TS%.txt"
popd
call :log "popd concluido"
if not "!PIPERR!"=="0" (
  call :log "ERRO: pip install falhou."
  set "STAGE_FAIL=1"
  goto FINAL
)
call :log "[FASE 2/6] OK"
:after_pip

rem ---------------------------------------------------------------------------
rem FASE 3: Imports
rem ---------------------------------------------------------------------------
call :phase 3 "Imports TTS / torch / faster_whisper"
call :log "Executando checagem de imports..."
call conda run -n "%ENV_NAME%" --no-capture-output python -c "import TTS; import torch; import faster_whisper; print('TTS', getattr(TTS,'__version__', '?')); print('torch', torch.__version__); print('cuda', torch.cuda.is_available())" >"%TEMP%\xtts_imp_%TS%.txt" 2>&1
set "IMPERR=!ERRORLEVEL!"
call :log "Exit code imports: !IMPERR!"
type "%TEMP%\xtts_imp_%TS%.txt"
call :log_tail "%TEMP%\xtts_imp_%TS%.txt"
if not "!IMPERR!"=="0" (
  call :log "ERRO: imports falharam."
  set "STAGE_FAIL=1"
  goto FINAL
)
call :log "[FASE 3/6] OK"

rem ---------------------------------------------------------------------------
rem FASE 4: Validacao dados
rem ---------------------------------------------------------------------------
call :phase 4 "validate_ptbr_dataset.py"
call :log "Script: %~dp0scripts\validate_ptbr_dataset.py"
call :log "Argumento: --data-root %DATA_ROOT%"
call conda run -n "%ENV_NAME%" --no-capture-output python "%~dp0scripts\validate_ptbr_dataset.py" --data-root "%DATA_ROOT%" >"%TEMP%\xtts_val_%TS%.txt" 2>&1
set "VALERR=!ERRORLEVEL!"
call :log "Exit code validacao: !VALERR!"
type "%TEMP%\xtts_val_%TS%.txt"
call :log_tail "%TEMP%\xtts_val_%TS%.txt"
if not "!VALERR!"=="0" (
  call :log "ERRO: validacao dos dados falhou."
  set "STAGE_FAIL=1"
  goto FINAL
)
call :log "[FASE 4/6] OK"

rem ---------------------------------------------------------------------------
rem FASE 5: Prep
rem ---------------------------------------------------------------------------
call :phase 5 "prep (--stage prep)"
call :log "xtts_conda_tee.ps1 + xtts_finetune_paraiba_experiment.py"
call :log "Comando resumido: prep --data-root ... --experiment-dir %EXP_DIR%"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\xtts_conda_tee.ps1" ^
  -LogMain "%LOG_FILE%" ^
  -CondaEnv "%ENV_NAME%" ^
  -PythonScript "%~dp0scripts\xtts_finetune_paraiba_experiment.py" ^
  --stage prep ^
  --data-root "%DATA_ROOT%" ^
  --audio-subdir "%AUDIO_SUBDIR%" ^
  --prep-language "%PREP_LANGUAGE%" ^
  --train-language "%TRAIN_LANGUAGE%" ^
  --speaker-name "%SPEAKER_NAME%" ^
  --experiment-dir "%EXP_DIR%" ^
  --epochs %EPOCHS% --batch-size %BATCH_SIZE% --grad-acumm %GRAD_ACUMM% ^
  --checkpoint-every-epochs %CHECKPOINT_EVERY_EPOCHS% --save-n-checkpoints %SAVE_N_CHECKPOINTS%
set "PREPERR=!ERRORLEVEL!"
call :log "Exit code PowerShell/prep: !PREPERR!"
if not "!PREPERR!"=="0" call :log "AVISO: codigo de saida prep nao-zero."
if not exist "%EXP_DIR%\artifacts\.stage_prep_ok" (
  call :log "ERRO: prep incompleto (sem .stage_prep_ok)."
  set "STAGE_FAIL=1"
  goto FINAL
)
call :log "[FASE 5/6] OK - marcador .stage_prep_ok presente."

rem ---------------------------------------------------------------------------
rem FASE 6: Train
rem ---------------------------------------------------------------------------
call :phase 6 "train (--stage train)"
call :log "experiment-dir: %EXP_DIR%"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\xtts_conda_tee.ps1" ^
  -LogMain "%LOG_FILE%" ^
  -CondaEnv "%ENV_NAME%" ^
  -PythonScript "%~dp0scripts\xtts_finetune_paraiba_experiment.py" ^
  --stage train ^
  --train-language "%TRAIN_LANGUAGE%" ^
  --experiment-dir "%EXP_DIR%" ^
  --epochs %EPOCHS% --batch-size %BATCH_SIZE% --grad-acumm %GRAD_ACUMM% ^
  --inference-samples %INFERENCE_SAMPLES% ^
  --checkpoint-every-epochs %CHECKPOINT_EVERY_EPOCHS% --save-n-checkpoints %SAVE_N_CHECKPOINTS%
set "TRAINERR=!ERRORLEVEL!"
call :log "Exit code treino: !TRAINERR!"
if not "!TRAINERR!"=="0" call :log "AVISO: codigo de saida treino nao-zero."
if not exist "%EXP_DIR%\artifacts\.stage_train_ok" (
  call :log "ERRO: treino incompleto (sem .stage_train_ok)."
  set "STAGE_FAIL=1"
  goto FINAL
)
call :log "SUCESSO. Artefatos: %EXP_DIR%\artifacts\"
call :log "[FASE 6/6] OK"

:FINAL
echo.
call :log "=============================================================================="
if defined STAGE_FAIL (
  call :log "FIM COM ERRO (STAGE_FAIL definido)"
  echo.
  echo   ENCERRADO COM ERRO
  echo   Log desta execucao: %LOG_FILE%
  echo.
  powershell -NoProfile -Command "if (Test-Path '%LOG_FILE%') { Get-Content -LiteralPath '%LOG_FILE%' -Tail 100 -Encoding utf8 }"
) else (
  call :log "FIM OK"
  echo.
  echo   Concluido sem erro aparente no batch.
  echo   Log: %LOG_FILE%
  echo   Artefatos: %EXP_DIR%\artifacts\
)
echo.
echo   Ultimo log (atalho): %LOG_DIR%\xtts-ptbr-ULTIMO.txt
echo ================================================================================
echo.
echo Pressione qualquer tecla para fechar...
pause >nul
if defined STAGE_FAIL exit /b 1
exit /b 0

rem ---------- Subrotinas: log com hora (arquivo NOVO por execucao) ----------
:phase
echo.
echo ------------------------------------------------------------------------------
echo   [FASE %~1/6] %~2
echo ------------------------------------------------------------------------------
call :log "---------- [FASE %~1/6] %~2 ----------"
goto :eof

:log
set "_LINE=%~1"
for /f "tokens=*" %%t in ('powershell -NoProfile -Command "Get-Date -Format \"yyyy-MM-dd HH:mm:ss\""') do set "_TS=%%t"
echo [!_TS!] !_LINE!
echo [!_TS!] !_LINE! >>"%LOG_FILE%"
goto :eof

:log_tail
for /f "tokens=*" %%t in ('powershell -NoProfile -Command "Get-Date -Format \"yyyy-MM-dd HH:mm:ss\""') do set "_TS=%%t"
echo [!_TS!] --- copia saida temporaria para o log ---
echo [!_TS!] --- copia saida temporaria para o log --- >>"%LOG_FILE%"
type "%~1" >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"
goto :eof
