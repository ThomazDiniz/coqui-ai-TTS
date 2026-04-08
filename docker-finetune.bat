@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

rem Fine-tuning em primeiro plano (fecha o terminal = pode parar o container).
rem   docker-finetune.bat           ^|  docker-finetune.bat nobuild
rem
rem Em segundo plano (recomendado para treinos longos):
rem   docker-finetune-detached.bat  ^|  docker-finetune-detached.bat nobuild
rem Ver logs depois:
rem   docker-finetune-logs.bat

call "%~dp0run-docker.bat" finetune %*
exit /b !ERRORLEVEL!
