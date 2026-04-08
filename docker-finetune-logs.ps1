# Segue logs do fine-tune sem terminar sozinho: ficheiro ui_console.log (volume) ou docker compose logs -f.
# Uso: .\docker-finetune-logs.ps1 [-Mode auto|file|docker]
param(
    [ValidateSet("auto", "file", "docker")]
    [string]$Mode = "auto",
    [string]$LogFile = "",
    [string]$ComposeProfile = "finetune",
    [string]$Service = "xtts-finetune"
)

$ErrorActionPreference = "Continue"
if (-not $LogFile) {
    $LogFile = Join-Path $PSScriptRoot "data\xtts-work\ui_console.log"
}

Set-Location $PSScriptRoot

function Wait-PathExists {
    param([string]$Path)
    while (-not (Test-Path -LiteralPath $Path)) {
        Write-Host "[coqui-tts] Aguardando ficheiro: $Path"
        Start-Sleep -Seconds 2
    }
}

function Follow-FileForever {
    param([string]$Path)
    while ($true) {
        Wait-PathExists -Path $Path
        Write-Host "[coqui-tts] A seguir ficheiro (Ctrl+C para sair): $Path"
        try {
            Get-Content -LiteralPath $Path -Wait -Tail 200 -Encoding utf8
        } catch {
            Write-Host "[coqui-tts] Erro ao ler: $_"
        }
        Write-Host "[coqui-tts] Stream do ficheiro parou; a reconectar em 2s..."
        Start-Sleep -Seconds 2
    }
}

function Wait-ContainerRunning {
    while ($true) {
        docker compose --profile $ComposeProfile exec -T $Service true 2>$null
        if ($LASTEXITCODE -eq 0) {
            return
        }
        Write-Host "[coqui-tts] Aguardando container '$Service' (suba com docker-finetune-detached.bat)..."
        Start-Sleep -Seconds 2
    }
}

function Follow-DockerLogsForever {
    while ($true) {
        Wait-ContainerRunning
        Write-Host "[coqui-tts] docker compose logs -f $Service (Ctrl+C para sair; se cair, reconecta)"
        docker compose --profile $ComposeProfile logs -f --tail 300 $Service
        Write-Host "[coqui-tts] Stream Docker terminou; a reconectar em 3s..."
        Start-Sleep -Seconds 3
    }
}

switch ($Mode) {
    "file" {
        Follow-FileForever -Path $LogFile
    }
    "docker" {
        Follow-DockerLogsForever
    }
    default {
        if (-not (Test-Path -LiteralPath $LogFile)) {
            Write-Host "[coqui-tts] A aguardar criacao de: $LogFile  (ou use docker-finetune-logs.bat docker)"
        }
        Follow-FileForever -Path $LogFile
    }
}
