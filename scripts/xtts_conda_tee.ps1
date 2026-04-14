# Roda Python do ambiente Conda com <script> <args...> (sem "conda run" na linha do treino).
# Grava stdout/stderr em LogMain (UTF-8 append) e mostra na tela (Write-Host).
# XTTS_LOG_FILE_ONLY=1 = so arquivo, sem eco no console.
param(
    [Parameter(Mandatory)][string]$LogMain,
    [Parameter()][string]$LogCopy,
    [Parameter(Mandatory)][string]$CondaEnv,
    [Parameter(Mandatory)][string]$PythonScript,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$PythonArgs
)
if ([string]::IsNullOrWhiteSpace($LogCopy)) { $LogCopy = $LogMain }
$ErrorActionPreference = "Continue"
try {
    $utf8 = [System.Text.UTF8Encoding]::new($false)
    [Console]::OutputEncoding = $utf8
    [Console]::InputEncoding = $utf8
} catch { }
try { [Console]::ResetColor() } catch { }

function Write-ConHost([string]$Text) {
    if ($env:XTTS_LOG_FILE_ONLY -eq "1") { return }
    try {
        Write-Host $Text -ForegroundColor White
    } catch {
        try { [Console]::WriteLine($Text) } catch { }
    }
}

$conda = (Get-Command conda -ErrorAction SilentlyContinue).Source
if (-not $conda) { $conda = "conda" }
$condaBase = $null
try {
    $condaBase = (& conda info --base 2>$null | Select-Object -First 1)
    if ($condaBase) {
        $condaBase = $condaBase.Trim()
        $cx = Join-Path $condaBase "Scripts\conda.exe"
        if (Test-Path -LiteralPath $cx) { $conda = $cx }
    }
} catch {}

$pythonExe = $null
if ($condaBase) {
    if ($CondaEnv -eq 'base') {
        $cand = Join-Path $condaBase "python.exe"
    } else {
        $cand = Join-Path (Join-Path (Join-Path $condaBase "envs") $CondaEnv) "python.exe"
    }
    if (Test-Path -LiteralPath $cand) { $pythonExe = $cand }
}
if (-not $pythonExe) {
    try {
        $out = & $conda run -n $CondaEnv python -c "import sys; print(sys.executable)" 2>&1
        if ($LASTEXITCODE -eq 0 -and $out) {
            $line = ($out | Where-Object { $null -ne $_ -and "$_".Trim() } | Select-Object -First 1)
            if ($line) { $pythonExe = $line.ToString().Trim() }
        }
    } catch {}
}

function Write-TeeLog([string]$Msg) {
    try { Add-Content -LiteralPath $LogMain -Value $Msg -Encoding utf8 } catch {}
    if ($LogCopy -ne $LogMain) { try { Add-Content -LiteralPath $LogCopy -Value $Msg -Encoding utf8 } catch {} }
    Write-ConHost $Msg
}

if (-not $pythonExe -or -not (Test-Path -LiteralPath $pythonExe)) {
    Write-TeeLog "ERRO: nao foi possivel achar python.exe do ambiente conda '$CondaEnv'."
    exit 1
}

try {
    $logDir = Split-Path -Parent -Path $LogMain
    if ($logDir -and -not (Test-Path -LiteralPath $logDir)) {
        New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    }
    if ($LogCopy -ne $LogMain) {
        $logDir2 = Split-Path -Parent -Path $LogCopy
        if ($logDir2 -and -not (Test-Path -LiteralPath $logDir2)) {
            New-Item -ItemType Directory -Path $logDir2 -Force | Out-Null
        }
    }
} catch { }

$cleanArgs = @($PythonArgs | Where-Object { $null -ne $_ -and "$_".Trim() -ne "" })
$allArgs = @($PythonScript) + $cleanArgs

$wd = $null
try {
    $wd = (Get-Item -LiteralPath $PythonScript).Directory.Parent.FullName
} catch { }

try {
    if ($wd) { Set-Location -LiteralPath $wd }
} catch {
    Write-TeeLog "AVISO: nao foi possivel definir WorkingDirectory: $wd"
}

$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:NO_COLOR = "1"
$env:PYTHONCOLORS = "0"

# Invocacao directa (argv array). Evita list2cmdline + Process.Arguments + leitura assincrona,
# que em alguns ambientes devolvia ExitCode=2 sem drenar stdout/stderr ate ao log.
$script:LM = $LogMain
$script:LC = $LogCopy
& $pythonExe @allArgs 2>&1 | ForEach-Object {
    $line = "$_"
    try {
        Add-Content -LiteralPath $script:LM -Value $line -Encoding utf8
        if ($script:LC -ne $script:LM) { try { Add-Content -LiteralPath $script:LC -Value $line -Encoding utf8 } catch {} }
    } catch {}
    Write-ConHost $line
}

$code = 0
if ($null -ne $LASTEXITCODE) { $code = [int]$LASTEXITCODE }
exit $code
