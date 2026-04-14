# Acompanha um arquivo de texto em tempo real (ultimas N linhas e depois linhas novas).
# Uso: powershell -File xtts_log_tail.ps1 -Path "E:\...\log.log"
param(
    [Parameter(Mandatory)][string]$Path
)
$ErrorActionPreference = "Continue"
$p = $Path.Trim()
if ([string]::IsNullOrWhiteSpace($p)) {
    [Console]::Error.WriteLine("ERRO: caminho vazio.")
    exit 1
}
while (-not (Test-Path -LiteralPath $p)) {
    Write-Host "Aguardando o arquivo: $p"
    Start-Sleep -Milliseconds 500
}
Write-Host "Exibindo em tempo real (Ctrl+C para sair): $p"
Write-Host ("=" * 72)
Get-Content -LiteralPath $p -Wait -Tail 120 -Encoding utf8
