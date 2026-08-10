[CmdletBinding()]
param(
    [switch]$CpuOnly,
    [ValidateRange(30, 300)]
    [int]$TimeoutSeconds = 120
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$composeFiles = @(
    "-f", (Join-Path $repoRoot "compose.yaml")
)
if (-not $CpuOnly) {
    $composeFiles += @("-f", (Join-Path $repoRoot "compose.gpu.yaml"))
}
$composeFiles += @("-f", (Join-Path $repoRoot "compose.cloudflare.yaml"))

Push-Location $repoRoot
try {
    & docker compose @composeFiles up -d --build
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Compose could not start the highlight service."
    }

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $tunnelUrl = $null
    while ((Get-Date) -lt $deadline -and -not $tunnelUrl) {
        $logs = (& docker compose @composeFiles logs --no-color cloudflared 2>&1 | Out-String)
        $matches = [regex]::Matches(
            $logs,
            "https://[a-z0-9-]+\.trycloudflare\.com",
            [System.Text.RegularExpressions.RegexOptions]::IgnoreCase
        )
        if ($matches.Count -gt 0) {
            $tunnelUrl = $matches[$matches.Count - 1].Value.TrimEnd("/")
            break
        }
        Start-Sleep -Seconds 2
    }

    if (-not $tunnelUrl) {
        & docker compose @composeFiles logs --no-color --tail 80 cloudflared
        throw "Cloudflare did not return a Quick Tunnel URL within $TimeoutSeconds seconds."
    }

    $healthy = $false
    while ((Get-Date) -lt $deadline -and -not $healthy) {
        try {
            $health = Invoke-RestMethod -Uri "$tunnelUrl/api/health" -TimeoutSec 10
            $healthy = $health.status -eq "ok"
        }
        catch {
            Start-Sleep -Seconds 2
        }
    }
    if (-not $healthy) {
        throw "The tunnel URL was created, but its health check did not become ready."
    }

    $tokenPath = Join-Path $repoRoot "data\.upload-token"
    if (-not (Test-Path -LiteralPath $tokenPath)) {
        throw "Upload token was not created at $tokenPath."
    }
    $token = (Get-Content -Raw -LiteralPath $tokenPath).Trim()
    if (-not $token) {
        throw "Upload token is empty."
    }

    $phoneUrl = "$tunnelUrl/#token=$([uri]::EscapeDataString($token))"
    $urlPath = Join-Path $repoRoot "data\remote-access-url.txt"
    Set-Content -LiteralPath $urlPath -Value $phoneUrl -Encoding UTF8

    Write-Host ""
    Write-Host "Cloudflare Quick Tunnel is ready." -ForegroundColor Green
    Write-Host "Open this token-protected link on your phone:"
    Write-Output $phoneUrl
    Write-Host ""
    Write-Warning "Anyone with the full link can use this service. Do not share it."
    Write-Warning "Keep Docker Desktop and this computer running. The URL changes if cloudflared is recreated."
    Write-Host "The latest link is also saved to $urlPath"
}
finally {
    Pop-Location
}
