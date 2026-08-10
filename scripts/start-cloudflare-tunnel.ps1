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
if ($CpuOnly) {
    $composeFiles += @("-f", (Join-Path $repoRoot "compose.cpu.yaml"))
}
$composeFiles += @("-f", (Join-Path $repoRoot "compose.cloudflare.yaml"))

function Get-LatestTunnelUrl {
    $logs = (& docker compose @composeFiles logs --no-color cloudflared 2>&1 | Out-String)
    $matches = [regex]::Matches(
        $logs,
        "https://[a-z0-9-]+\.trycloudflare\.com",
        [System.Text.RegularExpressions.RegexOptions]::IgnoreCase
    )
    if ($matches.Count -eq 0) {
        return $null
    }
    return $matches[$matches.Count - 1].Value.TrimEnd("/")
}

function Get-CloudflaredHealth {
    $containerId = (& docker compose @composeFiles ps -q cloudflared 2>$null | Out-String).Trim()
    if (-not $containerId) {
        return "missing"
    }

    $health = (& docker inspect `
        --format "{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}" `
        $containerId 2>$null | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $health) {
        return "unknown"
    }
    return $health
}

Push-Location $repoRoot
try {
    & docker compose @composeFiles up -d --build
    if ($LASTEXITCODE -ne 0) {
        if (-not $CpuOnly) {
            throw (
                "Docker Compose could not start the GPU highlight service. " +
                "Confirm NVIDIA Container Toolkit is available, or run this script with -CpuOnly."
            )
        }
        throw "Docker Compose could not start the highlight service."
    }

    $previousTunnelUrl = Get-LatestTunnelUrl
    $cloudflaredHealth = Get-CloudflaredHealth
    $requireFreshUrl = $false
    if ($cloudflaredHealth -eq "unhealthy") {
        Write-Warning "The existing Quick Tunnel is unhealthy. Creating a fresh tunnel URL."
        & docker compose @composeFiles restart cloudflared
        if ($LASTEXITCODE -ne 0) {
            throw "Cloudflare could not be restarted."
        }
        $requireFreshUrl = [bool]$previousTunnelUrl
    }

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $tunnelUrl = $null
    while ((Get-Date) -lt $deadline -and -not $tunnelUrl) {
        $candidateUrl = Get-LatestTunnelUrl
        if ($candidateUrl -and (-not $requireFreshUrl -or $candidateUrl -ne $previousTunnelUrl)) {
            $tunnelUrl = $candidateUrl
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
        & docker compose @composeFiles logs --no-color --tail 80 cloudflared
        throw (
            "The tunnel URL was created, but it could not connect to Cloudflare. " +
            "Check whether this network allows outbound TCP or UDP port 7844, " +
            "or switch networks and run this script again."
        )
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
