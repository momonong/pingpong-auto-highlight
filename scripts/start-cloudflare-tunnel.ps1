[CmdletBinding()]
param(
    [switch]$CpuOnly,
    [switch]$UsePublishedImage,
    [ValidateRange(30, 300)]
    [int]$TimeoutSeconds = 120
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "deployment-common.ps1")
Assert-HighlightCraftComposeVersion
$dataRoot = Get-HighlightCraftDataRoot -RepositoryRoot $repoRoot
$composeFiles = @(
    "-f", (Join-Path $repoRoot "compose.yaml")
)
if ($UsePublishedImage) {
    $composeFiles += @("-f", (Join-Path $repoRoot "compose.release.yaml"))
}
if ($CpuOnly) {
    $composeFiles += @("-f", (Join-Path $repoRoot "compose.cpu.yaml"))
}
$composeFiles += @("-f", (Join-Path $repoRoot "compose.cloudflare.yaml"))

function Get-LatestTunnelUrl {
    $logs = (& docker compose @composeFiles logs --no-color cloudflared 2>&1 | Out-String)
    $matches = [regex]::Matches(
        $logs,
        "\|\s*(https://[a-z0-9-]+\.trycloudflare\.com)\s*\|",
        [System.Text.RegularExpressions.RegexOptions]::IgnoreCase
    )
    if ($matches.Count -eq 0) {
        return $null
    }
    return $matches[$matches.Count - 1].Groups[1].Value.TrimEnd("/")
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
    Assert-HighlightCraftNoActiveWork -RepositoryRoot $repoRoot -DataRoot $dataRoot
    $previousTunnelUrl = Get-LatestTunnelUrl

    if ($UsePublishedImage) {
        & docker compose @composeFiles pull pingpong-highlight cloudflared
        if ($LASTEXITCODE -ne 0) {
            throw "Docker Compose could not pull the published HighlightCraft image from Docker Hub."
        }
    }
    else {
        & docker compose @composeFiles build pingpong-highlight
        if ($LASTEXITCODE -ne 0) {
            throw "Docker Compose could not build the HighlightCraft image."
        }
        & docker compose @composeFiles pull --policy missing cloudflared
        if ($LASTEXITCODE -ne 0) {
            throw "Docker Compose could not prepare the cloudflared image."
        }
    }

    # Pulling or building can take long enough for new work to arrive. Check
    # again immediately before `up`, the operation that may recreate the app.
    Assert-HighlightCraftNoActiveWork -RepositoryRoot $repoRoot -DataRoot $dataRoot
    & docker compose @composeFiles up -d --no-build pingpong-highlight cloudflared
    if ($LASTEXITCODE -ne 0) {
        if (-not $CpuOnly) {
            throw (
                "Docker Compose could not start the GPU highlight service. " +
                "Confirm NVIDIA Container Toolkit is available, or run this script with -CpuOnly."
            )
        }
        throw "Docker Compose could not start the highlight service."
    }

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
    $lastCandidateUrl = $null
    while ((Get-Date) -lt $deadline -and -not $tunnelUrl) {
        $candidateUrl = Get-LatestTunnelUrl
        if ($candidateUrl -and (-not $requireFreshUrl -or $candidateUrl -ne $previousTunnelUrl)) {
            if ($candidateUrl -ne $lastCandidateUrl) {
                Write-Host "Checking Cloudflare URL: $candidateUrl"
                $lastCandidateUrl = $candidateUrl
            }
            try {
                $health = Invoke-RestMethod -Uri "$candidateUrl/api/health" -TimeoutSec 10
                if ($health.status -eq "ok") {
                    $tunnelUrl = $candidateUrl
                    break
                }
            }
            catch {
                # The hostname can appear in logs before Cloudflare starts routing it.
            }
        }
        Start-Sleep -Seconds 2
    }

    if (-not $tunnelUrl) {
        & docker compose @composeFiles logs --no-color --tail 80 cloudflared
        if (-not $lastCandidateUrl) {
            throw "Cloudflare did not return a new Quick Tunnel URL within $TimeoutSeconds seconds."
        }
        throw (
            "Cloudflare created $lastCandidateUrl, but its public health check did not pass. " +
            "Check whether this network allows outbound TCP or UDP port 7844, " +
            "or switch networks and run this script again."
        )
    }

    Stop-HighlightCraftOverlayService -RepositoryRoot $repoRoot `
        -Overlay "compose.ngrok.yaml" -Service "ngrok"

    $phoneUrl = "$tunnelUrl/"
    $urlPath = Join-Path $dataRoot "remote-access-url.txt"
    Set-Content -LiteralPath $urlPath -Value $phoneUrl -Encoding UTF8

    Write-Host ""
    Write-Host "Cloudflare Quick Tunnel is ready." -ForegroundColor Green
    Write-Host "Open this HTTPS link on your phone and sign in:"
    Write-Output $phoneUrl
    Write-Host ""
    $generatedAdminPasswordPath = Join-Path $dataRoot ".admin-password"
    if (Test-Path -LiteralPath $generatedAdminPasswordPath) {
        Write-Host "A generated bootstrap password is stored at $generatedAdminPasswordPath"
    }
    Write-Warning "Share accounts individually; do not send an administrator password to testers."
    Write-Warning "Keep Docker Desktop and this computer running. The URL changes if cloudflared is recreated."
    Write-Host "The latest link is also saved to $urlPath"
}
finally {
    Pop-Location
}

exit 0
