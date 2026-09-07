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
$composeFiles += @("-f", (Join-Path $repoRoot "compose.localhost.yaml"))

function Invoke-NativeCommandSilently {
    param([Parameter(Mandatory = $true)][scriptblock]$Command)

    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $Command *> $null
        return $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
}

function Get-ContainerInspection {
    param([Parameter(Mandatory = $true)][string]$ContainerId)

    $inspectionOutput = & docker inspect $ContainerId 2>&1
    $inspectionExitCode = $LASTEXITCODE
    if ($inspectionExitCode -ne 0) {
        throw "Could not inspect Docker container $ContainerId; unable to prove localhost safety."
    }
    $json = ($inspectionOutput | Out-String).Trim()
    if (-not $json) {
        throw "Docker returned no inspection data for $ContainerId; unable to prove localhost safety."
    }
    try {
        return @(ConvertFrom-Json $json)[0]
    }
    catch {
        throw "Docker returned invalid inspection data; unable to prove localhost safety."
    }
}

function Get-ComposeServiceContainerId {
    param(
        [Parameter(Mandatory = $true)][string]$Service,
        [string]$Overlay = ""
    )

    $arguments = @("compose", "-f", (Join-Path $repoRoot "compose.yaml"))
    if ($Overlay) {
        $arguments += @("-f", (Join-Path $repoRoot $Overlay))
    }
    $arguments += @("ps", "-a", "-q", $Service)
    $idOutput = & docker @arguments 2>&1
    $idExitCode = $LASTEXITCODE
    if ($idExitCode -ne 0) {
        throw "Could not query the $Service container; unable to prove localhost safety."
    }
    $containerIds = @(
        $idOutput |
            ForEach-Object { $_.ToString().Trim() } |
            Where-Object { $_ }
    )
    if ($containerIds.Count -gt 1) {
        throw "Found multiple $Service containers; unable to prove localhost safety."
    }
    if ($containerIds.Count -eq 0) {
        return $null
    }
    return $containerIds[0]
}

function Get-HighlightContainerInspection {
    $containerId = Get-ComposeServiceContainerId -Service "pingpong-highlight"
    if (-not $containerId) {
        return $null
    }
    return Get-ContainerInspection -ContainerId $containerId
}

function Get-PublishedPort {
    param([Parameter(Mandatory = $true)]$Inspection)

    $bindings = @($Inspection.NetworkSettings.Ports."8000/tcp")
    if ($bindings.Count -eq 0 -or -not $bindings[0].HostPort) {
        throw "HighlightCraft does not have a published host port for container port 8000."
    }
    return [int]$bindings[0].HostPort
}

function Assert-NoActiveWork {
    Assert-HighlightCraftNoActiveWork -RepositoryRoot $repoRoot -DataRoot $dataRoot
}

function Stop-TunnelService {
    param(
        [Parameter(Mandatory = $true)][string]$Overlay,
        [Parameter(Mandatory = $true)][string]$Service
    )

    $overlayPath = Join-Path $repoRoot $Overlay
    $exitCode = Invoke-NativeCommandSilently {
        docker compose -f (Join-Path $repoRoot "compose.yaml") -f $overlayPath stop $Service
    }
    if ($exitCode -ne 0) {
        throw "Docker Compose could not stop the $Service tunnel service."
    }
    if (Test-TunnelRunning -Overlay $Overlay -Service $Service) {
        throw "The $Service tunnel service is still running after Docker Compose stopped it."
    }
}

function Test-TunnelRunning {
    param(
        [Parameter(Mandatory = $true)][string]$Overlay,
        [Parameter(Mandatory = $true)][string]$Service
    )

    $containerId = Get-ComposeServiceContainerId -Service $Service -Overlay $Overlay
    if (-not $containerId) {
        return $false
    }
    $inspection = Get-ContainerInspection -ContainerId $containerId
    return (
        $null -ne $inspection -and
        ($inspection.State.Running -or $inspection.State.Restarting)
    )
}

function Clear-StaleRemoteUrls {
    $staleRemoteUrlPaths = @(
        (Join-Path $dataRoot "remote-access-url.txt"),
        (Join-Path $dataRoot "ngrok-access-url.txt")
    )
    foreach ($staleUrlPath in $staleRemoteUrlPaths) {
        if (Test-Path -LiteralPath $staleUrlPath) {
            [IO.File]::Delete($staleUrlPath)
        }
    }
}

function Stop-ServicesFailClosed {
    Write-Warning (
        "Localhost safety verification failed after mode switching began. " +
        "Stopping HighlightCraft and both tunnel services to fail closed."
    )
    $cleanupProblems = @()
    if ((Invoke-NativeCommandSilently {
        docker compose -f (Join-Path $repoRoot "compose.yaml") `
            -f (Join-Path $repoRoot "compose.ngrok.yaml") stop ngrok
    }) -ne 0) {
        $cleanupProblems += "ngrok stop command failed"
    }
    if ((Invoke-NativeCommandSilently {
        docker compose -f (Join-Path $repoRoot "compose.yaml") `
            -f (Join-Path $repoRoot "compose.cloudflare.yaml") stop cloudflared
    }) -ne 0) {
        $cleanupProblems += "cloudflared stop command failed"
    }
    if ((Invoke-NativeCommandSilently {
        docker compose @composeFiles stop pingpong-highlight
    }) -ne 0) {
        $cleanupProblems += "pingpong-highlight stop command failed"
    }

    $unsafeServices = @()
    try {
        if (Test-TunnelRunning -Overlay "compose.ngrok.yaml" -Service "ngrok") {
            $unsafeServices += "ngrok"
        }
    }
    catch {
        $cleanupProblems += "could not verify ngrok state"
    }
    try {
        if (Test-TunnelRunning -Overlay "compose.cloudflare.yaml" -Service "cloudflared") {
            $unsafeServices += "cloudflared"
        }
    }
    catch {
        $cleanupProblems += "could not verify cloudflared state"
    }
    try {
        $appInspection = Get-HighlightContainerInspection
        if (
            $null -ne $appInspection -and
            ($appInspection.State.Running -or $appInspection.State.Restarting)
        ) {
            $unsafeServices += "pingpong-highlight"
        }
    }
    catch {
        $cleanupProblems += "could not verify pingpong-highlight state"
    }
    if ($unsafeServices.Count -gt 0 -or $cleanupProblems.Count -gt 0) {
        $details = @()
        if ($unsafeServices.Count -gt 0) {
            $details += "still active: $($unsafeServices -join ', ')"
        }
        if ($cleanupProblems.Count -gt 0) {
            $details += "cleanup errors: $($cleanupProblems -join ', ')"
        }
        Write-Warning (
            "Automatic fail-closed cleanup could not be verified ($($details -join '; ')). " +
            "Open Docker Desktop and stop these containers manually before continuing."
        )
    }
}

$previousIgnoreOrphans = $env:COMPOSE_IGNORE_ORPHANS
$locationPushed = $false
$transitionStarted = $false
$tunnelsSecured = $false
$applicationMutationStarted = $false
try {
    if ((Invoke-NativeCommandSilently { docker info }) -ne 0) {
        throw "Docker Desktop is not running. Start Docker Desktop and try again."
    }

    Push-Location $repoRoot
    $locationPushed = $true
    $env:COMPOSE_IGNORE_ORPHANS = "true"

    Assert-NoActiveWork
    $transitionStarted = $true
    Stop-TunnelService -Overlay "compose.ngrok.yaml" -Service "ngrok"
    Stop-TunnelService -Overlay "compose.cloudflare.yaml" -Service "cloudflared"
    $tunnelsSecured = $true
    Clear-StaleRemoteUrls

    if ($UsePublishedImage) {
        & docker compose @composeFiles pull pingpong-highlight
        if ($LASTEXITCODE -ne 0) {
            throw "Docker Compose could not pull the published HighlightCraft image from Docker Hub."
        }
    }
    else {
        & docker compose @composeFiles build pingpong-highlight
        if ($LASTEXITCODE -ne 0) {
            throw "Docker Compose could not build the HighlightCraft image."
        }
    }

    try {
        Assert-NoActiveWork
    }
    catch {
        Write-Warning (
            "The tunnel services are stopped, but new active work appeared during image preparation. " +
            "The existing HighlightCraft service was left running; wait for it to finish and run again."
        )
        throw
    }

    $applicationMutationStarted = $true
    & docker compose @composeFiles up -d --no-build --no-deps --wait `
        --wait-timeout $TimeoutSeconds pingpong-highlight
    if ($LASTEXITCODE -ne 0) {
        if (-not $CpuOnly) {
            throw (
                "Docker Compose could not start the GPU highlight service. " +
                "Confirm NVIDIA Container Toolkit is available; use -CpuOnly only as a temporary fallback."
            )
        }
        throw "Docker Compose could not start the highlight service."
    }

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $healthy = $false
    do {
        $inspection = Get-HighlightContainerInspection
        if ($null -ne $inspection -and $inspection.State.Running) {
            $hostPort = Get-PublishedPort -Inspection $inspection
            try {
                $health = Invoke-RestMethod `
                    -Uri "http://127.0.0.1:$hostPort/api/health" -TimeoutSec 5
                if ($health.status -eq "ok") {
                    $healthy = $true
                    break
                }
            }
            catch {
                # The published port can appear just before the application is ready.
            }
        }
        Start-Sleep -Seconds 1
    } while ((Get-Date) -lt $deadline)
    if (-not $healthy) {
        throw "HighlightCraft did not become healthy within $TimeoutSeconds seconds."
    }

    $bindings = @($inspection.NetworkSettings.Ports."8000/tcp")
    $unsafeBindings = @($bindings | Where-Object { $_.HostIp -ne "127.0.0.1" })
    if ($bindings.Count -eq 0 -or $unsafeBindings.Count -gt 0) {
        throw "Localhost safety check failed: HighlightCraft is not bound exclusively to 127.0.0.1."
    }
    if (
        (Test-TunnelRunning -Overlay "compose.ngrok.yaml" -Service "ngrok") -or
        (Test-TunnelRunning -Overlay "compose.cloudflare.yaml" -Service "cloudflared")
    ) {
        throw "Localhost safety check failed: a tunnel container is still running."
    }

    $localUrl = "http://127.0.0.1:$hostPort/"
    $localUrlPath = Join-Path $dataRoot "local-access-url.txt"
    [IO.File]::WriteAllText(
        $localUrlPath,
        $localUrl + [Environment]::NewLine,
        [Text.UTF8Encoding]::new($false)
    )
    Write-Host ""
    Write-Host "HighlightCraft localhost-only mode is ready." -ForegroundColor Green
    Write-Host "Open this link on this computer:"
    Write-Output $localUrl
    Write-Host ""
    $generatedAdminPasswordPath = Join-Path $dataRoot ".admin-password"
    Write-Host "Sign in with your HighlightCraft username and password."
    if (Test-Path -LiteralPath $generatedAdminPasswordPath) {
        Write-Host "A generated bootstrap password is stored at $generatedAdminPasswordPath"
    }
    Write-Host "The link is also saved to $localUrlPath"
    Write-Host "ngrok and Cloudflare Tunnel are stopped; browser video traffic stays on this computer."
    Write-Warning "127.0.0.1 works only on this computer. A phone cannot open this address."
}
catch {
    if ($transitionStarted -and (-not $tunnelsSecured -or $applicationMutationStarted)) {
        Stop-ServicesFailClosed
    }
    throw
}
finally {
    if ($locationPushed) {
        Pop-Location
    }
    if ($null -eq $previousIgnoreOrphans) {
        Remove-Item Env:COMPOSE_IGNORE_ORPHANS -ErrorAction SilentlyContinue
    }
    else {
        $env:COMPOSE_IGNORE_ORPHANS = $previousIgnoreOrphans
    }
}
