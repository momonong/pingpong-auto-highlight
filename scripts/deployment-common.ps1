function Assert-HighlightCraftComposeVersion {
    [CmdletBinding()]
    param()

    $versionOutput = (& docker compose version --short 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Compose v2.30.0 or newer is required. Install or update the Docker Compose plugin."
    }
    $match = [regex]::Match($versionOutput, '(?<!\d)(\d+)\.(\d+)\.(\d+)')
    if (-not $match.Success) {
        throw "Could not read the Docker Compose version from '$versionOutput'."
    }
    $installed = [Version]::new(
        [int]$match.Groups[1].Value,
        [int]$match.Groups[2].Value,
        [int]$match.Groups[3].Value
    )
    if ($installed -lt [Version]::new(2, 30, 0)) {
        throw "Docker Compose v2.30.0 or newer is required; found v$installed."
    }
}

function Get-HighlightCraftDataRoot {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][string]$RepositoryRoot)

    $configuredPath = $env:PINGPONG_DATA_PATH
    if (-not $configuredPath) {
        $envPath = Join-Path $RepositoryRoot ".env"
        if (Test-Path -LiteralPath $envPath) {
            $assignment = @(
                Get-Content -LiteralPath $envPath |
                    Where-Object { $_ -match '^\s*PINGPONG_DATA_PATH\s*=' } |
                    Select-Object -Last 1
            )
            if ($assignment.Count -gt 0) {
                $configuredPath = ($assignment[0] -split '=', 2)[1].Trim()
                if (
                    $configuredPath.Length -ge 2 -and
                    (
                        ($configuredPath.StartsWith('"') -and $configuredPath.EndsWith('"')) -or
                        ($configuredPath.StartsWith("'") -and $configuredPath.EndsWith("'"))
                    )
                ) {
                    $configuredPath = $configuredPath.Substring(1, $configuredPath.Length - 2)
                }
            }
        }
    }

    if (-not $configuredPath) {
        $configuredPath = "./data"
    }
    if ([IO.Path]::IsPathRooted($configuredPath)) {
        return [IO.Path]::GetFullPath($configuredPath)
    }
    return [IO.Path]::GetFullPath((Join-Path $RepositoryRoot $configuredPath))
}

function Get-HighlightCraftServiceContainerId {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$RepositoryRoot,
        [Parameter(Mandatory = $true)][string]$Service,
        [string]$Overlay = ""
    )

    $arguments = @("compose", "-f", (Join-Path $RepositoryRoot "compose.yaml"))
    if ($Overlay) {
        $arguments += @("-f", (Join-Path $RepositoryRoot $Overlay))
    }
    $arguments += @("ps", "-a", "-q", $Service)
    $idOutput = & docker @arguments 2>$null
    $idExitCode = $LASTEXITCODE
    if ($idExitCode -ne 0) {
        throw "Could not query the $Service container; unable to prove deployment safety."
    }
    $containerIds = @(
        $idOutput |
            ForEach-Object { $_.ToString().Trim() } |
            Where-Object { $_ }
    )
    if ($containerIds.Count -gt 1) {
        throw "Found multiple $Service containers; unable to prove deployment safety."
    }
    if ($containerIds.Count -eq 0) {
        return $null
    }
    return $containerIds[0]
}

function Get-HighlightCraftContainerInspection {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][string]$ContainerId)

    $inspectionOutput = & docker inspect $ContainerId 2>$null
    $inspectionExitCode = $LASTEXITCODE
    if ($inspectionExitCode -ne 0) {
        throw "Could not inspect Docker container $ContainerId; unable to prove deployment safety."
    }
    $json = ($inspectionOutput | Out-String).Trim()
    if (-not $json) {
        throw "Docker returned no inspection data for $ContainerId; unable to prove deployment safety."
    }
    try {
        return @(ConvertFrom-Json $json)[0]
    }
    catch {
        throw "Docker returned invalid inspection data; unable to prove deployment safety."
    }
}

function Get-HighlightCraftApplicationInspection {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][string]$RepositoryRoot)

    $containerId = Get-HighlightCraftServiceContainerId `
        -RepositoryRoot $RepositoryRoot -Service "pingpong-highlight"
    if (-not $containerId) {
        return $null
    }
    return Get-HighlightCraftContainerInspection -ContainerId $containerId
}

function Get-HighlightCraftPublishedPort {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)]$Inspection)

    $bindings = @($Inspection.NetworkSettings.Ports."8000/tcp")
    if ($bindings.Count -eq 0 -or -not $bindings[0].HostPort) {
        throw "HighlightCraft does not have a published host port for container port 8000."
    }
    return [int]$bindings[0].HostPort
}

function Get-HighlightCraftSecret {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Description
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "HighlightCraft is running, but its $Description is unavailable; refusing to restart it."
    }
    try {
        $secret = (Get-Content -Raw -LiteralPath $Path).Trim()
    }
    catch {
        throw "HighlightCraft is running, but its $Description cannot be read; refusing to restart it."
    }
    if (-not $secret) {
        throw "HighlightCraft is running, but its $Description is empty; refusing to restart it."
    }
    return $secret
}

function Assert-HighlightCraftNoActiveWork {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$RepositoryRoot,
        [Parameter(Mandatory = $true)][string]$DataRoot
    )

    $inspection = Get-HighlightCraftApplicationInspection -RepositoryRoot $RepositoryRoot
    if ($null -eq $inspection) {
        return
    }
    if ($inspection.State.Restarting) {
        throw "HighlightCraft is restarting; unable to verify active work safely."
    }
    if (-not $inspection.State.Running) {
        return
    }

    $maintenanceTokenPath = Join-Path $DataRoot ".maintenance-token"
    $legacyTokenPath = Join-Path $DataRoot ".upload-token"
    if (Test-Path -LiteralPath $maintenanceTokenPath) {
        $maintenanceToken = Get-HighlightCraftSecret `
            -Path $maintenanceTokenPath -Description "maintenance token"
    }
    else {
        # A running 1.3.0 service has no maintenance token. Its legacy token is
        # sufficient to receive the explicit 404 that activates the fallback.
        $maintenanceToken = Get-HighlightCraftSecret `
            -Path $legacyTokenPath -Description "access token"
    }
    $hostPort = Get-HighlightCraftPublishedPort -Inspection $inspection
    $headers = @{ "X-Upload-Token" = $maintenanceToken }
    try {
        $workState = Invoke-RestMethod `
            -Uri "http://127.0.0.1:$hostPort/api/maintenance/active-work" `
            -Headers $headers -TimeoutSec 10
    }
    catch {
        $statusCode = $null
        try {
            $statusCode = [int]$_.Exception.Response.StatusCode
        }
        catch {
            # PowerShell editions expose HTTP status details differently.
        }
        if ($statusCode -ne 404) {
            throw "Could not verify HighlightCraft's current work state; refusing to restart it."
        }

        # Version 1.3.0 does not have the maintenance endpoint. Always load its
        # own legacy token here: a previous 1.4.0 start may have left a distinct
        # maintenance token behind before the operator rolled back to 1.3.0.
        $legacyToken = Get-HighlightCraftSecret `
            -Path $legacyTokenPath -Description "legacy access token"
        $legacyHeaders = @{ "X-Upload-Token" = $legacyToken }
        try {
            # Keep jobs last. A transfer that finishes while this fallback is
            # sampled can only become a queued job before the final request.
            $imports = @(
                (Invoke-RestMethod -Uri "http://127.0.0.1:$hostPort/api/drive-imports?scope=all" `
                    -Headers $legacyHeaders -TimeoutSec 10).imports
            )
            $uploads = @(
                (Invoke-RestMethod -Uri "http://127.0.0.1:$hostPort/api/uploads?scope=all" `
                    -Headers $legacyHeaders -TimeoutSec 10).uploads
            )
            $jobs = @(
                (Invoke-RestMethod -Uri "http://127.0.0.1:$hostPort/api/jobs?scope=all" `
                    -Headers $legacyHeaders -TimeoutSec 10).jobs
            )
        }
        catch {
            throw "Could not verify the legacy HighlightCraft work state; refusing to restart it."
        }
        $legacyActiveJobs = @(
            $jobs | Where-Object { $_.status -in @("queued", "processing") }
        ).Count
        $legacyActiveImports = @(
            $imports | Where-Object { $_.status -in @("queued", "resolving", "downloading") }
        ).Count
        $legacyIncompleteUploads = @(
            $uploads | Where-Object { [int64]$_.offset -lt [int64]$_.size }
        ).Count
        $workState = [pscustomobject]@{
            active = [bool](
                $legacyActiveJobs -or $legacyActiveImports -or $legacyIncompleteUploads
            )
            jobs = [pscustomobject]@{
                queued = @($jobs | Where-Object status -eq "queued").Count
                processing = @($jobs | Where-Object status -eq "processing").Count
                completed = @($jobs | Where-Object status -eq "completed").Count
            }
            drive_imports = [pscustomobject]@{ active = $legacyActiveImports }
            uploads = [pscustomobject]@{ incomplete = $legacyIncompleteUploads }
        }
    }

    $activeJobs = [int]$workState.jobs.queued + [int]$workState.jobs.processing
    $activeImports = [int]$workState.drive_imports.active
    $incompleteUploads = [int]$workState.uploads.incomplete
    if ($workState.active) {
        throw (
            "HighlightCraft still has active work " +
            "($activeJobs jobs, $activeImports Drive imports, " +
            "$incompleteUploads incomplete uploads). " +
            "Wait for it to finish, or remove abandoned uploads in the UI, then run again."
        )
    }

    Write-Host (
        "Safe to switch modes: $([int]$workState.jobs.completed) " +
        "completed jobs and no active transfers or processing."
    )
}

function Stop-HighlightCraftOverlayService {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$RepositoryRoot,
        [Parameter(Mandatory = $true)][string]$Overlay,
        [Parameter(Mandatory = $true)][string]$Service
    )

    $arguments = @(
        "compose",
        "-f", (Join-Path $RepositoryRoot "compose.yaml"),
        "-f", (Join-Path $RepositoryRoot $Overlay),
        "stop", $Service
    )
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & docker @arguments *> $null
        $stopExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($stopExitCode -ne 0) {
        throw "Docker Compose could not stop the $Service tunnel service."
    }

    $containerId = Get-HighlightCraftServiceContainerId `
        -RepositoryRoot $RepositoryRoot -Overlay $Overlay -Service $Service
    if (-not $containerId) {
        return
    }
    $inspection = Get-HighlightCraftContainerInspection -ContainerId $containerId
    if ($inspection.State.Running -or $inspection.State.Restarting) {
        throw "The $Service tunnel service is still running after Docker Compose stopped it."
    }
}
