[CmdletBinding()]
param(
    [ValidatePattern("^[a-z0-9]+(?:[._-][a-z0-9]+)*$")]
    [string]$Namespace = "momonong",
    [ValidatePattern("^[a-z0-9]+(?:[._-][a-z0-9]+)*$")]
    [string]$Repository = "pingpong-auto-highlight",
    [switch]$SkipLatest,
    [switch]$SkipGpuSmokeTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$metadataPath = Join-Path $repoRoot "data\docker-build-metadata.json"
$publishedImagePath = Join-Path $repoRoot "data\published-image.txt"

function Invoke-NativeSilently {
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

function Get-ProjectVersion {
    $pyprojectPath = Join-Path $repoRoot "pyproject.toml"
    $pyproject = Get-Content -Raw -LiteralPath $pyprojectPath
    $versionMatch = [regex]::Match(
        $pyproject,
        '(?m)^version[ \t]*=[ \t]*"(?<version>[0-9]+\.[0-9]+\.[0-9]+)"[ \t]*\r?$'
    )
    if (-not $versionMatch.Success) {
        throw "Could not read a semantic project version from $pyprojectPath."
    }
    return $versionMatch.Groups["version"].Value
}

function Assert-PublicRepository {
    param(
        [Parameter(Mandatory = $true)][string]$Owner,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][datetime]$Deadline
    )

    $repositoryUrl = "https://hub.docker.com/v2/repositories/$Owner/$Name/"
    while ((Get-Date) -lt $Deadline) {
        try {
            $repositoryInfo = Invoke-RestMethod -Uri $repositoryUrl -TimeoutSec 15
            if ($repositoryInfo.is_private) {
                throw "Docker Hub created $Owner/$Name as private; change it to public before continuing."
            }
            return
        }
        catch {
            if ($_.Exception.Response -and $_.Exception.Response.StatusCode.value__ -eq 404) {
                Start-Sleep -Seconds 2
                continue
            }
            throw
        }
    }
    throw "Docker Hub did not expose $Owner/$Name as a public repository before the verification timeout."
}

Push-Location $repoRoot
try {
    if ((Invoke-NativeSilently { docker info }) -ne 0) {
        throw "Docker Desktop is not running."
    }
    if ((Invoke-NativeSilently { docker buildx version }) -ne 0) {
        throw "Docker Buildx is required to publish SBOM and provenance attestations."
    }

    $branch = (& git branch --show-current | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $branch -ne "main") {
        throw "Publish only from the main branch; current branch is '$branch'."
    }
    $workingTree = (& git status --porcelain | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $workingTree) {
        throw "Publish only from a clean Git working tree. Commit and merge the release first."
    }

    $revision = (& git rev-parse HEAD | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $revision) {
        throw "Could not determine the Git revision."
    }
    $version = Get-ProjectVersion
    $imageBase = "docker.io/$Namespace/$Repository"
    $versionTag = "${imageBase}:$version"
    $latestTag = "${imageBase}:latest"

    New-Item -ItemType Directory -Path (Split-Path -Parent $metadataPath) -Force | Out-Null

    $buildArguments = @(
        "buildx", "build",
        "--platform", "linux/amd64",
        "--pull",
        "--provenance=mode=max",
        "--sbom=true",
        "--build-arg", "APP_VERSION=$version",
        "--build-arg", "VCS_REF=$revision",
        "--metadata-file", $metadataPath,
        "--tag", $versionTag
    )
    if (-not $SkipLatest) {
        $buildArguments += @("--tag", $latestTag)
    }
    $buildArguments += @("--push", ".")

    Write-Host "Publishing $versionTag for linux/amd64 with SBOM and provenance."
    & docker @buildArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Buildx could not publish $versionTag."
    }

    Assert-PublicRepository `
        -Owner $Namespace `
        -Name $Repository `
        -Deadline (Get-Date).AddSeconds(45)

    $remoteInspection = (& docker buildx imagetools inspect $versionTag 2>&1 | Out-String)
    if ($LASTEXITCODE -ne 0) {
        throw "The published image manifest could not be inspected."
    }
    $digestMatch = [regex]::Match(
        $remoteInspection,
        '(?m)^Digest:[ \t]+(?<digest>sha256:[a-f0-9]{64})[ \t]*\r?$'
    )
    if (-not $digestMatch.Success) {
        throw "Docker Hub did not return a manifest digest for $versionTag."
    }
    $digest = $digestMatch.Groups["digest"].Value
    if ($remoteInspection -notmatch '(?m)^[ \t]*Platform:[ \t]+linux/amd64[ \t]*\r?$') {
        throw "The published manifest does not contain the required linux/amd64 platform."
    }

    if (-not $SkipLatest) {
        $latestInspection = (& docker buildx imagetools inspect $latestTag 2>&1 | Out-String)
        if ($LASTEXITCODE -ne 0) {
            throw "The published latest tag could not be inspected."
        }
        $latestDigestMatch = [regex]::Match(
            $latestInspection,
            '(?m)^Digest:[ \t]+(?<digest>sha256:[a-f0-9]{64})[ \t]*\r?$'
        )
        if (-not $latestDigestMatch.Success -or $latestDigestMatch.Groups["digest"].Value -ne $digest) {
            throw "The latest tag does not resolve to the same digest as $versionTag."
        }
    }

    & docker pull --platform linux/amd64 $versionTag
    if ($LASTEXITCODE -ne 0) {
        throw "The published image could not be pulled back from Docker Hub."
    }

    $reportedVersion = (& docker run --rm $versionTag python -c `
        "import pingpong_highlight; print(pingpong_highlight.__version__)" | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $reportedVersion -ne $version) {
        throw "Published image reports version '$reportedVersion'; expected '$version'."
    }

    if (-not $SkipGpuSmokeTest) {
        $doctorOutput = (& docker run --rm `
            --gpus all `
            -e "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video" `
            $versionTag `
            pingpong-highlight doctor 2>&1 | Out-String)
        if ($LASTEXITCODE -ne 0) {
            throw "The published image failed its NVIDIA GPU smoke test."
        }
        if (
            $doctorOutput -notmatch 'NVIDIA NVDEC\uFF1A\u53EF\u7528' -or
            $doctorOutput -notmatch 'NVIDIA NVENC\uFF1A\u53EF\u7528'
        ) {
            Write-Host $doctorOutput.TrimEnd()
            throw "The published image did not expose both NVIDIA NVDEC and NVENC."
        }
    }

    $immutableReference = "$imageBase@$digest"
    $publishedRecord = @(
        "version=$version",
        "tag=$versionTag",
        "digest=$digest",
        "immutable=$immutableReference",
        "revision=$revision",
        "platform=linux/amd64"
    ) -join [Environment]::NewLine
    [IO.File]::WriteAllText(
        $publishedImagePath,
        $publishedRecord + [Environment]::NewLine,
        [Text.UTF8Encoding]::new($false)
    )

    Write-Host ""
    Write-Host "Docker Hub publication verified." -ForegroundColor Green
    Write-Host "Tag:       $versionTag"
    Write-Host "Digest:    $digest"
    Write-Host "Platform:  linux/amd64"
    Write-Host "GPU check: $(if ($SkipGpuSmokeTest) { 'skipped' } else { 'NVDEC and NVENC available' })"
    Write-Host "Record:    $publishedImagePath"
}
finally {
    Pop-Location
}

exit 0
