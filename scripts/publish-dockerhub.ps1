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

function Test-IsHttpNotFound {
    param([Parameter(Mandatory = $true)][object]$ErrorRecord)

    $responseProperty = $ErrorRecord.Exception.PSObject.Properties["Response"]
    if ($null -eq $responseProperty -or $null -eq $responseProperty.Value) {
        return $false
    }
    $statusCodeProperty = $responseProperty.Value.PSObject.Properties["StatusCode"]
    if ($null -eq $statusCodeProperty) {
        return $false
    }
    return [int]$statusCodeProperty.Value -eq 404
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
                throw "Docker Hub repository $Owner/$Name is private; change it to public before continuing."
            }
            return $repositoryInfo
        }
        catch {
            if (Test-IsHttpNotFound -ErrorRecord $_) {
                Start-Sleep -Seconds 2
                continue
            }
            throw
        }
    }
    throw "Docker Hub did not expose $Owner/$Name as a public repository before the verification timeout."
}

function Assert-ImmutableVersionTagPolicy {
    param(
        [Parameter(Mandatory = $true)][object]$RepositoryInfo,
        [Parameter(Mandatory = $true)][string]$Owner,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Version,
        [Parameter(Mandatory = $true)][string]$CandidateTagName
    )

    $recommendedPattern = '^[0-9]+\.[0-9]+\.[0-9]+$'
    $settingsProperty = $RepositoryInfo.PSObject.Properties["immutable_tags_settings"]
    if ($null -eq $settingsProperty) {
        throw (
            "Docker Hub did not report immutable tag settings for $Owner/$Name. " +
            "In Docker Hub Settings > General, choose 'Specific tags are immutable' " +
            "with regex $recommendedPattern, then retry."
        )
    }

    $settings = $settingsProperty.Value
    if ($null -eq $settings) {
        throw (
            "Docker Hub returned empty immutable tag settings for $Owner/$Name. " +
            "In Docker Hub Settings > General, choose 'Specific tags are immutable' " +
            "with regex $recommendedPattern, then retry."
        )
    }
    $enabledProperty = $settings.PSObject.Properties["enabled"]
    $rulesProperty = $settings.PSObject.Properties["rules"]
    if ($null -eq $enabledProperty -or $enabledProperty.Value -ne $true) {
        throw (
            "Docker Hub immutable tags are not enabled for $Owner/$Name. " +
            "In Docker Hub Settings > General, choose 'Specific tags are immutable' " +
            "with regex $recommendedPattern, then retry."
        )
    }
    if ($null -eq $rulesProperty) {
        throw (
            "Docker Hub did not report immutable tag rules for $Owner/$Name. " +
            "In Docker Hub Settings > General, use regex $recommendedPattern, then retry."
        )
    }

    $versionIsImmutable = $false
    $unsafeRules = @()
    foreach ($ruleValue in @($rulesProperty.Value)) {
        $rule = [string]$ruleValue
        if (-not $rule) {
            continue
        }
        try {
            $compiledRule = [regex]::new($rule)
        }
        catch {
            throw "Docker Hub returned an invalid immutable tag regex '$rule' for $Owner/$Name."
        }
        if ($compiledRule.IsMatch($Version)) {
            $versionIsImmutable = $true
        }
        if ($compiledRule.IsMatch("latest") -or $compiledRule.IsMatch($CandidateTagName)) {
            $unsafeRules += $rule
        }
    }

    if (-not $versionIsImmutable) {
        throw (
            "Docker Hub's immutable tag rules do not protect version $Version for $Owner/$Name. " +
            "In Docker Hub Settings > General, use regex $recommendedPattern, then retry."
        )
    }
    if ($unsafeRules.Count -gt 0) {
        $unsafeRuleList = ($unsafeRules | Sort-Object -Unique) -join ", "
        throw (
            "Docker Hub immutable rule(s) $unsafeRuleList also match 'latest' or " +
            "'$CandidateTagName'. Use the version-only regex $recommendedPattern in " +
            "Settings > General so candidate validation and latest promotion remain possible."
        )
    }
}

function Assert-VersionTagAvailable {
    param(
        [Parameter(Mandatory = $true)][string]$Owner,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Version
    )

    $tagUrl = "https://hub.docker.com/v2/repositories/$Owner/$Name/tags/$Version/"
    try {
        $null = Invoke-RestMethod -Uri $tagUrl -TimeoutSec 15
    }
    catch {
        if (Test-IsHttpNotFound -ErrorRecord $_) {
            return
        }
        throw
    }
    throw "Version tag $Owner/$Name`:$Version already exists; published versions are immutable."
}

function Write-PublishedImageRecord {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Version,
        [Parameter(Mandatory = $true)][string]$VersionTag,
        [Parameter(Mandatory = $true)][string]$Digest,
        [Parameter(Mandatory = $true)][string]$ImmutableReference,
        [Parameter(Mandatory = $true)][string]$Revision,
        [Parameter(Mandatory = $true)][string]$CandidateTag,
        [Parameter(Mandatory = $true)][ValidateSet("pending", "verified", "skipped")][string]$LatestStatus
    )

    $publishedRecord = @(
        "version=$Version",
        "tag=$VersionTag",
        "digest=$Digest",
        "immutable=$ImmutableReference",
        "revision=$Revision",
        "platform=linux/amd64",
        "candidate=$CandidateTag",
        "latest=$LatestStatus"
    ) -join [Environment]::NewLine
    [IO.File]::WriteAllText(
        $Path,
        $publishedRecord + [Environment]::NewLine,
        [Text.UTF8Encoding]::new($false)
    )
}

function New-LatestRecoveryMessage {
    param(
        [Parameter(Mandatory = $true)][string]$VersionTag,
        [Parameter(Mandatory = $true)][string]$LatestTag,
        [Parameter(Mandatory = $true)][string]$ImmutableReference,
        [Parameter(Mandatory = $true)][string]$Digest,
        [Parameter(Mandatory = $true)][string]$PublishedImagePath,
        [Parameter(Mandatory = $true)][string]$Failure
    )

    return @(
        "The immutable version $VersionTag is already published and verified as $ImmutableReference.",
        "The latest tag was not verified; $PublishedImagePath remains marked latest=pending.",
        "Do not rerun the full publisher for this version. After resolving registry access, run:",
        "  docker buildx imagetools create --tag $LatestTag $ImmutableReference",
        "  docker buildx imagetools inspect $LatestTag",
        "Confirm that the reported digest is $Digest.",
        "After confirmation, change only latest=pending to latest=verified in the release record.",
        "Original latest-promotion failure: $Failure"
    ) -join [Environment]::NewLine
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
    $candidateTagName = "candidate-$version-$($revision.Substring(0, 12))"
    $candidateTag = "${imageBase}:$candidateTagName"

    $repositoryInfo = Assert-PublicRepository `
        -Owner $Namespace `
        -Name $Repository `
        -Deadline (Get-Date).AddSeconds(15)
    Assert-ImmutableVersionTagPolicy `
        -RepositoryInfo $repositoryInfo `
        -Owner $Namespace `
        -Name $Repository `
        -Version $version `
        -CandidateTagName $candidateTagName
    Assert-VersionTagAvailable -Owner $Namespace -Name $Repository -Version $version

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
        "--tag", $candidateTag
    )
    $buildArguments += @("--push", ".")

    Write-Host "Publishing validation candidate $candidateTag for linux/amd64."
    & docker @buildArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Buildx could not publish the validation candidate."
    }

    $remoteInspection = (& docker buildx imagetools inspect $candidateTag 2>&1 | Out-String)
    if ($LASTEXITCODE -ne 0) {
        throw "The candidate image manifest could not be inspected."
    }
    $digestMatch = [regex]::Match(
        $remoteInspection,
        '(?m)^Digest:[ \t]+(?<digest>sha256:[a-f0-9]{64})[ \t]*\r?$'
    )
    if (-not $digestMatch.Success) {
        throw "Docker Hub did not return a manifest digest for $candidateTag."
    }
    $digest = $digestMatch.Groups["digest"].Value
    if ($remoteInspection -notmatch '(?m)^[ \t]*Platform:[ \t]+linux/amd64[ \t]*\r?$') {
        throw "The published manifest does not contain the required linux/amd64 platform."
    }

    $immutableReference = "$imageBase@$digest"
    & docker pull --platform linux/amd64 $immutableReference
    if ($LASTEXITCODE -ne 0) {
        throw "The candidate image could not be pulled back by immutable digest."
    }

    $reportedVersion = (& docker run --rm $immutableReference python -c `
        "import pingpong_highlight; print(pingpong_highlight.__version__)" | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $reportedVersion -ne $version) {
        throw "Candidate image reports version '$reportedVersion'; expected '$version'."
    }

    if (-not $SkipGpuSmokeTest) {
        $doctorOutput = (& docker run --rm `
            --gpus all `
            -e "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video" `
            $immutableReference `
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

    # Re-check immediately before promotion. The Docker Hub immutable-tag policy
    # is the authoritative atomic guard; the availability check is defense in depth.
    $repositoryInfo = Assert-PublicRepository `
        -Owner $Namespace `
        -Name $Repository `
        -Deadline (Get-Date).AddSeconds(15)
    Assert-ImmutableVersionTagPolicy `
        -RepositoryInfo $repositoryInfo `
        -Owner $Namespace `
        -Name $Repository `
        -Version $version `
        -CandidateTagName $candidateTagName
    Assert-VersionTagAvailable -Owner $Namespace -Name $Repository -Version $version
    $versionPromotionArguments = @(
        "buildx", "imagetools", "create",
        "--tag", $versionTag,
        $immutableReference
    )
    & docker @versionPromotionArguments
    $versionPromotionExitCode = $LASTEXITCODE

    $versionInspection = (& docker buildx imagetools inspect $versionTag 2>&1 | Out-String)
    if ($LASTEXITCODE -ne 0) {
        throw (
            "Candidate validation passed, but the official version promotion could not be verified. " +
            "No latest update was attempted. Inspect $versionTag before retrying; the immutable " +
            "tag may already exist if the registry accepted the write before the client failed."
        )
    }
    $versionDigestMatch = [regex]::Match(
        $versionInspection,
        '(?m)^Digest:[ \t]+(?<digest>sha256:[a-f0-9]{64})[ \t]*\r?$'
    )
    if (-not $versionDigestMatch.Success -or $versionDigestMatch.Groups["digest"].Value -ne $digest) {
        throw (
            "The official version tag does not resolve to the validated digest. " +
            "No latest update was attempted."
        )
    }
    if ($versionPromotionExitCode -ne 0) {
        Write-Warning (
            "Version promotion returned exit code $versionPromotionExitCode, but registry inspection " +
            "confirmed the validated digest; continuing without republishing the immutable tag."
        )
    }

    $latestStatus = $(if ($SkipLatest) { "skipped" } else { "pending" })
    try {
        Write-PublishedImageRecord `
            -Path $publishedImagePath `
            -Version $version `
            -VersionTag $versionTag `
            -Digest $digest `
            -ImmutableReference $immutableReference `
            -Revision $revision `
            -CandidateTag $candidateTag `
            -LatestStatus $latestStatus
    }
    catch {
        throw (
            "The immutable version $versionTag is published as $immutableReference, but its durable " +
            "release record could not be written to $publishedImagePath. No latest update was attempted. " +
            "Original record failure: $($_.Exception.Message)"
        )
    }

    if (-not $SkipLatest) {
        $latestPromotionArguments = @(
            "buildx", "imagetools", "create",
            "--tag", $latestTag,
            $immutableReference
        )
        & docker @latestPromotionArguments
        $latestPromotionExitCode = $LASTEXITCODE

        $latestInspection = (& docker buildx imagetools inspect $latestTag 2>&1 | Out-String)
        if ($LASTEXITCODE -ne 0) {
            throw (New-LatestRecoveryMessage `
                -VersionTag $versionTag `
                -LatestTag $latestTag `
                -ImmutableReference $immutableReference `
                -Digest $digest `
                -PublishedImagePath $publishedImagePath `
                -Failure "The promoted latest tag could not be inspected.")
        }
        $latestDigestMatch = [regex]::Match(
            $latestInspection,
            '(?m)^Digest:[ \t]+(?<digest>sha256:[a-f0-9]{64})[ \t]*\r?$'
        )
        if (-not $latestDigestMatch.Success -or $latestDigestMatch.Groups["digest"].Value -ne $digest) {
            throw (New-LatestRecoveryMessage `
                -VersionTag $versionTag `
                -LatestTag $latestTag `
                -ImmutableReference $immutableReference `
                -Digest $digest `
                -PublishedImagePath $publishedImagePath `
                -Failure "The latest tag does not resolve to the validated digest.")
        }
        if ($latestPromotionExitCode -ne 0) {
            Write-Warning (
                "Latest promotion returned exit code $latestPromotionExitCode, but registry inspection " +
                "confirmed the validated digest."
            )
        }

        try {
            Write-PublishedImageRecord `
                -Path $publishedImagePath `
                -Version $version `
                -VersionTag $versionTag `
                -Digest $digest `
                -ImmutableReference $immutableReference `
                -Revision $revision `
                -CandidateTag $candidateTag `
                -LatestStatus "verified"
        }
        catch {
            throw (
                "Both $versionTag and $latestTag resolve to $immutableReference, but the durable release " +
                "record could not be updated from latest=pending to latest=verified. " +
                "Original record failure: $($_.Exception.Message)"
            )
        }
    }

    Write-Host ""
    Write-Host "Docker Hub publication verified." -ForegroundColor Green
    Write-Host "Tag:       $versionTag"
    Write-Host "Digest:    $digest"
    Write-Host "Platform:  linux/amd64"
    Write-Host "Latest:    $(if ($SkipLatest) { 'skipped' } else { 'verified' })"
    Write-Host "GPU check: $(if ($SkipGpuSmokeTest) { 'skipped' } else { 'NVDEC and NVENC available' })"
    Write-Host "Record:    $publishedImagePath"
}
finally {
    Pop-Location
}

exit 0
