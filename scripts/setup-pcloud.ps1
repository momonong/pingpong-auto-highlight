$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$version = "1.75.0"
$archiveName = "rclone-v$version-windows-amd64.zip"
$expectedSha256 = "203581f0a7baeae873f2347483a798c79e2eaf5c384a4e9d866aa374f1c89ac0"
$toolRoot = Join-Path $projectRoot ".tools\rclone\v$version"
$archivePath = Join-Path $toolRoot $archiveName
$rclonePath = Join-Path $toolRoot "rclone-v$version-windows-amd64\rclone.exe"
$configDirectory = Join-Path $projectRoot "secrets\rclone"
$configPath = Join-Path $configDirectory "rclone.conf"

New-Item -ItemType Directory -Force -Path $toolRoot | Out-Null
New-Item -ItemType Directory -Force -Path $configDirectory | Out-Null

if (-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
    $downloadUrl = "https://downloads.rclone.org/v$version/$archiveName"
    Write-Host "Downloading rclone v$version from the official release site..."
    Invoke-WebRequest -Uri $downloadUrl -OutFile $archivePath
}

$actualSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $archivePath).Hash.ToLowerInvariant()
if ($actualSha256 -ne $expectedSha256) {
    throw "rclone archive checksum mismatch; expected $expectedSha256, got $actualSha256"
}

if (-not (Test-Path -LiteralPath $rclonePath -PathType Leaf)) {
    Expand-Archive -LiteralPath $archivePath -DestinationPath $toolRoot -Force
}
if (-not (Test-Path -LiteralPath $rclonePath -PathType Leaf)) {
    throw "rclone executable was not found after extraction: $rclonePath"
}

Write-Host ""
Write-Host "In rclone config:"
Write-Host "  1. Create a new remote named: highlightcraft-pcloud"
Write-Host "  2. Choose storage type: pcloud"
Write-Host "  3. Leave client_id and client_secret blank"
Write-Host "  4. Use the browser to sign in and authorize once"
Write-Host ""
& $rclonePath --config $configPath config
if ($LASTEXITCODE -ne 0) {
    throw "rclone config failed ($LASTEXITCODE)"
}

$remotes = @(& $rclonePath --config $configPath listremotes)
if ($LASTEXITCODE -ne 0) {
    throw "rclone listremotes failed ($LASTEXITCODE)"
}
if (@($remotes | Where-Object { $_ -ceq "highlightcraft-pcloud:" }).Count -eq 0) {
    throw "The required remote highlightcraft-pcloud: was not created"
}

Write-Host ""
Write-Host "OAuth config saved outside Git at: $configPath"
Write-Host "Next: docker compose run --rm --no-deps pcloud-admin doctor"
