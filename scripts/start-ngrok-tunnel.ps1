[CmdletBinding()]
param(
    [switch]$CpuOnly,
    [switch]$UsePublishedImage,
    [switch]$ReplaceAuthtoken,
    [ValidateRange(30, 300)]
    [int]$TimeoutSeconds = 120,
    [ValidateRange(1024, 65535)]
    [int]$InspectPort = 4040
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$tokenPath = Join-Path $repoRoot "data\.ngrok-authtoken"
$agentConfigPath = Join-Path $repoRoot "data\.ngrok-agent.yml"
$composeFiles = @(
    "-f", (Join-Path $repoRoot "compose.yaml")
)
if ($UsePublishedImage) {
    $composeFiles += @("-f", (Join-Path $repoRoot "compose.release.yaml"))
}
if ($CpuOnly) {
    $composeFiles += @("-f", (Join-Path $repoRoot "compose.cpu.yaml"))
}
$composeFiles += @("-f", (Join-Path $repoRoot "compose.ngrok.yaml"))

function Get-NativeCommandOutput {
    param([Parameter(Mandatory = $true)][scriptblock]$Command)

    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        return (& $Command 2>&1 | Out-String)
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
}

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

function ConvertFrom-SecureValue {
    param([Parameter(Mandatory = $true)][Security.SecureString]$Value)

    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($Value)
    try {
        return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    }
}

function Get-NgrokAuthtoken {
    if (-not $ReplaceAuthtoken -and $env:NGROK_AUTHTOKEN) {
        $environmentToken = $env:NGROK_AUTHTOKEN.Trim()
        if ($environmentToken) {
            return $environmentToken
        }
    }

    if (-not $ReplaceAuthtoken -and (Test-Path -LiteralPath $tokenPath)) {
        $savedToken = (Get-Content -Raw -LiteralPath $tokenPath).Trim()
        if ($savedToken) {
            return $savedToken
        }
    }

    Write-Host "ngrok needs the authtoken from your own free account." -ForegroundColor Yellow
    Write-Host "Open https://dashboard.ngrok.com/get-started/your-authtoken and copy only the token."
    $secureToken = Read-Host "Paste the ngrok authtoken (input is hidden)" -AsSecureString
    $newToken = (ConvertFrom-SecureValue -Value $secureToken).Trim()
    if (-not $newToken) {
        throw "The ngrok authtoken cannot be empty."
    }

    $dataDir = Split-Path -Parent $tokenPath
    New-Item -ItemType Directory -Path $dataDir -Force | Out-Null
    [IO.File]::WriteAllText($tokenPath, $newToken, [Text.UTF8Encoding]::new($false))
    Write-Host "Saved the authtoken locally in data/.ngrok-authtoken (this directory is ignored by Git)."
    return $newToken
}

function Write-NgrokAgentConfig {
    param([Parameter(Mandatory = $true)][string]$Authtoken)

    $escapedAuthtoken = $Authtoken.Replace("'", "''")
    $config = @"
version: 3
agent:
    authtoken: '$escapedAuthtoken'
    web_addr: 0.0.0.0:4040
"@
    [IO.File]::WriteAllText(
        $agentConfigPath,
        $config + [Environment]::NewLine,
        [Text.UTF8Encoding]::new($false)
    )
}

function Get-NgrokPublicUrl {
    param([Parameter(Mandatory = $true)][int]$Port)

    try {
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/tunnels" -TimeoutSec 3
        $httpsTunnel = @($response.tunnels) |
            Where-Object { $_.public_url -and $_.public_url.StartsWith("https://") } |
            Select-Object -First 1
        if ($httpsTunnel) {
            return $httpsTunnel.public_url.TrimEnd("/")
        }
    }
    catch {
        # The local agent API starts shortly after the container itself.
    }

    return $null
}

function Show-NgrokFailure {
    param([Parameter(Mandatory = $true)][string]$SensitiveToken)

    $redactedLogs = $null
    $logs = Get-NativeCommandOutput {
        docker compose @composeFiles logs --no-color --tail 40 ngrok
    }
    if ($logs) {
        $redactedLogs = $logs.Replace($SensitiveToken, "[REDACTED]")
    }
    if ($logs -match "ERR_NGROK_105|authentication failed|authtoken") {
        Write-Warning "ngrok rejected the saved authtoken. Run the launcher again with -ReplaceAuthtoken."
        return
    }
    if ($redactedLogs) {
        Write-Host $redactedLogs.TrimEnd()
    }
}

$previousInspectPort = $env:NGROK_INSPECT_PORT
$previousIgnoreOrphans = $env:COMPOSE_IGNORE_ORPHANS
$locationPushed = $false
try {
    $dockerInfoExitCode = Invoke-NativeCommandSilently { docker info }
    if ($dockerInfoExitCode -ne 0) {
        throw "Docker Desktop is not running. Start Docker Desktop and try again."
    }

    $authtoken = Get-NgrokAuthtoken
    Write-NgrokAgentConfig -Authtoken $authtoken
    $env:NGROK_INSPECT_PORT = $InspectPort.ToString()
    $env:COMPOSE_IGNORE_ORPHANS = "true"

    Push-Location $repoRoot
    $locationPushed = $true

    if ($UsePublishedImage) {
        & docker compose @composeFiles pull pingpong-highlight ngrok
        if ($LASTEXITCODE -ne 0) {
            throw "Docker Compose could not pull the published RallyCut image from Docker Hub."
        }
        & docker compose @composeFiles up -d pingpong-highlight ngrok
    }
    else {
        & docker compose @composeFiles up -d --build pingpong-highlight ngrok
    }
    if ($LASTEXITCODE -ne 0) {
        if (-not $CpuOnly) {
            throw (
                "Docker Compose could not start the GPU highlight service and ngrok. " +
                "Confirm NVIDIA Container Toolkit is available; use -CpuOnly only as a temporary fallback."
            )
        }
        throw "Docker Compose could not start the highlight service and ngrok."
    }

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $tunnelUrl = $null
    $authenticationFailed = $false
    while ((Get-Date) -lt $deadline -and -not $tunnelUrl) {
        $candidateUrl = Get-NgrokPublicUrl -Port $InspectPort
        if ($candidateUrl) {
            try {
                $health = Invoke-RestMethod `
                    -Uri "$candidateUrl/api/health" `
                    -Headers @{ "ngrok-skip-browser-warning" = "rallycut" } `
                    -TimeoutSec 10
                if ($health.status -eq "ok") {
                    $tunnelUrl = $candidateUrl
                    break
                }
            }
            catch {
                # The public hostname can appear before its edge route is ready.
            }
        }

        $containerId = (& docker compose @composeFiles ps -q ngrok 2>$null | Out-String).Trim()
        if ($containerId) {
            $recentLogs = Get-NativeCommandOutput {
                docker logs --tail 20 $containerId
            }
            if ($recentLogs -match "ERR_NGROK_105|authentication failed") {
                $authenticationFailed = $true
                break
            }
            $containerState = (& docker inspect --format "{{.State.Status}}" $containerId 2>$null | Out-String).Trim()
            if ($containerState -eq "exited" -or $containerState -eq "dead") {
                break
            }
        }
        Start-Sleep -Seconds 2
    }

    if (-not $tunnelUrl) {
        Show-NgrokFailure -SensitiveToken $authtoken
        $null = Invoke-NativeCommandSilently {
            docker compose @composeFiles rm -s -f ngrok
        }
        if ($authenticationFailed) {
            throw "ngrok rejected the saved authtoken. Run again with -ReplaceAuthtoken."
        }
        throw (
            "ngrok did not become reachable within $TimeoutSeconds seconds. " +
            "If port $InspectPort is already in use, run again with -InspectPort 4041. " +
            "If the public page reports 'Network bandwidth exceeded', use " +
            ".\scripts\start-localhost.ps1 -UsePublishedImage instead."
        )
    }

    $uploadTokenPath = Join-Path $repoRoot "data\.upload-token"
    if (-not (Test-Path -LiteralPath $uploadTokenPath)) {
        throw "The RallyCut access token was not created at $uploadTokenPath."
    }
    $uploadToken = (Get-Content -Raw -LiteralPath $uploadTokenPath).Trim()
    if (-not $uploadToken) {
        throw "The RallyCut access token is empty."
    }

    $phoneUrl = "$tunnelUrl/#token=$([uri]::EscapeDataString($uploadToken))"
    $ngrokUrlPath = Join-Path $repoRoot "data\ngrok-access-url.txt"
    $latestUrlPath = Join-Path $repoRoot "data\remote-access-url.txt"
    $urlFileContent = $phoneUrl + [Environment]::NewLine
    [IO.File]::WriteAllText($ngrokUrlPath, $urlFileContent, [Text.UTF8Encoding]::new($false))
    [IO.File]::WriteAllText($latestUrlPath, $urlFileContent, [Text.UTF8Encoding]::new($false))

    $cloudflareFiles = @(
        "-f", (Join-Path $repoRoot "compose.yaml"),
        "-f", (Join-Path $repoRoot "compose.cloudflare.yaml")
    )
    $null = Invoke-NativeCommandSilently {
        docker compose @cloudflareFiles stop cloudflared
    }

    Write-Host ""
    Write-Host "ngrok tunnel is ready." -ForegroundColor Green
    Write-Host "Open this token-protected link on your phone:"
    Write-Output $phoneUrl
    Write-Host ""
    Write-Host "If ngrok first shows 'Visit Site', tap it and then open the complete link above again."
    Write-Warning "Anyone with the complete link can use this service. Do not share it."
    Write-Warning "Keep Docker Desktop and this computer awake while using RallyCut."
    Write-Host "The latest link is also saved to $latestUrlPath"
}
finally {
    if ($locationPushed) {
        Pop-Location
    }
    if ($null -eq $previousInspectPort) {
        Remove-Item Env:NGROK_INSPECT_PORT -ErrorAction SilentlyContinue
    }
    else {
        $env:NGROK_INSPECT_PORT = $previousInspectPort
    }
    if ($null -eq $previousIgnoreOrphans) {
        Remove-Item Env:COMPOSE_IGNORE_ORPHANS -ErrorAction SilentlyContinue
    }
    else {
        $env:COMPOSE_IGNORE_ORPHANS = $previousIgnoreOrphans
    }
}

exit 0
