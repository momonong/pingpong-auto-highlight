from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_base_compose_owns_gpu_and_clip_context_defaults() -> None:
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")

    assert "gpus: all" in compose
    assert "PINGPONG_CLIP_PRE_ROLL_SECONDS" in compose
    assert "PINGPONG_CLIP_POST_ROLL_SECONDS" in compose
    assert not (ROOT / "compose.gpu.yaml").exists()


def test_localhost_compose_override_is_loopback_only() -> None:
    compose = (ROOT / "compose.localhost.yaml").read_text(encoding="utf-8")

    assert "ports: !override" in compose
    assert '"127.0.0.1:${PINGPONG_PORT:-8000}:8000"' in compose
    assert 'PINGPONG_PUBLIC_URL: "http://127.0.0.1:${PINGPONG_PORT:-8000}"' in compose
    assert "ngrok:" not in compose
    assert "cloudflared:" not in compose


def test_localhost_launcher_stops_tunnels_and_fails_closed() -> None:
    script = (ROOT / "scripts" / "start-localhost.ps1").read_text(encoding="utf-8")
    wrapper = (ROOT / "scripts" / "start-localhost.sh").read_text(encoding="utf-8")

    assert "Assert-NoActiveWork" in script
    assert 'Stop-TunnelService -Overlay "compose.ngrok.yaml" -Service "ngrok"' in script
    assert 'Stop-TunnelService -Overlay "compose.cloudflare.yaml" -Service "cloudflared"' in script
    assert 'Where-Object { $_.HostIp -ne "127.0.0.1" }' in script
    assert "a tunnel container is still running" in script
    assert "Docker Compose could not stop the $Service tunnel service" in script
    assert "Stop-ServicesFailClosed" in script
    assert "Automatic fail-closed cleanup could not be verified" in script
    assert '$arguments += @("ps", "-a", "-q", $Service)' in script
    assert "if ($idExitCode -ne 0)" in script
    assert "if ($inspectionExitCode -ne 0)" in script
    assert "Get-HighlightCraftDataRoot" in script
    assert 'Join-Path $dataRoot "local-access-url.txt"' in script
    assert 'Join-Path $dataRoot "remote-access-url.txt"' in script
    assert 'Join-Path $dataRoot "ngrok-access-url.txt"' in script
    assert "#token=" not in script
    assert "start-localhost.ps1" in wrapper
    assert "--wait-timeout $TimeoutSeconds" in script
    assert script.count("Assert-NoActiveWork") == 3
    assert "Clear-StaleRemoteUrls" in script
    assert "docker compose @composeFiles build pingpong-highlight" in script
    assert "up -d --no-build --no-deps --wait" in script

    switch_sequence = script.split("$transitionStarted = $true", maxsplit=1)[1]
    assert switch_sequence.index('Service "ngrok"') < switch_sequence.index('Service "cloudflared"')
    assert switch_sequence.index('Service "cloudflared"') < switch_sequence.index(
        "Clear-StaleRemoteUrls"
    )
    assert switch_sequence.index("Clear-StaleRemoteUrls") < switch_sequence.index(
        "docker compose @composeFiles pull pingpong-highlight"
    )
    assert switch_sequence.index("docker compose @composeFiles pull pingpong-highlight") < (
        switch_sequence.rindex("Assert-NoActiveWork")
    )
    assert switch_sequence.index("docker compose @composeFiles build pingpong-highlight") < (
        switch_sequence.rindex("Assert-NoActiveWork")
    )
    assert switch_sequence.rindex("Assert-NoActiveWork") < switch_sequence.index(
        "$applicationMutationStarted = $true"
    )
    assert switch_sequence.index("$applicationMutationStarted = $true") < (
        switch_sequence.index("up -d --no-build --no-deps --wait")
    )


def test_deployment_launchers_check_global_work_before_recreating_app() -> None:
    common = (ROOT / "scripts" / "deployment-common.ps1").read_text(encoding="utf-8")
    localhost = (ROOT / "scripts" / "start-localhost.ps1").read_text(encoding="utf-8")
    ngrok = (ROOT / "scripts" / "start-ngrok-tunnel.ps1").read_text(encoding="utf-8")
    cloudflare = (ROOT / "scripts" / "start-cloudflare-tunnel.ps1").read_text(encoding="utf-8")

    assert "/api/maintenance/active-work" in common
    assert common.index("/api/drive-imports?scope=all") < common.index("/api/uploads?scope=all")
    assert common.index("/api/uploads?scope=all") < common.index("/api/jobs?scope=all")
    assert '-Path $legacyTokenPath -Description "legacy access token"' in common
    assert "Assert-HighlightCraftNoActiveWork" in localhost

    for script in (ngrok, cloudflare):
        assert script.count("Assert-HighlightCraftNoActiveWork") == 2
        assert script.rindex("Assert-HighlightCraftNoActiveWork") < script.index("up -d --no-build")
        assert "up -d --build" not in script


def test_tunnel_launchers_stop_the_other_tunnel_after_health_check() -> None:
    ngrok = (ROOT / "scripts" / "start-ngrok-tunnel.ps1").read_text(encoding="utf-8")
    cloudflare = (ROOT / "scripts" / "start-cloudflare-tunnel.ps1").read_text(encoding="utf-8")

    assert (
        "Stop-HighlightCraftOverlayService -RepositoryRoot $repoRoot `\n"
        '        -Overlay "compose.cloudflare.yaml" -Service "cloudflared"'
    ) in ngrok
    assert (
        "Stop-HighlightCraftOverlayService -RepositoryRoot $repoRoot `\n"
        '        -Overlay "compose.ngrok.yaml" -Service "ngrok"'
    ) in cloudflare
    assert ngrok.index("Stop-HighlightCraftOverlayService") < ngrok.index(
        '$phoneUrl = "$tunnelUrl/"'
    )
    assert cloudflare.index("Stop-HighlightCraftOverlayService") < cloudflare.index(
        '$phoneUrl = "$tunnelUrl/"'
    )


def test_readme_documents_localhost_fallback() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "## 只在這台電腦使用（localhost，不走 tunnel）" in readme
    assert "./scripts/start-localhost.sh -UsePublishedImage" in readme
    assert ".\\scripts\\start-localhost.ps1 -UsePublishedImage" in readme
    assert "cat ./data/local-access-url.txt" in readme
    assert "127.0.0.1` 只能由這台電腦開啟" in readme
    assert "NVIDIA NVDEC" in readme
    assert "NVIDIA NVENC" in readme
    assert "Network bandwidth exceeded" in readme
    assert "所有管理指令都必須重複啟動時的同一組 `-f` 檔案" in readme
    assert (
        "docker compose -f compose.yaml -f compose.deploy.yaml -f compose.ngrok.yaml down" in readme
    )
