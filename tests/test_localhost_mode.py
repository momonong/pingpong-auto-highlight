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
    assert (
        'Stop-TunnelService -Overlay "compose.cloudflare.yaml" -Service "cloudflared"'
        in script
    )
    assert 'Where-Object { $_.HostIp -ne "127.0.0.1" }' in script
    assert "a tunnel container is still running" in script
    assert "Docker Compose could not stop the $Service tunnel service" in script
    assert "Stop-ServicesFailClosed" in script
    assert "Automatic fail-closed cleanup could not be verified" in script
    assert '$arguments += @("ps", "-a", "-q", $Service)' in script
    assert "if ($idExitCode -ne 0)" in script
    assert "if ($inspectionExitCode -ne 0)" in script
    assert 'data\\local-access-url.txt' in script
    assert 'data\\remote-access-url.txt' in script
    assert 'data\\ngrok-access-url.txt' in script
    assert "#token=$([uri]::EscapeDataString($token))" in script
    assert "start-localhost.ps1" in wrapper
    assert "--wait-timeout $TimeoutSeconds" in script
    assert script.count("Assert-NoActiveWork") == 3
    assert "Clear-StaleRemoteUrls" in script
    assert "docker compose @composeFiles build pingpong-highlight" in script
    assert "up -d --no-build --no-deps --wait" in script

    switch_sequence = script.split("$transitionStarted = $true", maxsplit=1)[1]
    assert switch_sequence.index('Service "ngrok"') < switch_sequence.index(
        'Service "cloudflared"'
    )
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
