from __future__ import annotations

import argparse
import socket
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import qrcode
import uvicorn

from pingpong_highlight.config import Settings
from pingpong_highlight.pipeline.media import has_nvenc, probe_media, require_media_tools
from pingpong_highlight.pipeline.processor import HighlightProcessor
from pingpong_highlight.web import create_app


def _lan_address() -> str:
    connection = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        connection.connect(("1.1.1.1", 80))
        return str(connection.getsockname()[0])
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"
    finally:
        connection.close()


def _print_qr(url: str) -> None:
    qr = qrcode.QRCode(border=1)
    qr.add_data(url)
    qr.make(fit=True)
    qr.print_ascii(invert=True)


def _serve(args: argparse.Namespace) -> int:
    settings = Settings.from_env(data_dir=args.data_dir, host=args.host, port=args.port)
    require_media_tools()
    address = _lan_address() if settings.host in {"0.0.0.0", "::"} else settings.host
    url = f"http://{address}:{settings.port}/?token={quote(settings.upload_token)}"
    print("\n桌球精華服務已準備好。手機與電腦需在同一個區域網路。")
    print(f"手機網址：{url}\n")
    if not args.no_qr:
        _print_qr(url)
        print()
    print(f"資料目錄：{settings.data_dir}")
    print("按 Ctrl+C 停止服務。Windows 第一次執行時請允許私人網路存取。\n")
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=args.log_level,
    )
    return 0


def _analyze(args: argparse.Namespace) -> int:
    source = args.video.expanduser().resolve()
    if not source.is_file():
        print(f"找不到影片：{source}", file=sys.stderr)
        return 2
    settings = Settings.from_env(data_dir=args.data_dir)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = (args.output or settings.outputs_dir / f"manual-{source.stem}-{timestamp}").resolve()
    processor = HighlightProcessor(settings)
    last_stage = ""

    def progress(value: float, stage: str) -> None:
        nonlocal last_stage
        if stage != last_stage or value >= 1.0:
            print(f"[{value:6.1%}] {stage}")
            last_stage = stage

    result = processor.run(source, output, progress)
    count = result["summary"]["point_count"]
    print(f"完成：剪出 {count} 個精彩得分，直式集錦輸出於 {output}")
    return 0


def _doctor(_args: argparse.Namespace) -> int:
    try:
        require_media_tools()
    except RuntimeError as exc:
        print(f"媒體工具：失敗（{exc}）")
        return 1
    print("FFmpeg / ffprobe：可用")
    print(f"NVIDIA NVENC：{'可用' if has_nvenc() else '未偵測到，會使用 CPU'}")
    return 0


def _probe(args: argparse.Namespace) -> int:
    info = probe_media(args.video.expanduser().resolve())
    print(
        f"{info.width}x{info.height}, {info.fps:.3f} fps, {info.duration:.2f}s, "
        f"video={info.video_codec}, audio={info.audio_codec or 'none'}, rotation={info.rotation}°"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="桌球影片自動精華工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="啟動供手機上傳的區域網路服務")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--data-dir", type=Path, default=None)
    serve.add_argument("--no-qr", action="store_true")
    serve.add_argument("--log-level", default="info")
    serve.set_defaults(handler=_serve)

    analyze = subparsers.add_parser("analyze", help="直接分析電腦上的影片")
    analyze.add_argument("video", type=Path)
    analyze.add_argument("--output", type=Path, default=None)
    analyze.add_argument("--data-dir", type=Path, default=None)
    analyze.set_defaults(handler=_analyze)

    probe = subparsers.add_parser("probe", help="檢查手機影片的媒體資訊")
    probe.add_argument("video", type=Path)
    probe.set_defaults(handler=_probe)

    doctor = subparsers.add_parser("doctor", help="檢查 FFmpeg 與 GPU 編碼能力")
    doctor.set_defaults(handler=_doctor)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.handler(args))
