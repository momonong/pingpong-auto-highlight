from __future__ import annotations

import argparse
import socket
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import qrcode
import uvicorn

from pingpong_highlight.archive import (
    ArchiveCandidate,
    ArchiveError,
    PCloudArchiver,
    RclonePCloudBackend,
    discover_archive_candidates,
)
from pingpong_highlight.candidate_evaluation import (
    CandidateEvaluationError,
    freeze_active_candidate_evaluation,
)
from pingpong_highlight.candidate_run import (
    CandidateRunError,
    run_candidate_analysis,
)
from pingpong_highlight.candidate_scoring import score_candidate_run
from pingpong_highlight.config import Settings
from pingpong_highlight.db import Database, StateConflict
from pingpong_highlight.media_work import archive_work_lock, media_work_lock
from pingpong_highlight.pipeline.media import has_nvdec, has_nvenc, probe_media, require_media_tools
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


def _service_url(settings: Settings, address: str) -> str:
    base_url = (
        settings.public_url.rstrip("/")
        if settings.public_url
        else (f"http://{address}:{settings.port}")
    )
    return f"{base_url}/#token={quote(settings.upload_token)}"


def _serve(args: argparse.Namespace) -> int:
    settings = Settings.from_env(data_dir=args.data_dir, host=args.host, port=args.port)
    require_media_tools()
    address = _lan_address() if settings.host in {"0.0.0.0", "::"} else settings.host
    url = _service_url(settings, address)
    print("\n桌球剪輯服務已準備好。手機與電腦需在同一個區域網路。")
    print(f"手機網址：{url}\n")
    if not args.no_qr:
        _print_qr(url)
        print()
    print("1. 保持這個視窗與電腦開啟。")
    print("2. 手機掃描 QR code，從相簿選片或貼上公開 Google Drive 連結。")
    print("3. 上傳完成後可關閉手機頁面，電腦會繼續處理。")
    print("4. 回到同一網址即可預覽、下載或分享完成的 MP4。")
    print(f"\n資料目錄：{settings.data_dir}")
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
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = (args.output or settings.outputs_dir / f"manual-{source.stem}-{timestamp}").resolve()
    processor = HighlightProcessor(settings)
    last_stage = ""

    def progress(value: float, stage: str) -> None:
        nonlocal last_stage
        if stage != last_stage or value >= 1.0:
            print(f"[{value:6.1%}] {stage}")
            last_stage = stage

    with media_work_lock(settings.data_dir):
        result = processor.run(source, output, progress)
    count = result["summary"]["point_count"]
    print(f"完成：儲存 {count} 個精彩球素材，輸出於 {output}")
    return 0


def _rebuild_library(args: argparse.Namespace) -> int:
    settings = Settings.from_env(data_dir=args.data_dir)
    database = Database(settings.database_path)
    job = database.get_job(args.job_id)
    if job is None or job.status != "completed":
        print(f"找不到已完成的來源工作：{args.job_id}", file=sys.stderr)
        return 2
    upload = database.get_upload(job.upload_id)
    if upload is None or not upload.path.is_file():
        print("來源影片不存在，無法重建素材庫。", file=sys.stderr)
        return 2

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    relative_output = Path("clip-sets") / f"highlight-library-v3-{timestamp}"
    output = settings.outputs_dir / job.id / relative_output
    processor = HighlightProcessor(settings)
    last_stage = ""

    def progress(value: float, stage: str) -> None:
        nonlocal last_stage
        if stage != last_stage or value >= 1.0:
            print(f"[{value:6.1%}] {stage}")
            last_stage = stage

    with media_work_lock(settings.data_dir):
        result = processor.run(
            upload.path,
            output,
            progress,
            source_name=upload.filename,
        )
    count = result["summary"]["point_count"]
    if count == 0:
        print("這次沒有找到合格素材，保留目前素材庫版本。")
        return 0
    activated = database.activate_highlight_result(
        job.id,
        result,
        file_prefix=relative_output.as_posix(),
        library_version=str(result["algorithm_version"]),
    )
    print(f"完成：已啟用 {activated} 個素材，舊版片段仍保留但不再顯示。")
    return 0


def _doctor(_args: argparse.Namespace) -> int:
    try:
        require_media_tools()
    except RuntimeError as exc:
        print(f"媒體工具：失敗（{exc}）")
        return 1
    print("FFmpeg / ffprobe：可用")
    print(f"NVIDIA NVDEC：{'可用' if has_nvdec() else '未偵測到，影片解碼會使用 CPU'}")
    print(f"NVIDIA NVENC：{'可用' if has_nvenc() else '未偵測到，影片編碼會使用 CPU'}")
    return 0


def _probe(args: argparse.Namespace) -> int:
    info = probe_media(args.video.expanduser().resolve())
    print(
        f"{info.width}x{info.height}, {info.fps:.3f} fps, {info.duration:.2f}s, "
        f"video={info.video_codec}, audio={info.audio_codec or 'none'}, rotation={info.rotation}°"
    )
    return 0


def _resolve_review_database(
    data_dir: Path,
    requested: Path | None,
) -> Path:
    if requested is not None:
        return requested.expanduser().resolve()
    matches = sorted(data_dir.glob("state.training-baseline-*.sqlite3"))
    if len(matches) != 1:
        raise CandidateEvaluationError(
            "Specify --review-database; automatic discovery requires exactly one "
            "state.training-baseline-*.sqlite3 file"
        )
    return matches[0].resolve()


def _freeze_active_evaluation(args: argparse.Namespace) -> int:
    settings = Settings.from_env(data_dir=args.data_dir)
    try:
        review_database = _resolve_review_database(
            settings.data_dir,
            args.review_database,
        )
        destination, metrics = freeze_active_candidate_evaluation(
            settings.data_dir,
            review_database=review_database,
            run_id=args.run_id,
            output_root=args.output_root,
            progress=print,
        )
    except CandidateEvaluationError as exc:
        print(f"無法建立 frozen evaluation：{exc}", file=sys.stderr)
        return 2
    strict = metrics["aggregate"]["strict_candidate_recall"]
    print(
        f"Strict candidate recall：{strict['hits']}/{strict['total']} "
        f"({strict['micro_recall']:.2%})"
    )
    print(f"GO/STOP：{metrics['gate']['decision']}")
    print(f"報告：{destination / 'report.md'}")
    print("這是 legacy diagnostic freeze，舊 artifact 缺少生成當下的完整 receipt。")
    return 2


def _run_candidate_evaluation(args: argparse.Namespace) -> int:
    settings = Settings.from_env(data_dir=args.data_dir)
    try:
        destination = run_candidate_analysis(
            settings,
            dataset_path=args.dataset,
            run_id=args.run_id,
            output_root=args.output_root,
            require_gpu=not args.allow_cpu,
            allow_dirty=args.allow_dirty,
            progress=print,
        )
    except CandidateRunError as exc:
        print(f"無法執行 candidate-only analysis：{exc}", file=sys.stderr)
        return 2
    print(f"Candidate-only run：{destination}")
    print("沒有輸出 MP4，也沒有修改 active 素材庫或 runtime database。")
    return 0


def _score_candidate_evaluation(args: argparse.Namespace) -> int:
    settings = Settings.from_env(data_dir=args.data_dir)
    output_root = args.output_root or (settings.data_dir / "evaluations" / "candidate-recall")
    try:
        destination, metrics = score_candidate_run(
            dataset_path=args.dataset,
            candidate_run=args.candidate_run,
            run_id=args.run_id,
            output_root=output_root,
        )
    except CandidateEvaluationError as exc:
        print(f"無法評分 candidate run：{exc}", file=sys.stderr)
        return 2
    strict = metrics["aggregate"]["strict_candidate_recall"]
    print(
        f"Strict candidate recall：{strict['hits']}/{strict['total']} "
        f"({strict['micro_recall']:.2%})"
    )
    print(f"GO/STOP：{metrics['gate']['decision']}")
    print(f"報告：{destination / 'report.md'}")
    return 0 if metrics["gate"]["threshold_met"] else 3


def _human_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.1f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024
    raise AssertionError("unreachable")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _pcloud_context(
    args: argparse.Namespace,
) -> tuple[Settings, Database, RclonePCloudBackend]:
    settings = Settings.from_env(data_dir=args.data_dir)
    database = Database(settings.database_path)
    return settings, database, RclonePCloudBackend(settings)


def _pcloud_doctor(args: argparse.Namespace) -> int:
    _settings, _database, backend = _pcloud_context(args)
    try:
        result = backend.doctor()
    except ArchiveError as exc:
        print(f"pCloud 設定尚未完成：{exc}", file=sys.stderr)
        return 1
    print(f"rclone：{result.rclone_version}")
    print(
        f"pCloud remote：{result.remote} ({result.region}, {result.hostname}, "
        f"account={result.account_id}, root={result.root_folder_id})"
    )
    print("連線與唯讀目錄檢查：通過")
    return 0


def _pcloud_bootstrap(args: argparse.Namespace) -> int:
    settings, database, backend = _pcloud_context(args)
    try:
        doctor = backend.doctor()
        PCloudArchiver(
            settings,
            database,
            backend,
            remote_region=doctor.region,
            remote_hostname=doctor.hostname,
            remote_account_id=doctor.account_id,
            remote_root_folder_id=doctor.root_folder_id,
        ).validate_catalog_target()
        paths = backend.bootstrap(dry_run=args.dry_run)
    except (ArchiveError, StateConflict) as exc:
        print(f"無法建立 pCloud 目錄：{exc}", file=sys.stderr)
        return 1
    mode = "預覽" if args.dry_run else "完成"
    print(f"pCloud {doctor.region} archive 目錄 {mode}：")
    for path in paths:
        print(f"- {path}/")
    return 0


def _filter_archive_candidates(
    candidates: list[ArchiveCandidate],
    kind: str,
) -> list[ArchiveCandidate]:
    if kind == "all":
        return candidates
    expected = {
        "original": "original",
        "highlight": "highlight_clip",
        "compilation": "compilation",
    }[kind]
    return [candidate for candidate in candidates if candidate.media_kind == expected]


def _archive_plan_lines(
    candidates: list[ArchiveCandidate],
    database: Database,
) -> tuple[list[str], int, int]:
    statuses = {
        (record.owner_type, record.owner_id): record.archive_state
        for record in database.list_storage_objects()
    }
    lines: list[str] = []
    total_bytes = 0
    problems = 0
    for candidate in candidates:
        exists = candidate.local_path.is_file()
        size = candidate.local_path.stat().st_size if exists else candidate.byte_size
        state = statuses.get((candidate.owner_type, candidate.owner_id), "not-registered")
        if not exists:
            state = "local-missing"
            problems += 1
        elif size <= 0:
            state = "local-empty"
            problems += 1
        elif candidate.byte_size > 0 and size != candidate.byte_size:
            state = "catalog-size-mismatch"
            problems += 1
        total_bytes += size
        lines.append(
            f"[{candidate.media_kind}] {state} | {_human_bytes(size)} | {candidate.remote_path}"
        )
    return lines, total_bytes, problems


def _pcloud_plan(args: argparse.Namespace) -> int:
    settings, database, _backend = _pcloud_context(args)
    try:
        candidates = _filter_archive_candidates(
            discover_archive_candidates(settings, database),
            args.kind,
        )
    except ArchiveError as exc:
        print(f"無法建立 archive plan：{exc}", file=sys.stderr)
        return 1
    lines, total_bytes, problems = _archive_plan_lines(candidates, database)
    for line in lines:
        print(line)
    print(
        f"共 {len(candidates)} 個檔案，約 {_human_bytes(total_bytes)}，本機有 {problems} 個問題。"
    )
    print("plan 不會建立遠端檔案或登記 storage object。")
    return 0 if problems == 0 else 1


def _pcloud_archive(args: argparse.Namespace) -> int:
    settings, database, backend = _pcloud_context(args)
    try:
        candidates = _filter_archive_candidates(
            discover_archive_candidates(settings, database),
            args.kind,
        )
    except ArchiveError as exc:
        print(f"無法建立 archive plan：{exc}", file=sys.stderr)
        return 1

    existing = {
        (record.owner_type, record.owner_id): record for record in database.list_storage_objects()
    }
    candidates = [
        candidate
        for candidate in candidates
        if existing.get((candidate.owner_type, candidate.owner_id)) is None
        or existing[(candidate.owner_type, candidate.owner_id)].archive_state != "verified"
    ]
    if args.limit is not None:
        candidates = candidates[: args.limit]

    if not args.execute:
        lines, total_bytes, problems = _archive_plan_lines(candidates, database)
        for line in lines:
            print(line)
        print(
            f"將處理 {len(candidates)} 個檔案，約 {_human_bytes(total_bytes)}，"
            f"本機有 {problems} 個問題。"
        )
        print("尚未執行。確認後在同一指令加上 --execute。")
        return 0 if problems == 0 else 1

    if not candidates:
        print("沒有尚待歸檔的檔案。")
        return 0

    try:
        doctor = backend.doctor()
    except ArchiveError as exc:
        print(f"pCloud 設定尚未完成：{exc}", file=sys.stderr)
        return 1
    print(f"使用 pCloud {doctor.region}，本次處理 {len(candidates)} 個檔案。")
    archiver = PCloudArchiver(
        settings,
        database,
        backend,
        remote_region=doctor.region,
        remote_hostname=doctor.hostname,
        remote_account_id=doctor.account_id,
        remote_root_folder_id=doctor.root_folder_id,
    )
    try:
        archiver.validate_catalog_target()
    except StateConflict as exc:
        print(f"pCloud archive target 與既有 catalog 不一致：{exc}", file=sys.stderr)
        return 1
    completed = 0
    with archive_work_lock(settings.data_dir):
        for index, candidate in enumerate(candidates, start=1):
            print(f"[{index}/{len(candidates)}] 雜湊、上傳並驗證：{candidate.local_relative_path}")
            try:
                result = archiver.archive(candidate)
            except (ArchiveError, OSError, StateConflict) as exc:
                print(f"歸檔失敗，原本本機檔案未變更：{exc}", file=sys.stderr)
                return 1
            completed += 1
            print(f"已驗證：{result.remote_path}")
    print(f"完成 {completed} 個 pCloud archive。沒有刪除任何本機或 Drive 檔案。")
    return 0


def _pcloud_verify(args: argparse.Namespace) -> int:
    settings, database, backend = _pcloud_context(args)
    expected_kind = {
        "original": "original",
        "highlight": "highlight_clip",
        "compilation": "compilation",
    }.get(args.kind)
    records = [
        record
        for record in database.list_storage_objects()
        if record.verified_at is not None
        and (expected_kind is None or record.media_kind == expected_kind)
    ]
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        print("沒有可重新驗證的 pCloud archive object。")
        return 0
    try:
        doctor = backend.doctor()
    except ArchiveError as exc:
        print(f"pCloud 設定尚未完成：{exc}", file=sys.stderr)
        return 1
    archiver = PCloudArchiver(
        settings,
        database,
        backend,
        remote_region=doctor.region,
        remote_hostname=doctor.hostname,
        remote_account_id=doctor.account_id,
        remote_root_folder_id=doctor.root_folder_id,
    )
    with archive_work_lock(settings.data_dir):
        for index, record in enumerate(records, start=1):
            print(f"[{index}/{len(records)}] 重新驗證：{record.remote_path}")
            try:
                archiver.verify_record(record)
            except (ArchiveError, OSError, StateConflict) as exc:
                print(f"remote verification 失敗：{exc}", file=sys.stderr)
                return 1
    print(f"完成 {len(records)} 個 pCloud archive remote verification。")
    return 0


def _pcloud_status(args: argparse.Namespace) -> int:
    _settings, database, _backend = _pcloud_context(args)
    records = database.list_storage_objects()
    if not records:
        print("尚未登記任何 pCloud archive object。")
        return 0
    totals: dict[str, int] = {}
    for record in records:
        totals[record.archive_state] = totals.get(record.archive_state, 0) + 1
        print(
            f"[{record.media_kind}] {record.archive_state} | "
            f"attempts={record.attempts} | {record.remote_path}"
        )
        if record.last_error:
            print(f"  error: {record.last_error}")
    summary = ", ".join(f"{state}={count}" for state, count in sorted(totals.items()))
    print(f"共 {len(records)} 個：{summary}")
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

    rebuild = subparsers.add_parser(
        "rebuild-library",
        help="以目前模型重新擷取既有來源的精彩球素材",
    )
    rebuild.add_argument("job_id")
    rebuild.add_argument("--data-dir", type=Path, default=None)
    rebuild.set_defaults(handler=_rebuild_library)

    probe = subparsers.add_parser("probe", help="檢查手機影片的媒體資訊")
    probe.add_argument("video", type=Path)
    probe.set_defaults(handler=_probe)

    doctor = subparsers.add_parser("doctor", help="檢查 FFmpeg 與 GPU 編解碼能力")
    doctor.set_defaults(handler=_doctor)

    evaluation = subparsers.add_parser(
        "evaluation",
        help="凍結並量測精彩球候選能力",
    )
    evaluation_commands = evaluation.add_subparsers(
        dest="evaluation_command",
        required=True,
    )
    freeze_active = evaluation_commands.add_parser(
        "freeze-active",
        help="只讀匯入目前 active candidates，建立不可覆寫的診斷基線",
    )
    freeze_active.add_argument("--data-dir", type=Path, default=None)
    freeze_active.add_argument("--review-database", type=Path, default=None)
    freeze_active.add_argument("--output-root", type=Path, default=None)
    freeze_active.add_argument("--run-id", default=None)
    freeze_active.set_defaults(handler=_freeze_active_evaluation)
    run_candidates = evaluation_commands.add_parser(
        "run-candidates",
        help="以 GPU 重跑候選與 signals，不輸出 MP4 或修改素材庫",
    )
    run_candidates.add_argument("--dataset", type=Path, required=True)
    run_candidates.add_argument("--run-id", required=True)
    run_candidates.add_argument("--data-dir", type=Path, default=None)
    run_candidates.add_argument("--output-root", type=Path, default=None)
    run_candidates.add_argument(
        "--allow-cpu",
        action="store_true",
        help="NVDEC 不可用時允許 CPU 解碼",
    )
    run_candidates.add_argument(
        "--allow-dirty",
        action="store_true",
        help="只供診斷，正式 baseline 應保持 clean worktree",
    )
    run_candidates.set_defaults(handler=_run_candidate_evaluation)
    score_candidates = evaluation_commands.add_parser(
        "score-candidates",
        help="驗證 immutable receipts 並以 frozen 規則量測 candidate recall",
    )
    score_candidates.add_argument("--dataset", type=Path, required=True)
    score_candidates.add_argument("--candidate-run", type=Path, required=True)
    score_candidates.add_argument("--run-id", required=True)
    score_candidates.add_argument("--data-dir", type=Path, default=None)
    score_candidates.add_argument("--output-root", type=Path, default=None)
    score_candidates.set_defaults(handler=_score_candidate_evaluation)

    pcloud = subparsers.add_parser("pcloud", help="管理 pCloud 長期影片 archive")
    pcloud.add_argument("--data-dir", type=Path, default=None)
    pcloud_commands = pcloud.add_subparsers(dest="pcloud_command", required=True)

    pcloud_doctor = pcloud_commands.add_parser(
        "doctor",
        help="檢查 rclone OAuth、帳號區域與唯讀連線",
    )
    pcloud_doctor.set_defaults(handler=_pcloud_doctor)

    pcloud_bootstrap = pcloud_commands.add_parser(
        "bootstrap",
        help="建立固定且可重複執行的 pCloud 目錄",
    )
    pcloud_bootstrap.add_argument("--dry-run", action="store_true")
    pcloud_bootstrap.set_defaults(handler=_pcloud_bootstrap)

    for command_name, handler, help_text in (
        ("plan", _pcloud_plan, "預覽本機影片到 pCloud 的名稱與路徑"),
        ("archive", _pcloud_archive, "複製、驗證並登記影片 archive"),
        ("verify", _pcloud_verify, "重新驗證已歸檔影片與 manifest"),
    ):
        command = pcloud_commands.add_parser(command_name, help=help_text)
        command.add_argument(
            "--kind",
            choices=("all", "original", "highlight", "compilation"),
            default="all",
        )
        if command_name in {"archive", "verify"}:
            command.add_argument("--limit", type=_positive_int, default=None)
        if command_name == "archive":
            command.add_argument("--execute", action="store_true")
        command.set_defaults(handler=handler)

    pcloud_status = pcloud_commands.add_parser(
        "status",
        help="列出已登記 archive object 與驗證狀態",
    )
    pcloud_status.set_defaults(handler=_pcloud_status)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.handler(args))
