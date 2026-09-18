"""Local, fixed-candidate editing experiments. No ranking, inference or DB writes."""
# Embedded review-page HTML/JS keeps its own line layout and Chinese punctuation.
# ruff: noqa: E501, RUF001

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import random
import subprocess
import time
from itertools import pairwise
from pathlib import Path

FPS = 30


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def run(command):
    return subprocess.run(command, check=True, capture_output=True, timeout=180)


def interval(value):
    if len(value) != 2 or not all(math.isfinite(x) for x in value) or not 0 <= value[0] < value[1]:
        raise ValueError("Invalid finite source interval")
    return value


def validate(plan):
    rows = plan["candidates"]
    ids = [c["candidate_id"] for c in rows]
    if ids != plan["frozen_order"] or len(set(ids)) != len(ids):
        raise ValueError("Candidate identity/order changed")
    for c in rows:
        for name in ("original_render", "context", "selected_render"):
            a, b = interval(c[name])
            if b > plan["source_duration"]:
                raise ValueError("Outside source")
        start, end = c["event"]["start_uncertainty"], c["event"]["end_uncertainty"]
        interval(start)
        interval(end)
        a, b = c["selected_render"]
        if not a <= start[0] < start[1] <= end[0] < end[1] <= b:
            raise ValueError("Selected render truncates observed event envelope")
        next_start = c["event"].get("next_start_lower_bound")
        if next_start is not None and b >= next_start:
            raise ValueError("Selected render includes next event")
        if c["event"]["provenance"] != "assistant_visual_inspection_not_human_truth":
            raise ValueError("Do not promote observations to human truth")
    by_id = {c["candidate_id"]: c for c in rows}
    total_seconds = sum(
        sum(v["range"][1] - v["range"][0] for v in p["variants"]) for p in plan["comparisons"]
    )
    total_seconds += sum(
        c["context"][1] - c["context"][0] + 4 * (c["selected_render"][1] - c["selected_render"][0])
        for c in rows
    )
    if (
        total_seconds > plan["max_output_seconds"]
        or 2 * len(plan["comparisons"]) + 2 * len(rows) + 3 > plan["max_media_files"]
    ):
        raise ValueError("Bounded render budget exceeded")
    for pair in plan["comparisons"]:
        if pair["candidate_id"] not in by_id or len(pair["variants"]) != 2:
            raise ValueError("Pair must retain one candidate")
        ranges = [interval(v["range"]) for v in pair["variants"]]
        if sum(ranges[0][i] != ranges[1][i] for i in (0, 1)) != 1:
            raise ValueError("Each comparison must change exactly one edge")
        c = by_id[pair["candidate_id"]]
        for a, b in ranges:
            if not c["context"][0] <= a < b <= c["context"][1]:
                raise ValueError("Comparison outside inspected context")
            if pair["stage"] == "rhythm" and not (
                a <= c["event"]["start_uncertainty"][0] and b >= c["event"]["end_uncertainty"][1]
            ):
                raise ValueError("Rhythm experiment cannot truncate event")
    return by_id


def quantize(bounds):
    a, b = interval(bounds)
    return math.floor(a * FPS + 1e-8) / FPS, math.ceil(b * FPS - 1e-8) / FPS


def verify_media(path, expected):
    probe = json.loads(
        run(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]
        ).stdout
    )
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    audio = next(s for s in probe["streams"] if s["codec_type"] == "audio")
    duration = float(video["duration"])
    if abs(duration - expected) > 1 / FPS + 0.002:
        raise ValueError("Rendered video duration mismatch")
    if abs(float(audio["duration"]) - duration) > 0.04:
        raise ValueError("A/V length mismatch")
    run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-threads",
            "2",
            "-i",
            str(path),
            "-map",
            "0:v",
            "-map",
            "0:a",
            "-f",
            "null",
            "-",
        ]
    )
    return {
        "sha256": sha(path),
        "duration": duration,
        "audio_duration": audio["duration"],
        "video_start": video.get("start_time"),
        "audio_start": audio.get("start_time"),
        "width": video["width"],
        "height": video["height"],
        "decoded_frames": int(video["nb_frames"]),
        "full_av_decode": "passed",
    }


def encode(source, target, bounds):
    a, b = quantize(bounds)
    length = b - a
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-threads",
        "2",
        "-ss",
        str(a),
        "-i",
        str(source),
        "-t",
        str(length),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-vf",
        "fps=30,scale=640:-2,setsar=1,format=yuv420p",
        "-af",
        "aresample=48000:first_pts=0",
        "-c:v",
        "libx264",
        "-threads",
        "2",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-ar",
        "48000",
        "-movflags",
        "+faststart",
        "-map_metadata",
        "-1",
        str(target),
    ]
    started = time.monotonic()
    run(command)
    return {
        "file": target.name,
        "requested_source_range": bounds,
        "render_source_range": [a, b],
        "command": command,
        "wall_seconds": time.monotonic() - started,
        **verify_media(target, length),
    }


def transition_guard(rows, overlap):
    if not math.isfinite(overlap) or not 0 <= overlap <= 0.5:
        raise ValueError("Only bounded short dissolve allowed")
    guards = []
    for left, right in pairwise(rows):
        _, end = quantize(left["selected_render"])
        start, _ = quantize(right["selected_render"])
        if end - overlap < left["event"]["end_uncertainty"][1] or (
            start + overlap > right["event"]["start_uncertainty"][0]
        ):
            raise ValueError("Transition overlaps an observed event or its uncertainty")
        guards.append(
            {
                "left": left["candidate_id"],
                "right": right["candidate_id"],
                "outgoing_source_overlap": [end - overlap, end],
                "incoming_source_overlap": [start, start + overlap],
                "observed_event_overlap": False,
                "human_transition_accepted": None,
            }
        )
    return guards


def reel(clips, rows, target, overlap=0):
    guards = transition_guard(rows, overlap)
    command = ["ffmpeg", "-nostdin", "-v", "error"]
    for clip in clips:
        command += ["-threads", "2", "-i", str(target.parent / clip["file"])]
    filters = []
    # AAC decoding includes padding beyond a clip's displayed video duration.
    # Concat uses the longest input stream: trim both streams to exact frame clocks
    # so padding cannot delay later cuts or remove the reel's last frames.
    lengths = [c["decoded_frames"] / FPS for c in clips]
    for i, clip in enumerate(clips):
        samples = round(clip["decoded_frames"] * 48000 / FPS)
        filters += [
            f"[{i}:v]trim=end_frame={clip['decoded_frames']},settb=AVTB,"
            f"setpts=N/({FPS}*TB)[v{i}]",
            f"[{i}:a]aresample=48000,apad,atrim=end_sample={samples},"
            f"asetpts=N/SR/TB[a{i}]",
        ]
    duration = sum(lengths) - overlap * (len(clips) - 1)
    if overlap:
        video, audio, elapsed = "v0", "a0", lengths[0]
        for i in range(1, len(clips)):
            filters += [
                f"[{video}][v{i}]xfade=transition=fade:duration={overlap}:"
                f"offset={elapsed - overlap}[x{i}]",
                f"[{audio}][a{i}]acrossfade=d={overlap}:c1=tri:c2=tri[y{i}]",
            ]
            elapsed += lengths[i] - overlap
            video, audio = f"x{i}", f"y{i}"
    else:
        filters += [
            "".join(f"[v{i}][a{i}]" for i in range(len(clips)))
            + f"concat=n={len(clips)}:v=1:a=1[v][a]"
        ]
        video, audio = "v", "a"
    command += [
        "-filter_complex_threads",
        "2",
        "-filter_complex",
        ";".join(filters),
        "-map",
        f"[{video}]",
        "-map",
        f"[{audio}]",
        "-t",
        str(duration),
        "-c:v",
        "libx264",
        "-threads",
        "2",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(target),
    ]
    run(command)
    timeline, elapsed = [], 0
    for row, clip, length in zip(rows, clips, lengths, strict=True):
        timeline.append(
            {
                "candidate_id": row["candidate_id"],
                "source_range": clip["render_source_range"],
                "reel_range": [elapsed, elapsed + length],
            }
        )
        elapsed += length - overlap
    return {
        "file": target.name,
        "timeline": timeline,
        "overlap_seconds": overlap,
        "transition_guards": guards,
        "command": command,
        "budget_overrun_seconds": max(0, duration - 55),
        **verify_media(target, duration),
    }


def page(packet, plan):
    body = """<!doctype html><html lang="zh-Hant"><meta charset="utf-8">
<title>固定回合比較</title><style>body{max-width:960px;margin:auto;padding:24px;
background:#17191d;color:#eee;font:18px/1.6 sans-serif}video{width:100%;max-height:470px}
select,button,textarea{font:inherit;margin:6px;padding:8px}article{border-top:1px solid #555;
padding:18px 0}a{color:#8de}summary{cursor:pointer}</style>
<h1>固定回合比較</h1><p>先判斷每分完整性，再評節奏。方法名稱隱藏；每組只變一個切點。
未判定請留白。這些影片尚未獲得真人驗收，較長或較短不代表較好。</p>
<label>播放順序 <select id="sequence"><option value="0">第一輪</option>
<option value="1">反向順序</option></select></label>
<p>評分保存在這個瀏覽器；請按底部匯出保存。原始標記不受影響。</p>"""
    for i, pair in enumerate(packet["pairs"]):
        body += f"<article><h2>比較 {i + 1} · {html.escape(pair['stage_label'])}</h2>"
        for media in pair["media"]:
            token = media["token"]
            body += f'<section data-token="{token}"><h3>版本 {token}</h3>'
            body += f'<video controls preload="none" src="{media["file"]}"></video>'
            for field, label, options in (
                ("complete", "完整性", ["完整", "不完整", "無法判定"]),
                ("rhythm", "觀看節奏", ["太急", "合適", "太拖"]),
            ):
                body += f'<label>{label}<select data-token="{token}" data-field="{field}">'
                body += '<option value="">待確認</option>'
                body += "".join(f"<option>{x}</option>" for x in options) + "</select></label>"
            body += f'<textarea data-token="{token}" data-field="note" placeholder="備註"></textarea></section>'
        body += "<details><summary>評完再看來源時間與差異</summary>"
        for media in pair["media"]:
            body += f"<p>{media['token']}：原片 {media['requested_source_range']} 秒。</p>"
        body += html.escape(pair["difference"]) + "</details></article>"
    body += "<details><summary>原片前後文／逐段核對</summary>"
    for c in packet["contexts"]:
        body += f"<p>{c['candidate_id']} · 原片 {c['requested_source_range']}</p>"
        body += f'<video controls preload="none" src="{c["file"]}"></video>'
    body += "</details><h2>六段採用切點核對</h2><p>完整性尚未由真人確認。請逐段觀看；以下判斷與節奏評分分開。</p>"
    for i, c in enumerate(packet["selected_clips"], 1):
        token = f"point-{i}"
        body += f'<h3>第 {i} 段</h3><video controls preload="none" src="{c["file"]}"></video>'
        body += f'<select class="gate-review" data-token="{token}" data-field="complete"><option value="">待確認</option>'
        body += "<option>完整</option><option>不完整</option><option>無法判定</option></select>"
    body += """<h2>合輯比較</h2><p>先完成六段完整性核對。以下同一批六段、同一組切點；
順序與轉場分開比較。來源事件區間是助手觀察，仍需真人確認。</p>
<p id="gate">六段皆評為完整後開啟合輯比較。</p>
<div id="reels" hidden>"""
    for r in packet["reels"]:
        body += f'<section class="reel"><h3>合輯 {r["token"]}</h3><video controls preload="none" src="{r["file"]}"></video></section>'
    for field, label, opts in (
        ("transition", "轉場（合輯 K／M）", ["K", "M", "無差異", "無法判定"]),
        ("order", "順序（合輯 K／N）", ["K", "N", "無差異", "無法判定"]),
    ):
        body += f'<label>{label}<select data-token="reel" data-field="{field}"><option value="">待確認</option>'
        body += "".join(f"<option>{x}</option>" for x in opts) + "</select></label>"
    body += '</div><p><button id="export">匯出人工回饋 JSON</button><span id="save"></span></p>'
    binding = {
        "version": plan["version"],
        "plan_sha256": packet["plan_sha256"],
        "source_sha256": plan["source_sha256"],
        "initial_pair_order": [[m["token"] for m in p["media"]] for p in packet["pairs"]],
        "media": [
            {"token": m["token"], "sha256": m["sha256"]}
            for p in packet["pairs"]
            for m in p["media"]
        ]
        + [{"token": r["token"], "sha256": r["sha256"]} for r in packet["reels"]],
    }
    binding["media"] += [
        {"token": f"point-{i}", "sha256": c["sha256"]}
        for i, c in enumerate(packet["selected_clips"], 1)
    ]
    body += (
        "<script>const binding="
        + json.dumps(binding)
        + """;
const key='rally-edit:'+binding.plan_sha256;let records=[];
let activeVideo=null;
for(const video of document.querySelectorAll('video'))video.onplay=()=>{
 if(activeVideo&&activeVideo!==video){activeVideo.pause();activeVideo.load()}
 activeVideo=video;
};
try{records=JSON.parse(localStorage.getItem(key)||'[]')}catch(e){}
for(const e of document.querySelectorAll('[data-field]')){
 const previous=records.filter(r=>r.token===e.dataset.token&&r.field===e.dataset.field).at(-1);
 if(previous)e.value=previous.value;
 e.onchange=()=>{records.push({token:e.dataset.token,field:e.dataset.field,value:e.value||null,
 actor:'human-browser-input',presentation_sequence:document.querySelector('#sequence').value,
 created_at:new Date().toISOString()});
 updateGate();
 try{localStorage.setItem(key,JSON.stringify(records));document.querySelector('#save').textContent='已暫存，請匯出'}
 catch(x){document.querySelector('#save').textContent='暫存失敗，請立即匯出'}};
}
document.querySelector('#sequence').onchange=()=>{
 for(const article of document.querySelectorAll('article')){
 const sections=[...article.querySelectorAll('section')];sections.reverse().forEach(s=>article.insertBefore(s,article.querySelector('details')));
 } records.push({field:'presentation_order',value:document.querySelector('#sequence').value,created_at:new Date().toISOString()});
 const reels=document.querySelector('#reels');[...reels.querySelectorAll('.reel')].reverse().forEach(s=>reels.insertBefore(s,reels.querySelector('label')));
};
function updateGate(){document.querySelector('#reels').hidden=![...document.querySelectorAll('.gate-review')].every(e=>e.value==='完整')}
updateGate();
document.querySelector('#export').onclick=()=>{const a=document.createElement('a');
 a.href=URL.createObjectURL(new Blob([JSON.stringify({...binding,records},null,2)],{type:'application/json'}));
 a.download='rally-edit-feedback.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),5000)};
</script></html>"""
    )
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    by_id = validate(plan)
    if sha(args.source) != plan["source_sha256"]:
        raise ValueError("Source SHA mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "plan.json", plan)
    packet = {
        "plan_sha256": sha(args.plan),
        "source_sha256": plan["source_sha256"],
        "code_sha256": sha(__file__),
        "human_status": "pending",
        "model_calls": 0,
        "pairs": [],
        "contexts": [],
        "reels": [],
        "selected_clips": [],
    }
    key = {}
    rng = random.Random(plan["blind_seed"])
    token_numbers = iter(rng.sample(range(100, 999), 2 * len(plan["comparisons"])))
    for i, pair in enumerate(plan["comparisons"]):
        media = []
        # Alternate treatment/control first; reverse round available without rerendering.
        variants = pair["variants"] if i % 2 == 0 else pair["variants"][::-1]
        for variant in variants:
            token = str(next(token_numbers))
            rendered = encode(args.source, args.output / f"{token}.mp4", variant["range"])
            rendered["token"] = token
            key[token] = {
                "candidate_id": pair["candidate_id"],
                "variant": variant["name"],
                "comparison": pair["id"],
            }
            media.append(rendered)
        packet["pairs"].append(
            {
                "stage_label": "事件與邊界" if pair["stage"] == "boundary" else "觀看節奏",
                "difference": pair["difference"],
                "media": media,
            }
        )
        write(args.output / "packet.partial.json", packet)
    for i, c in enumerate(plan["candidates"]):
        context = encode(args.source, args.output / f"context-{i + 1}.mp4", c["context"])
        context["candidate_id"] = c["candidate_id"]
        packet["contexts"].append(context)
        selected = encode(args.source, args.output / f"point-{i + 1}.mp4", c["selected_render"])
        selected["candidate_id"] = c["candidate_id"]
        packet["selected_clips"].append(selected)
    for token, order, overlap in (
        ("K", plan["frozen_order"], 0),
        ("M", plan["frozen_order"], 0.2),
        ("N", plan["presentation_order"], 0),
    ):
        if sorted(order) != sorted(plan["frozen_order"]):
            raise ValueError("Presentation order changed candidate membership")
        clips = {c["candidate_id"]: c for c in packet["selected_clips"]}
        result = reel(
            [clips[k] for k in order],
            [by_id[k] for k in order],
            args.output / f"{token}.mp4",
            overlap,
        )
        result["token"] = token
        packet["reels"].append(result)
    write(args.output / "method-key.json", key)
    write(args.output / "packet.json", packet)
    write(
        args.output / "human-feedback.json",
        {"status": "pending", "records": [], "plan_sha256": packet["plan_sha256"]},
    )
    (args.output / "index.html").write_text(page(packet, plan))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "pairs": len(packet["pairs"]),
                "reel_seconds": [r["duration"] for r in packet["reels"]],
            }
        )
    )


if __name__ == "__main__":
    main()
