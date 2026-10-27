"""Verify decoded endpoints and source-audio correspondence of an edit review pack."""

import argparse
import json
from pathlib import Path

import numpy as np

from pingpong_highlight.edit_comparison import run, sha, write


def pixels(path, time):
    return np.frombuffer(
        run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-threads",
                "2",
                "-ss",
                str(max(0, time)),
                "-i",
                str(path),
                "-frames:v",
                "1",
                "-vf",
                "scale=320:-2,format=gray",
                "-f",
                "rawvideo",
                "-",
            ]
        ).stdout,
        dtype=np.uint8,
    ).astype(float)


def pcm(path, start, duration):
    return np.frombuffer(
        run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-threads",
                "2",
                "-ss",
                str(start),
                "-i",
                str(path),
                "-t",
                str(duration),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "8000",
                "-f",
                "f32le",
                "-",
            ]
        ).stdout,
        dtype=np.float32,
    )


def frame_samples(path, indices):
    expression = "+".join(f"eq(n,{n})" for n in indices)
    data = run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-threads",
            "2",
            "-i",
            str(path),
            "-vf",
            f"select='{expression}',scale=320:180,format=gray",
            "-fps_mode",
            "passthrough",
            "-frames:v",
            str(len(indices)),
            "-f",
            "rawvideo",
            "-",
        ]
    ).stdout
    return np.frombuffer(data, dtype=np.uint8).reshape(len(indices), 180, 320).astype(float)


def direct_reel_endpoints(directory, reel, clips):
    by_id = {c["candidate_id"]: c for c in clips}
    ordered = [by_id[r["candidate_id"]] for r in reel["timeline"]]
    indices, offset = [], 0
    for clip in ordered:
        indices.extend([offset, offset + clip["decoded_frames"] - 1])
        offset += clip["decoded_frames"]
    actual = frame_samples(directory / reel["file"], indices)
    results = []
    for i, clip in enumerate(ordered):
        expected = frame_samples(directory / clip["file"], [0, clip["decoded_frames"] - 1])
        errors = np.sqrt(np.mean((actual[2 * i : 2 * i + 2] - expected) ** 2, axis=(1, 2)))
        results.append(
            {
                "candidate_id": clip["candidate_id"],
                "reel_frame_indices": indices[2 * i : 2 * i + 2],
                "rmse": errors.tolist(),
                "passed": bool(np.max(errors) < 5),
            }
        )
    return {"segments": results, "passed": all(r["passed"] for r in results)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = json.loads(args.packet.read_text())
    if sha(args.source) != packet["source_sha256"]:
        raise ValueError("Source changed")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "source_sha256": packet["source_sha256"],
        "clips": [],
        "reels": [],
        "human_listening": None,
        "human_visual_acceptance": None,
        "semantics": "Decoded signal/clock fidelity, not perceptual acceptance",
    }
    clips = packet["selected_clips"] + [m for p in packet["pairs"] for m in p["media"]]
    for i, clip in enumerate(clips):
        path = args.packet.parent / clip["file"]
        a, b = clip["render_source_range"]
        ends = []
        for output_time, source_time in [(0, a), (clip["duration"] - 1 / 30, b - 1 / 30)]:
            actual = pixels(path, output_time)
            errors = []
            for shift in [-1 / 30, 0, 1 / 30]:
                expected = pixels(args.source, source_time + shift)
                errors.append(float(np.sqrt(np.mean((actual - expected) ** 2))))
            best = int(np.argmin(errors))
            ends.append(
                {
                    "output_time": output_time,
                    "planned_source_time": source_time,
                    "best_shift_seconds": [-1 / 30, 0, 1 / 30][best],
                    "gray_rmse": errors[best],
                    "threshold": 12,
                }
            )
        expected_audio = pcm(args.source, a, b - a)
        actual_audio = pcm(path, 0, clip["duration"])
        n = min(len(expected_audio), len(actual_audio))
        correlation = float(np.corrcoef(expected_audio[:n], actual_audio[:n])[0, 1])
        rms = float(np.sqrt(np.mean(actual_audio**2)))
        row = {
            "file": clip["file"],
            "sha256_matches": sha(path) == clip["sha256"],
            "endpoints": ends,
            "audio_source_correlation": correlation,
            "audio_rms": rms,
            "audio_peak": float(np.max(np.abs(actual_audio))),
            "audio_decoded_samples": len(actual_audio),
        }
        row["passed"] = (
            row["sha256_matches"]
            and all(e["gray_rmse"] < 12 for e in ends)
            and correlation > 0.9
            and rms > 1e-5
        )
        report["clips"].append(row)
        if i < 6:
            run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-threads",
                    "2",
                    "-i",
                    str(path),
                    "-vf",
                    f"select='eq(n,0)+eq(n,{clip['decoded_frames'] - 1})',scale=640:-2,tile=2x1",
                    "-frames:v",
                    "1",
                    str(args.output / f"point-{i + 1}-endpoints.jpg"),
                ]
            )
    for reel in packet["reels"]:
        audio = pcm(args.packet.parent / reel["file"], 0, reel["duration"])
        seams = []
        for row in reel["timeline"][1:]:
            time = row["reel_range"][0]
            index = round(time * 8000)
            window = audio[max(0, index - 800) : index + 2400]
            seams.append(
                {
                    "time": time,
                    "rms": float(np.sqrt(np.mean(window**2))),
                    "peak": float(np.max(np.abs(window))),
                    "max_adjacent_sample_jump": float(np.max(np.abs(np.diff(window)))),
                }
            )
        report["reels"].append(
            {
                "file": reel["file"],
                "audio_seams": seams,
                "clipped_sample_fraction": float(np.mean(np.abs(audio) >= 1)),
                "direct_reel_endpoints": direct_reel_endpoints(
                    args.packet.parent, reel, packet["selected_clips"]
                )
                if reel.get("overlap_seconds") == 0
                else None,
            }
        )
    report["passed"] = all(c["passed"] for c in report["clips"]) and all(
        r["direct_reel_endpoints"] is None or r["direct_reel_endpoints"]["passed"]
        for r in report["reels"]
    )
    write(args.output / "media-signal-validation.json", report)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "clips": len(clips),
                "minimum_audio_correlation": min(
                    c["audio_source_correlation"] for c in report["clips"]
                ),
            }
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
