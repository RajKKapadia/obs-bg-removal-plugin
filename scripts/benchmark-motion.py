#!/usr/bin/env python3
"""Reproduce the isolated OBS motion benchmark and optionally add paced NVENC load.

Explicitly label whether the input is raw webcam footage or an existing composite.
The latter is useful for performance testing only, never matte-quality assessment.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--input-kind", choices=("raw", "composite"), required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "data/models/rvm_mobilenetv3_fp32.onnx")
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--clip-seconds", type=int, default=10)
    parser.add_argument("--source-fps", type=int, choices=(30, 60), default=30)
    parser.add_argument("--obs-fps", type=int, choices=(30, 60), default=60)
    parser.add_argument("--input-limit", type=int, choices=(640, 1280, 1920), default=1920)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--encode-load", action="store_true")
    args = parser.parse_args()
    if not 10 <= args.seconds <= 3600 or not 1 <= args.clip_seconds <= 60:
        parser.error("Use 10..3600 measurement seconds and 1..60 clip seconds")
    args.input = args.input.resolve(strict=True)
    args.model = args.model.resolve(strict=True)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Use a new or empty output directory")
    frames = output / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    probe = subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(args.input)], text=True)
    metadata = {"input_kind": args.input_kind, "input": str(args.input), "input_sha256": digest(args.input),
                "model": str(args.model), "model_sha256": digest(args.model), "settings": vars(args),
                "probe": json.loads(probe), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "limitations": "Isolated OBS replay; excludes camera buffering and display latency. Composite inputs cannot establish matte quality. Encoding load is a separate NVENC process, not the user's OBS recording scene."}
    (output / "metadata.json").write_text(json.dumps(metadata, default=str, indent=2) + "\n")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(args.input), "-t", str(args.clip_seconds),
                    "-vf", f"fps={args.source_fps}", "-compression_level", "1", str(frames / "%06d.png")], check=True)
    env = os.environ.copy()
    for key in tuple(env):
        if key.startswith("RMBG_TEST_"):
            del env[key]
    env.update(RMBG_TEST_SEQUENCE=str(frames), RMBG_TEST_MATCH_VIDEO="1", RMBG_TEST_RVM_SIZE=str(args.input_limit),
               RMBG_TEST_OBS_FPS=str(args.obs_fps), RMBG_TEST_SOURCE_FPS=str(args.source_fps),
               RMBG_TEST_WIDTH="1920", RMBG_TEST_HEIGHT="1080", RMBG_TEST_ACCEPTANCE="1")
    if args.foreground:
        env["RMBG_TEST_FOREGROUND"] = "1"
    encoder = None
    process = None
    with (output / "obs.log").open("w") as log, (output / "encoding.log").open("w") as encoding:
        try:
            if args.encode_load:
                # Regenerate CFR timestamps after looping to avoid the original
                # MKV's short-duration/end-of-loop timestamp discontinuities.
                encoder = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-re", "-stream_loop", "-1",
                    "-i", str(args.input), "-an", "-vf", f"fps={args.obs_fps},setpts=N/({args.obs_fps}*TB)",
                    "-fps_mode", "cfr", "-r", str(args.obs_fps), "-c:v", "h264_nvenc", "-preset", "p5", "-rc", "constqp",
                    "-qp", "20", "-bf", "2", "-rc-lookahead", "8", "-f", "null", "-"], stdout=encoding, stderr=encoding)
                time.sleep(2)
                if encoder.poll() is not None:
                    raise RuntimeError("NVENC load failed; see encoding.log")
            command = [str(ROOT / "build/obs-rmbg-smoke"), str(ROOT / "build/obs-rmbg.so"), str(ROOT / "data"), str(args.model),
                       str(frames / "000001.png"), str(output / "result.png"), args.device, "30", "0", str(args.seconds)]
            process = subprocess.Popen(command, env=env, stdout=log, stderr=log)
            while process.poll() is None:
                time.sleep(1)
                if encoder and encoder.poll() is not None:
                    process.terminate()
                    process.wait(timeout=15)
                    raise RuntimeError("Encoding load exited during the benchmark")
            exit_code = process.returncode
        finally:
            if process and process.poll() is None:
                process.terminate()
                process.wait(timeout=15)
            if encoder and encoder.poll() is None:
                encoder.terminate()
                encoder.wait(timeout=15)
    timing_path = output / "result.png.timings.csv"
    if not timing_path.exists():
        raise RuntimeError(f"OBS exited {exit_code} before measurement; inspect {output / 'obs.log'}")
    rows = list(csv.DictReader(timing_path.open()))
    steady = [r for r in rows if float(r["elapsed_s"]) >= 10]
    if not steady:
        raise RuntimeError(f"No steady-state timing samples; inspect {output / 'obs.log'}")
    metrics = {key: {"min": min(float(r[key]) for r in steady), "max": max(float(r[key]) for r in steady),
                     "mean": statistics.mean(float(r[key]) for r in steady)}
               for key in ("mask_rate", "pair_rate", "age_p95_ms", "processing_p95_ms", "queue_p95_ms", "rss_kib", "source_underruns", "cache_misses")}
    result = {"exit_code": exit_code, "steady_window_samples": len(steady), "metrics": metrics,
              "acceptance": exit_code == 0, "input_kind": args.input_kind,
              "note": "age_p95_ms contains overlapping five-second percentiles, not a whole-run percentile. Review RSS for plateau/growth; the executable's numeric gate does not automatically assess memory slope."}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
