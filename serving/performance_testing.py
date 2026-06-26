from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Iterable

import httpx


HF_TREE_API = "https://huggingface.co/api/datasets/xcczach/sample-data/tree/main?recursive=1"
HF_RESOLVE = "https://huggingface.co/datasets/xcczach/sample-data/resolve/main"
LOCAL_SAMPLE = Path("onnx/p232_001-009.wav")


def download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=30) as response:
        path.write_bytes(response.read())


def find_hf_sample() -> str | None:
    try:
        with urllib.request.urlopen(HF_TREE_API, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    for item in payload:
        name = item.get("path", "")
        if name.lower().endswith((".wav", ".flac", ".mp3")):
            return f"{HF_RESOLVE}/{name}"
    return None


def ensure_audio(audio_path: str | None, audio_url: str | None) -> Path:
    if audio_path:
        path = Path(audio_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"audio file not found: {audio_path}")
    if LOCAL_SAMPLE.exists():
        return LOCAL_SAMPLE

    url = audio_url or find_hf_sample()
    if not url:
        raise RuntimeError("could not locate a Hugging Face sample audio file")
    path = Path("serving/sample_audio") / Path(url.split("?")[0]).name
    if not path.exists():
        download_file(url, path)
    return path


async def one_request(client: httpx.AsyncClient, url: str, audio: bytes, filename: str) -> float:
    start = time.perf_counter()
    files = {"file": (filename, audio, "audio/wav")}
    response = await client.post(f"{url}/v1/enhance", files=files, params={"response_format": "wav"})
    response.raise_for_status()
    if not response.content:
        raise RuntimeError("empty response")
    return time.perf_counter() - start


async def run_level(url: str, audio_path: Path, concurrency: int, requests: int, timeout: float) -> dict[str, float]:
    audio = audio_path.read_bytes()
    total = max(concurrency, requests)
    latencies: list[float] = []
    started = time.perf_counter()
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        pending = set()
        completed = 0
        while completed < total:
            while len(pending) < concurrency and completed + len(pending) < total:
                pending.add(asyncio.create_task(one_request(client, url, audio, audio_path.name)))
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                latencies.append(task.result())
                completed += 1
    elapsed = time.perf_counter() - started
    return {
        "concurrency": float(concurrency),
        "requests": float(total),
        "avg_latency_s": statistics.fmean(latencies),
        "p50_latency_s": statistics.median(latencies),
        "p95_latency_s": sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)],
        "throughput_samples_s": total / elapsed,
    }


def print_table(rows: Iterable[dict[str, float]]) -> None:
    print("| concurrency | requests | avg latency (s) | p50 (s) | p95 (s) | throughput (samples/s) |")
    print("|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {int(row['concurrency'])} | {int(row['requests'])} | "
            f"{row['avg_latency_s']:.3f} | {row['p50_latency_s']:.3f} | "
            f"{row['p95_latency_s']:.3f} | {row['throughput_samples_s']:.3f} |"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the FastEnhancer HTTP service.")
    parser.add_argument("--url", default=os.environ.get("SERVICE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--audio-path")
    parser.add_argument("--audio-url")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 4, 16, 64, 256])
    parser.add_argument("--requests-per-level", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    audio_path = ensure_audio(args.audio_path, args.audio_url)
    print(f"Environment: OS={platform.platform()}, Python={platform.python_version()}")
    print(f"Service URL: {args.url}")
    print(f"Audio: {audio_path}")
    rows = []
    for concurrency in args.concurrency:
        row = await run_level(args.url.rstrip("/"), audio_path, concurrency, args.requests_per_level, args.timeout)
        rows.append(row)
        print_table([row])
    print("\nSummary")
    print_table(rows)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
