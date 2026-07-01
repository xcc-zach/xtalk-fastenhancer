from __future__ import annotations

import argparse
import asyncio
import io
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

import librosa
import numpy as np
import onnxruntime
import soundfile as sf
import uvicorn
import wave
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import Response


PCMDType = Literal["int16", "float32"]
REALTIME_SAMPLE_RATE = 16000
REALTIME_FRAME_SAMPLES = 512
REALTIME_FRAME_BYTES = REALTIME_FRAME_SAMPLES * 2
REALTIME_ENCODING = "pcm_s16le"
REALTIME_CHANNELS = 1


@dataclass
class ServiceConfig:
    onnx_path: str
    sample_rate: int
    n_fft: int
    hop_size: int
    provider: str
    workers: int
    intra_op_threads: int
    inter_op_threads: int


class Wav2WavEnhancer:
    def __init__(self, config: ServiceConfig) -> None:
        if not os.path.exists(config.onnx_path):
            raise FileNotFoundError(
                f"ONNX model not found: {config.onnx_path}. Run install.sh or pass --onnx-path."
            )
        self.config = config
        sess_options = onnxruntime.SessionOptions()
        sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = config.intra_op_threads
        sess_options.inter_op_num_threads = config.inter_op_threads
        self.session = onnxruntime.InferenceSession(
            config.onnx_path,
            sess_options=sess_options,
            providers=[config.provider],
        )
        wav_input = None
        for item in self.session.get_inputs():
            if item.name == "wav_in":
                wav_input = item
                break
        if wav_input is None:
            raise ValueError("This service expects a wav2wav ONNX model with input 'wav_in'.")
        if len(wav_input.shape) < 2 or not isinstance(wav_input.shape[1], int):
            raise ValueError("The ONNX wav_in input must have a static [1, samples] shape.")
        self.input_size = wav_input.shape[1]
        wav_output = self.session.get_outputs()[0]
        if len(wav_output.shape) < 2 or not isinstance(wav_output.shape[1], int):
            raise ValueError("The ONNX wav_out output must have a static [1, samples] shape.")
        self.output_size = wav_output.shape[1]
        if config.hop_size not in (0, self.output_size):
            config.hop_size = self.output_size
        elif config.hop_size == 0:
            config.hop_size = self.output_size
        self.delay = max(0, config.n_fft - self.output_size)

    def _empty_cache(self) -> dict[str, np.ndarray]:
        cache: dict[str, np.ndarray] = {}
        for item in self.session.get_inputs():
            if not item.name.startswith("cache_in_"):
                continue
            shape = [1 if not isinstance(dim, int) or dim <= 0 else dim for dim in item.shape]
            cache[item.name] = np.zeros(shape, dtype=np.float32)
        return cache

    def enhance(self, wav: np.ndarray) -> np.ndarray:
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = np.mean(wav, axis=1)
        wav = np.clip(wav.reshape(1, -1), -1.0, 1.0)
        length = wav.shape[-1]
        if length == 0:
            raise ValueError("empty audio")

        padded = np.pad(wav, ((0, 0), (0, self.input_size + self.delay)), mode="constant")
        onnx_input = self._empty_cache()
        chunks: list[np.ndarray] = []
        num_chunks = math.ceil((length + self.delay) / self.output_size)

        for chunk_idx in range(num_chunks):
            idx = chunk_idx * self.output_size
            onnx_input["wav_in"] = padded[:, idx : idx + self.input_size]
            out = self.session.run(None, onnx_input)
            chunks.append(out[0][0])
            for cache_idx, cache_value in enumerate(out[1:]):
                onnx_input[f"cache_in_{cache_idx}"] = cache_value

        enhanced = np.concatenate(chunks, axis=0)
        enhanced = enhanced[self.delay : self.delay + length]
        return np.clip(enhanced, -1.0, 1.0).astype(np.float32, copy=False)

    def create_realtime_stream(self) -> "RealtimeStreamState":
        if self.config.sample_rate != REALTIME_SAMPLE_RATE:
            raise ValueError(
                f"realtime websocket requires service sample_rate={REALTIME_SAMPLE_RATE}; "
                f"got sample_rate={self.config.sample_rate}"
            )
        if self.input_size != REALTIME_FRAME_SAMPLES or self.output_size != REALTIME_FRAME_SAMPLES:
            raise ValueError(
                "realtime websocket requires ONNX wav_in and wav_out frame size of "
                f"{REALTIME_FRAME_SAMPLES} samples; got input_size={self.input_size}, "
                f"output_size={self.output_size}"
            )
        return RealtimeStreamState(self)


class RealtimeStreamState:
    def __init__(self, enhancer: Wav2WavEnhancer) -> None:
        self.enhancer = enhancer
        self.reset()

    def reset(self) -> None:
        self.onnx_input = self.enhancer._empty_cache()
        self.is_first_frame = True

    def process_frame(self, pcm: bytes) -> bytes:
        if len(pcm) != REALTIME_FRAME_BYTES:
            raise ValueError(f"expected exactly {REALTIME_FRAME_BYTES} bytes")
        wav = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        self.onnx_input["wav_in"] = wav.reshape(1, REALTIME_FRAME_SAMPLES)
        out = self.enhancer.session.run(None, self.onnx_input)
        for cache_idx, cache_value in enumerate(out[1:]):
            self.onnx_input[f"cache_in_{cache_idx}"] = cache_value
        self.is_first_frame = False
        enhanced = np.asarray(out[0][0], dtype=np.float32)
        return float_to_pcm16(enhanced)


def decode_audio_file(data: bytes, target_sr: int) -> np.ndarray:
    try:
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except Exception as exc:
        raise ValueError(f"failed to read audio file: {exc}") from exc
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return np.asarray(wav, dtype=np.float32)


def decode_pcm(data: bytes, dtype: PCMDType) -> np.ndarray:
    if dtype == "int16":
        if len(data) % 2:
            raise ValueError("int16 PCM payload length must be divisible by 2")
        return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    if len(data) % 4:
        raise ValueError("float32 PCM payload length must be divisible by 4")
    return np.frombuffer(data, dtype="<f4").astype(np.float32)


def float_to_pcm16(wav: np.ndarray) -> bytes:
    pcm = np.clip(wav, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    return pcm.tobytes()


def encode_wav(wav: np.ndarray, sample_rate: int) -> bytes:
    payload = float_to_pcm16(wav)
    bio = io.BytesIO()
    with wave.open(bio, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(payload)
    return bio.getvalue()


def create_app(config: ServiceConfig) -> FastAPI:
    enhancer = Wav2WavEnhancer(config)
    executor = ThreadPoolExecutor(max_workers=config.workers)
    semaphore = asyncio.Semaphore(config.workers)
    app = FastAPI(title="FastEnhancer Serving", version="1.0.0")

    async def run_enhance(wav: np.ndarray) -> np.ndarray:
        loop = asyncio.get_running_loop()
        async with semaphore:
            return await loop.run_in_executor(executor, enhancer.enhance, wav)

    async def run_realtime_frame(state: RealtimeStreamState, pcm: bytes) -> bytes:
        loop = asyncio.get_running_loop()
        async with semaphore:
            return await loop.run_in_executor(executor, state.process_frame, pcm)

    def realtime_spec() -> dict[str, object]:
        return {
            "sample_rate": REALTIME_SAMPLE_RATE,
            "frame_samples": REALTIME_FRAME_SAMPLES,
            "encoding": REALTIME_ENCODING,
            "channels": REALTIME_CHANNELS,
        }

    def validate_start_message(payload: dict[str, object]) -> str | None:
        expected = realtime_spec()
        for key, value in expected.items():
            if payload.get(key) != value:
                return f"{key} must be {value!r}"
        return None

    async def send_ws_error(websocket: WebSocket, code: str, message: str) -> None:
        await websocket.send_json({"type": "error", "code": code, "message": message})

    @app.on_event("shutdown")
    async def shutdown_executor() -> None:
        executor.shutdown(wait=False, cancel_futures=True)

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "onnx_path": config.onnx_path,
            "sample_rate": config.sample_rate,
            "n_fft": config.n_fft,
            "hop_size": config.hop_size,
            "input_size": enhancer.input_size,
            "output_size": enhancer.output_size,
            "delay": enhancer.delay,
            "provider": config.provider,
            "workers": config.workers,
        }

    @app.post("/v1/enhance")
    async def enhance_file(
        file: UploadFile = File(...),
        response_format: Literal["wav", "pcm"] = Query("wav"),
    ) -> Response:
        data = await file.read()
        try:
            wav = decode_audio_file(data, config.sample_rate)
            enhanced = await run_enhance(wav)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if response_format == "pcm":
            return Response(float_to_pcm16(enhanced), media_type="application/octet-stream")
        return Response(encode_wav(enhanced, config.sample_rate), media_type="audio/wav")

    @app.post("/v1/enhance/pcm")
    async def enhance_pcm(
        request: Request,
        input_dtype: PCMDType = Query("int16"),
        response_format: Literal["wav", "pcm"] = Query("pcm"),
    ) -> Response:
        data = await request.body()
        try:
            wav = decode_pcm(data, input_dtype)
            enhanced = await run_enhance(wav)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if response_format == "wav":
            return Response(encode_wav(enhanced, config.sample_rate), media_type="audio/wav")
        return Response(float_to_pcm16(enhanced), media_type="application/octet-stream")

    @app.websocket("/ws/fastenhancer/realtime")
    async def realtime_websocket(websocket: WebSocket) -> None:
        await websocket.accept()
        state: RealtimeStreamState | None = None
        started = False
        try:
            while True:
                msg = await websocket.receive()
                if msg["type"] == "websocket.disconnect":
                    break

                text = msg.get("text")
                if text is not None:
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        await send_ws_error(websocket, "invalid_json", "control message must be valid JSON")
                        continue
                    if not isinstance(payload, dict):
                        await send_ws_error(websocket, "invalid_control", "control message must be a JSON object")
                        continue

                    msg_type = payload.get("type")
                    if msg_type == "start":
                        if started:
                            await send_ws_error(websocket, "already_started", "stream already started")
                            continue
                        validation_error = validate_start_message(payload)
                        if validation_error is not None:
                            await send_ws_error(websocket, "invalid_start", validation_error)
                            continue
                        try:
                            state = enhancer.create_realtime_stream()
                        except ValueError as exc:
                            await send_ws_error(websocket, "unsupported_model_frame_size", str(exc))
                            continue
                        started = True
                        await websocket.send_json({"type": "start_ack", **realtime_spec()})
                    elif msg_type == "reset":
                        if state is None:
                            await send_ws_error(websocket, "not_started", "send start before reset")
                            continue
                        state.reset()
                        await websocket.send_json({"type": "reset_ack"})
                    elif msg_type == "flush":
                        if not started:
                            await send_ws_error(websocket, "not_started", "send start before flush")
                            continue
                        await websocket.send_json({"type": "flush_ack"})
                    elif msg_type == "close":
                        await websocket.close()
                        break
                    else:
                        await send_ws_error(websocket, "unknown_control", "unknown control message type")
                    continue

                pcm = msg.get("bytes")
                if pcm is None:
                    continue
                if state is None:
                    await send_ws_error(websocket, "not_started", "send start before binary audio frames")
                    continue
                if len(pcm) != REALTIME_FRAME_BYTES:
                    await send_ws_error(
                        websocket,
                        "invalid_frame_size",
                        f"expected exactly {REALTIME_FRAME_BYTES} bytes",
                    )
                    continue
                try:
                    enhanced = await run_realtime_frame(state, pcm)
                except Exception as exc:
                    await send_ws_error(websocket, "inference_failed", str(exc))
                    continue
                await websocket.send_bytes(enhanced)
        except WebSocketDisconnect:
            pass

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve FastEnhancer wav2wav ONNX inference.")
    parser.add_argument("--onnx-path", default=os.environ.get("ONNX_PATH", "onnx/fastenhancer_t.onnx"))
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--sample-rate", type=int, default=int(os.environ.get("SAMPLE_RATE", "16000")))
    parser.add_argument("--n-fft", type=int, default=int(os.environ.get("N_FFT", "512")))
    parser.add_argument("--hop-size", type=int, default=int(os.environ.get("HOP_SIZE", "0")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", str(os.cpu_count() or 4))))
    parser.add_argument("--intra-op-threads", type=int, default=int(os.environ.get("ORT_INTRA_OP_THREADS", "1")))
    parser.add_argument("--inter-op-threads", type=int, default=int(os.environ.get("ORT_INTER_OP_THREADS", "1")))
    parser.add_argument("--provider", default=os.environ.get("ORT_PROVIDER", "CPUExecutionProvider"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ServiceConfig(
        onnx_path=args.onnx_path,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_size=args.hop_size,
        provider=args.provider,
        workers=args.workers,
        intra_op_threads=args.intra_op_threads,
        inter_op_threads=args.inter_op_threads,
    )
    app = create_app(config)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
