This README provides one-click FastEnhancer service deployment and client usage instructions. [Original README](README.original.md)

## Requirements

- Linux/macOS
- Python 3.9+
- CPU ONNXRuntime by default. Set `INSTALL_GPU=1` to install `onnxruntime-gpu`.

## Installation

```bash
chmod +x install.sh start.sh
bash install.sh
```

The installer creates `.venv`, installs serving dependencies, and tries to download the default wav2wav ONNX model to `onnx/fastenhancer_t.onnx` from GitHub Releases. If `pip` fails, it automatically retries with proxy environment variables unset. You can also set `MODEL_URL=... bash install.sh` to use a specific model URL.

## Start the Service

```bash
bash start.sh --port 8000 --onnx-path onnx/fastenhancer_t.onnx
```

Common options:

- `--host`: bind address, default `0.0.0.0`
- `--port`: bind port, default `8000`
- `--onnx-path`: wav2wav ONNX model path, default `onnx/fastenhancer_t.onnx`
- `--sample-rate`: service sample rate, default `16000`
- `--n-fft`: FFT size, default `512`
- `--hop-size`: streaming hop size, default `0` means infer from the ONNX input shape; you can also set it manually
- `--workers`: concurrent worker threads, default CPU core count

## Client Requests

### Health Check

```bash
curl http://localhost:8000/health
```

### Upload an Audio File and Return WAV

```bash
curl -X POST "http://localhost:8000/v1/enhance?response_format=wav" \
  -F "file=@onnx/p232_001-009.wav" \
  --output enhanced.wav
```

### Upload an Audio File and Return Raw PCM

```bash
curl -X POST "http://localhost:8000/v1/enhance?response_format=pcm" \
  -F "file=@onnx/p232_001-009.wav" \
  --output enhanced.pcm
```

### Upload Raw PCM and Return Raw PCM

The request body is mono little-endian PCM. The default input dtype is `int16`, and the sample rate must match the service `--sample-rate`.

```bash
curl -X POST "http://localhost:8000/v1/enhance/pcm?input_dtype=int16&response_format=pcm" \
  --data-binary "@input_16k_s16le.pcm" \
  -H "Content-Type: application/octet-stream" \
  --output enhanced_16k_s16le.pcm
```

`input_dtype` also supports `float32`. `response_format` supports `pcm` or `wav`.

## Performance Testing

### Run

Start the service first, then run:

```bash
source .venv/bin/activate
python serving/performance_testing.py --concurrency 1 4 16 64 256
```

The benchmark uses `onnx/p232_001-009.wav` first. If the local sample is unavailable, it tries to find and download an audio file from `https://huggingface.co/datasets/xcczach/sample-data`.

### Results

Validation environment: Ubuntu Linux 5.15.0-141-generic, Python 3.13.9, CPU `INTEL(R) XEON(R) GOLD 6530` (128 vCPUs), GPU `2 x NVIDIA GeForce RTX 4090` (the service used `CPUExecutionProvider` for this run), ONNX model `onnx/fastenhancer_t.onnx`, service option `--workers 2`. The benchmark audio was a 1-second 16 kHz mono WAV sample.

| concurrency | requests | avg latency (s) | P50 (s) | P95 (s) | throughput (samples/s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.103 | 0.103 | 0.103 | 0.497 |
| 4 | 4 | 0.156 | 0.156 | 0.205 | 13.412 |
| 16 | 16 | 0.427 | 0.415 | 0.692 | 19.861 |
| 64 | 64 | 1.268 | 1.319 | 2.089 | 24.807 |
| 256 | 256 | 5.551 | 5.188 | 9.487 | 24.128 |
