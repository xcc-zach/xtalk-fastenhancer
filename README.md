本 README 为 FastEnhancer 服务一键部署和客户端使用说明。[原始 README](README.original.md)

## 环境要求

- Linux/macOS
- Python 3.9+
- 默认使用 CPU ONNXRuntime；如需 GPU 版 ONNXRuntime，可设置 `INSTALL_GPU=1`

## 安装

```bash
chmod +x install.sh start.sh
bash install.sh
```

默认下载 FastEnhancer_T。也可以用 `--model` 指定 T/B/S/M/L 五种 wav2wav ONNX 模型：

```bash
bash install.sh --model t
bash install.sh --model b
bash install.sh --model s
bash install.sh --model m
bash install.sh --model l
```

安装脚本会创建 `.venv`、安装服务依赖，并尝试从 GitHub Releases 下载模型到 `onnx/fastenhancer_<model>.onnx`。如果 `pip` 下载失败，脚本会自动清除代理环境变量后重试。也可以通过 `MODEL_URL=... bash install.sh` 指定模型下载地址。

## 服务启动

```bash
bash start.sh --port 8000 --onnx-path onnx/fastenhancer_t.onnx
```

常用参数：

- `--host`：监听地址，默认 `0.0.0.0`
- `--port`：监听端口，默认 `8000`
- `--onnx-path`：wav2wav ONNX 模型路径，默认 `onnx/fastenhancer_t.onnx`
- `--sample-rate`：服务采样率，默认 `16000`
- `--n-fft`：FFT size，默认 `512`
- `--hop-size`：流式 hop size，默认 `0` 表示从 ONNX 输入形状自动推断；也可以手动指定
- `--workers`：并发工作线程数，默认 CPU 核数

## 客户端请求

### 健康检查

```bash
curl http://localhost:8000/health
```

### 上传音频文件，返回 WAV

```bash
curl -X POST "http://localhost:8000/v1/enhance?response_format=wav" \
  -F "file=@onnx/p232_001-009.wav" \
  --output enhanced.wav
```

### 上传音频文件，返回裸 PCM

```bash
curl -X POST "http://localhost:8000/v1/enhance?response_format=pcm" \
  -F "file=@onnx/p232_001-009.wav" \
  --output enhanced.pcm
```

### 上传裸 PCM，返回裸 PCM

请求体为单声道 little-endian PCM，默认 `int16`，采样率需与服务 `--sample-rate` 一致。

```bash
curl -X POST "http://localhost:8000/v1/enhance/pcm?input_dtype=int16&response_format=pcm" \
  --data-binary "@input_16k_s16le.pcm" \
  -H "Content-Type: application/octet-stream" \
  --output enhanced_16k_s16le.pcm
```

`input_dtype` 也支持 `float32`。`response_format` 支持 `pcm` 或 `wav`。

## 压测

### 运行

先启动服务，然后运行：

```bash
source .venv/bin/activate
python serving/performance_testing.py --concurrency 1 4 16 64 256
```

压测脚本优先使用 `onnx/p232_001-009.wav`。如果本地样例不存在，会尝试从 `https://huggingface.co/datasets/xcczach/sample-data` 查找并下载音频。

### 结果

本次验证环境：Ubuntu Linux 5.15.0-141-generic，Python 3.13.9，CPU `INTEL(R) XEON(R) GOLD 6530`（128 vCPU），GPU `2 x NVIDIA GeForce RTX 4090`（服务本次使用 `CPUExecutionProvider`），ONNX 模型 `onnx/fastenhancer_t.onnx`，服务参数 `--workers 2`。压测音频为 1 秒 16 kHz 单声道 WAV。

| 并发 | 请求数 | 平均延时 (s) | P50 (s) | P95 (s) | 吞吐 (samples/s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.103 | 0.103 | 0.103 | 0.497 |
| 4 | 4 | 0.156 | 0.156 | 0.205 | 13.412 |
| 16 | 16 | 0.427 | 0.415 | 0.692 | 19.861 |
| 64 | 64 | 1.268 | 1.319 | 2.089 | 24.807 |
| 256 | 256 | 5.551 | 5.188 | 9.487 | 24.128 |
