#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
MODEL_DIR="${MODEL_DIR:-onnx}"
MODEL_NAME="${MODEL_NAME:-fastenhancer_t.onnx}"
MODEL_PATH="${ONNX_PATH:-${MODEL_DIR}/${MODEL_NAME}}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

pip_install() {
  echo "+ python -m pip install $*"
  if python -m pip install "$@"; then
    return 0
  fi
  echo "pip install failed; retrying with proxy environment variables unset..."
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
    python -m pip install "$@"
}

pip_install --upgrade pip setuptools wheel

ORT_PACKAGE="${ORT_PACKAGE:-onnxruntime}"
if [[ "${INSTALL_GPU:-0}" == "1" ]]; then
  ORT_PACKAGE="${ORT_GPU_PACKAGE:-onnxruntime-gpu}"
fi

pip_install \
  fastapi \
  "uvicorn[standard]" \
  numpy \
  scipy \
  librosa \
  soundfile \
  python-multipart \
  requests \
  httpx \
  "${ORT_PACKAGE}"

mkdir -p "${MODEL_DIR}"

if [[ -f "${MODEL_PATH}" ]]; then
  echo "ONNX model already exists: ${MODEL_PATH}"
  exit 0
fi

download_model() {
  local tmp_path="${MODEL_PATH}.tmp"
  rm -f "${tmp_path}"

  local model_url="${MODEL_URL:-}"
  if [[ -z "${model_url}" ]]; then
    model_url="$(python - "$MODEL_NAME" "${SAMPLE_RATE:-16000}" <<'PYURL'
import json
import subprocess
import sys
import urllib.request

model_name = sys.argv[1].lower()
sample_rate = int(sys.argv[2])
api = "https://api.github.com/repos/aask1357/fastenhancer/releases"

fallbacks_16k = [
    "https://github.com/aask1357/fastenhancer/releases/download/onnx-vd-v1.0.0/fastenhancer_t.onnx",
    "https://github.com/aask1357/fastenhancer/releases/download/onnx-dns-v1.0.0/fastenhancer_t.onnx",
]
fallbacks_48k = [
    "https://github.com/aask1357/fastenhancer/releases/download/onnx-48khz-v1/fastenhancer_t.onnx",
]

def read_releases():
    try:
        req = urllib.request.Request(api, headers={"User-Agent": "fastenhancer-install"})
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        try:
            raw = subprocess.check_output(["curl", "-L", "-s", api], timeout=60)
            return json.loads(raw.decode("utf-8"))
        except Exception:
            raise

def release_score(release):
    tag = release.get("tag_name", "").lower()
    name = release.get("name", "").lower()
    label = f"{tag} {name}"
    if sample_rate == 48000:
        if "48khz" in label and "convstft" not in label:
            return 0
        if "48khz" in label:
            return 1
        return 9
    if "48khz" in label:
        return 9
    if "onnx-vd" in label:
        return 0
    if "onnx-dns" in label:
        return 1
    if "onnx" in label:
        return 2
    return 9

try:
    releases = read_releases()
    candidates = []
    for release in releases:
        score = release_score(release)
        if score >= 9:
            continue
        for asset in release.get("assets", []):
            name = asset.get("name", "").lower()
            if name != model_name:
                continue
            if name.endswith((".spec.onnx", "_spec.onnx")):
                continue
            url = asset.get("browser_download_url")
            if url:
                candidates.append((score, url))
    if candidates:
        print(sorted(candidates, key=lambda item: item[0])[0][1])
    else:
        print((fallbacks_48k if sample_rate == 48000 else fallbacks_16k)[0])
except Exception:
    print((fallbacks_48k if sample_rate == 48000 else fallbacks_16k)[0])
PYURL
)"
  fi

  echo "Downloading ${model_url} -> ${MODEL_PATH}"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 2 --connect-timeout 20 -o "${tmp_path}" "${model_url}"
  else
    python - "$model_url" "$tmp_path" <<'PYFETCH'
import sys
import urllib.request
url, target = sys.argv[1], sys.argv[2]
req = urllib.request.Request(url, headers={"User-Agent": "fastenhancer-install"})
with urllib.request.urlopen(req, timeout=120) as response, open(target, "wb") as output:
    output.write(response.read())
PYFETCH
  fi

  test -s "${tmp_path}"
  mv "${tmp_path}" "${MODEL_PATH}"
}

if download_model; then
  echo "Downloaded ONNX model: ${MODEL_PATH}"
else
  echo "Model download failed; retrying with proxy environment variables unset..."
  (
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
    download_model
  )
fi

echo "Install complete."
echo "Start with: bash start.sh --onnx-path ${MODEL_PATH}"
