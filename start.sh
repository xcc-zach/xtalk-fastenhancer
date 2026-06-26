#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

VENV_DIR="${VENV_DIR:-.venv}"
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Virtual environment not found: ${VENV_DIR}. Run bash install.sh first." >&2
  exit 1
fi

source "${VENV_DIR}/bin/activate"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
ONNX_PATH="${ONNX_PATH:-onnx/fastenhancer_t.onnx}"
SAMPLE_RATE="${SAMPLE_RATE:-16000}"
N_FFT="${N_FFT:-512}"
HOP_SIZE="${HOP_SIZE:-0}"
WORKERS="${WORKERS:-$(python - <<'PY'
import os
print(os.cpu_count() or 4)
PY
)}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --onnx-path) ONNX_PATH="$2"; shift 2 ;;
    --sample-rate) SAMPLE_RATE="$2"; shift 2 ;;
    --n-fft) N_FFT="$2"; shift 2 ;;
    --hop-size) HOP_SIZE="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

python -m serving.server \
  --host "${HOST}" \
  --port "${PORT}" \
  --onnx-path "${ONNX_PATH}" \
  --sample-rate "${SAMPLE_RATE}" \
  --n-fft "${N_FFT}" \
  --hop-size "${HOP_SIZE}" \
  --workers "${WORKERS}"
