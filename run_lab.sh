#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CONDA_BIN="/home/binly/miniconda3/bin/conda"
ENV_NAME="cuda-dev"

for script in 01_check_env.py 02_compile_transformer.py 03_capture_fx.py \
              04_graph_breaks.py 05_export_onnx.py; do
  echo "===== ${script} ====="
  "${CONDA_BIN}" run --no-capture-output -n "${ENV_NAME}" python "${script}"
done

echo "实验运行完成。"
