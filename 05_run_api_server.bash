#!/bin/bash
# =======================================================================================
# 05_run_api_server.bash - Llama-cpp API Server 起動用スクリプト
# 当スクリプトの実行方法：
# 1. 上記の設定セクションで、下記項目をあなたの環境に合わせて設定する。
#      - SERVER_EXE: llama-serverの実行ファイルへのフルパス
#      - MODEL_PATH: 使用するモデルファイルへのフルパス
#      - PORT: APIサーバーがリッスンするネットワークポート（デフォルトは8080）
# 2. ターミナルで以下のコマンドを入力して実行する。
#      bash 05_run_api_server.bash
# 3. または、実行権限を付与してから直接実行する。
#      chmod +x 05_run_api_server.bash
#      ./05_run_api_server.bash
# =======================================================================================

# --- Configuration ---
# Full path to the llama-server executable
SERVER_EXE="/home/ct1485/_local_wsl/08_tools/llama.cpp/build/bin/llama-server"

# Full path to your model file
MODEL_PATH="/home/ct1485/_local_wsl/08_tools/models/gguf/Llama-3.3-8B-Instruct.Q4_K_S.gguf"
# MODEL_PATH="/home/ct1485/_local_wsl/02_study/20260222_llm/model_gozaru_gguf/model.gguf" # ファインチューニングしたござるモデル

# Network port to listen on
PORT=8080  # ★ デフォルトは8080。必要に応じて変更してください。

# Number of layers to offload to GPU (-1 for all)
GPU_LAYERS=-1

# Context size (maximum tokens the model handles)
CTX_SIZE=2048

# --- Log Configuration ---
LOG_DIR="$(dirname "$0")/logs"
LOG_FILE="${LOG_DIR}/llama_cpp.log"
MAX_LOG_SIZE=$((10 * 1024 * 1024))  # 10MB
MAX_LOG_COUNT=5                      # 最大5世代分保持
PID_FILE="${LOG_DIR}/llama_cpp.pid"

# --- Functions ---

# ログローテート関数
rotate_log() {
    if [ -f "${LOG_FILE}" ] && [ "$(stat -c%s "${LOG_FILE}")" -ge "${MAX_LOG_SIZE}" ]; then
        echo "[INFO] Rotating log files..."
        # 古いログを1つずつずらす
        for i in $(seq $((MAX_LOG_COUNT - 1)) -1 1); do
            [ -f "${LOG_FILE}.${i}" ] && mv "${LOG_FILE}.${i}" "${LOG_FILE}.$((i + 1))"
        done
        mv "${LOG_FILE}" "${LOG_FILE}.1"
    fi
}

# 既存プロセスチェック関数
check_already_running() {
    if [ -f "${PID_FILE}" ]; then
        OLD_PID=$(cat "${PID_FILE}")
        if kill -0 "${OLD_PID}" 2>/dev/null; then
            echo "[ERROR] Server is already running. PID: ${OLD_PID}"
            echo "[ERROR] Stop it first: bash 06_stop_api_server.bash"
            exit 1
        else
            # PIDファイルが残っているが、プロセスは死んでいる場合
            echo "[WARN] Stale PID file found. Cleaning up..."
            rm -f "${PID_FILE}"
        fi
    fi
}    

# --- Main ---

# logsディレクトリ作成
mkdir -p "${LOG_DIR}"

# 既存プロセスチェック
check_already_running

# ログローテート
rotate_log

echo "[INFO] Starting Llama-cpp API Server..."
echo "[INFO] Model Path: ${MODEL_PATH}"
echo "[INFO] API URL:    http://localhost:${PORT}/v1"
echo "[INFO] Log File:   ${LOG_FILE}"

# バックグラウンドで起動、ログをファイルに出力
"${SERVER_EXE}" \
    -m "${MODEL_PATH}" \
    --n-gpu-layers ${GPU_LAYERS} \
    --ctx-size ${CTX_SIZE} \
    --port ${PORT} \
    >> "${LOG_FILE}" 2>&1 &

SERVER_PID=$!
echo "${SERVER_PID}" > "${PID_FILE}"

echo "[INFO] Server started. PID: ${SERVER_PID}"
echo "[INFO] Check log:   tail -f ${LOG_FILE}"
echo "[INFO] Stop server: bash 06_stop_api_server.bash"
