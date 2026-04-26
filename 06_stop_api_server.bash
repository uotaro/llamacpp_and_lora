#!/bin/bash
# =======================================================================================
# 06_stop_api_server.bash - Llama-cpp API Server 停止用スクリプト
# =======================================================================================

PID_FILE="$(dirname "$0")/logs/llama_cpp.pid"

if [ ! -f "${PID_FILE}" ]; then
    echo "[ERROR] PID file not found. Server may not be running."
    exit 1
fi

PID=$(cat "${PID_FILE}")

if kill -0 "${PID}" 2>/dev/null; then
    echo "[INFO] Stopping server. PID: ${PID}"
    kill "${PID}"
    rm -f "${PID_FILE}"
    echo "[INFO] Server stopped."
else
    echo "[WARN] Process not found. Cleaning up PID file."
    rm -f "${PID_FILE}"
fi
