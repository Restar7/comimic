#!/usr/bin/env bash
# 起一个只读静态服务，把 ppt 通过 Jupyter/VS Code 代理暴露成可下载链接。
# 用法：bash docs/ppt_work/serve_ppt.sh [port]   （默认 8099）
set -euo pipefail
PORT="${1:-8099}"
HERE="$(cd "$(dirname "$0")" && pwd)"
if pgrep -f "http.server ${PORT}" >/dev/null 2>&1; then
  echo "[serve] already running on ${PORT}"
else
  setsid nohup python3 -m http.server "${PORT}" --bind 0.0.0.0 --directory "${HERE}/serve" \
    > "/tmp/ppt_http_${PORT}.log" 2>&1 < /dev/null &
  sleep 1
  echo "[serve] started on ${PORT}"
fi
echo "[serve] local check:"
curl -sS -o /dev/null -w "  /            -> %{http_code}\n" "http://127.0.0.1:${PORT}/" || true
curl -sS -o /dev/null -w "  deck.pptx    -> %{http_code} %{size_download} bytes\n" \
  "http://127.0.0.1:${PORT}/COMIMIC_p21_deck.pptx" || true
