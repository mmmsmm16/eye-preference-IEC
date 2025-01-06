#!/bin/bash

# スクリプトが失敗した場合に即座に終了
set -e

# 開始メッセージ
echo "Starting FastAPI server..."

# 環境変数の確認
echo "Environment variables:"
echo "PYTHONPATH: $PYTHONPATH"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# ディレクトリ構造の確認
echo "Directory structure:"
ls -la /app/model/src

# サーバーの起動
echo "Launching uvicorn server..."
python3 -m uvicorn model.src.api.server:app --host 0.0.0.0 --port 8000 --reload --log-level debug
