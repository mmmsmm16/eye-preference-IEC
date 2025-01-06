FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Avoid tzdata interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    sudo \
    wget \
    build-essential \
    git \
    python3 \
    python3-pip \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Requirements.txtをコピーして依存関係をインストール
COPY model/requirements.txt /app/
RUN python3 -m pip install -r requirements.txt

# アプリケーションコードをコピー
COPY model/src /app/src

# データディレクトリを作成して権限を設定
RUN mkdir -p /app/data/experiment_sessions && \
    chmod -R 777 /app/data

# デバッグ用の起動スクリプト
RUN echo '#!/bin/bash\n\
set -ex\n\
echo "Current working directory: $(pwd)"\n\
echo "Directory contents:"\n\
ls -la\n\
echo "Data directory contents:"\n\
ls -la data\n\
echo "Starting FastAPI server..."\n\
PYTHONPATH=/app python3 -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload --log-level debug\n' > /app/start.sh && \
    chmod +x /app/start.sh

# サーバー起動
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "debug"]
