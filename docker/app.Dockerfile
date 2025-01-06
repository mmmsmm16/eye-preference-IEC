FROM node:18-slim

# システム依存関係のインストール
RUN apt-get update && apt-get install -y \
    libgtk-3-0 \
    libnotify4 \
    libnss3 \
    libxss1 \
    libxtst6 \
    xauth \
    libgconf-2-4 \
    libasound2 \
    libx11-xcb1 \
    libxkbfile1 \
    libdrm2 \
    libgbm1 \
    libxrender1 \
    libfontconfig1 \
    libice6 \
    && rm -rf /var/lib/apt/lists/*

# 非rootユーザーを作成
RUN groupadd -r electronuser && \
    useradd -r -g electronuser -G audio,video electronuser && \
    mkdir -p /home/electronuser && \
    chown -R electronuser:electronuser /home/electronuser

WORKDIR /app

# パッケージファイルをコピー
COPY app/package*.json ./

# 依存関係をインストール
RUN npm install && \
    npm install -g electron@27.0.4

# アプリケーションのソースコードをコピー
COPY app/ ./

# 権限を設定
RUN chown -R electronuser:electronuser /app

# npmのグローバルディレクトリの権限も設定
RUN mkdir -p /home/electronuser/.npm && \
    chown -R electronuser:electronuser /home/electronuser/.npm

USER electronuser

# 起動コマンドを修正
CMD ["sh", "-c", "npm run webpack && npx electron . --no-sandbox"]
