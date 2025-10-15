# NVIDIA CUDA対応のPyTorchイメージをベースに使用
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# 作業ディレクトリを設定
WORKDIR /app

# システムパッケージの更新と必要なパッケージのインストール
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Pythonパッケージのインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY src/ ./src/
COPY data/ ./data/
COPY scripts/ ./scripts/

# 作業ディレクトリを/appに設定
WORKDIR /app

# デフォルトコマンド
CMD ["python", "src/train_unet.py"]
