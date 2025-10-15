#!/bin/bash

# ローカル環境セットアップスクリプト

echo "=== ローカル環境セットアップ開始 ==="

# 必要なディレクトリを作成
echo "ディレクトリ構造を作成中..."
mkdir -p data/{train,val,test}/{images,masks}
mkdir -p data/test/predicted_masks
mkdir -p models
mkdir -p runs
mkdir -p logs

echo "ディレクトリ構造:"
tree data/ || ls -la data/

# Docker環境の確認
echo "=== Docker環境確認 ==="
if command -v docker &> /dev/null; then
    echo "Docker: $(docker --version)"
else
    echo "警告: Dockerがインストールされていません"
fi

if command -v docker-compose &> /dev/null; then
    echo "Docker Compose: $(docker-compose --version)"
else
    echo "警告: Docker Composeがインストールされていません"
fi

# NVIDIA Docker確認
echo "=== NVIDIA Docker確認 ==="
if command -v nvidia-docker &> /dev/null; then
    echo "NVIDIA Docker: 利用可能"
elif docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi &> /dev/null; then
    echo "NVIDIA Docker: 利用可能 (--gpus フラグ)"
else
    echo "警告: NVIDIA Dockerが利用できません"
fi

echo "=== セットアップ完了 ==="
echo ""
echo "使用方法:"
echo "1. 学習実行: docker-compose up unet-training"
echo "2. 推論実行: docker-compose --profile inference up unet-inference"
echo "3. Jupyter起動: docker-compose --profile jupyter up jupyter"
