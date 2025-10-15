## image recognition

画像認識プロジェクト用リポジトリ。

## プロジェクト概要

このプロジェクトは、U-Net++モデルを使用した画像セグメンテーションの学習と推論を行うためのリポジトリです。

## ディレクトリ構造

```
image-recognition/
├── src/                    # Pythonソースコード
│   ├── train_unet.py      # 学習用スクリプト
│   └── test_unet.py       # 推論用スクリプト
├── notebooks/             # Jupyter notebook
│   ├── Train_Unet.ipynb   # 学習用notebook
│   └── Test_Unet.ipynb    # 推論用notebook
├── data/                  # データセット
├── runs/                  # 学習結果・ログ
└── scripts/               # その他のスクリプト
```

## 使用方法

### 1. 学習 (train_unet.py)

U-Net++モデルを使用して画像セグメンテーションの学習を行います。

```bash
python src/train_unet.py
```

**主な機能:**

- U-Net++アーキテクチャを使用したセグメンテーションモデル
- ResNet18 エンコーダー + ImageNet 事前学習済み重み
- Dice 損失関数による最適化
- データ拡張（水平反転、明度・コントラスト調整、回転・スケール変換）
- 最良モデルの自動保存

**設定パラメータ:**

- 画像サイズ: 1024×1248
- バッチサイズ: 2
- エポック数: 50
- 学習率: 1e-3
- エンコーダー: ResNet18

### 2. 推論 (test_unet.py)

学習済みモデルを使用して新しい画像に対してセグメンテーションを実行します。

```bash
python src/test_unet.py
```

**主な機能:**

- 学習済みモデルの読み込み
- テスト画像の一括処理
- バイナリマスクの生成（閾値: 0.5）
- 結果の PNG 形式での保存

**設定パラメータ:**

- 入力画像ディレクトリ: `/content/drive/MyDrive/dataset_split/test/images20250806`
- 出力マスクディレクトリ: `/content/drive/MyDrive/dataset_split/test/predicted_masks20250806`
- モデルパス: `/content/drive/MyDrive/shioda-lab-unet/best_model.pth`

## 必要なライブラリ

```bash
pip install torch torchvision
pip install segmentation-models-pytorch
pip install albumentations
pip install pillow
pip install tqdm
pip install numpy
```

## データセット形式

- **入力画像**: BMP 形式
- **マスク画像**: PNG 形式（バイナリマスク）
- **出力マスク**: PNG 形式（バイナリマスク）

## Docker 環境での実行

### 前提条件

- Docker & Docker Compose
- NVIDIA Docker Runtime（GPU 使用時）
- NVIDIA GPU（推奨）

### セットアップ

```bash
# 1. セットアップスクリプト実行
chmod +x scripts/setup_local.sh
./scripts/setup_local.sh

# 2. Dockerイメージビルド
docker-compose build
```

### 使用方法

#### 学習実行

```bash
# 学習を実行
docker-compose up unet-training

# バックグラウンド実行
docker-compose up -d unet-training
```

#### 推論実行

```bash
# 推論を実行
docker-compose --profile inference up unet-inference
```

#### Jupyter Lab 起動

```bash
# Jupyter Labを起動（http://localhost:8888）
docker-compose --profile jupyter up jupyter
```

### ディレクトリ構造（ローカル環境）

```
image-recognition/
├── src/
│   ├── train_unet.py      # 学習用スクリプト
│   ├── test_unet.py       # 推論用スクリプト
│   └── config_local.py    # ローカル用設定
├── data/
│   ├── train/
│   │   ├── images/        # 学習用画像
│   │   └── masks/         # 学習用マスク
│   ├── val/
│   │   ├── images/        # 検証用画像
│   │   └── masks/         # 検証用マスク
│   └── test/
│       ├── images/        # テスト用画像
│       └── predicted_masks/ # 推論結果
├── models/                # 学習済みモデル
├── runs/                  # 学習ログ・結果
├── logs/                  # ログファイル
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 注意事項

### Google Colab 環境

- このプロジェクトは Google Colab 環境での使用を前提としています
- パス設定は Google Drive のマウントを前提としています

### ローカル環境

- Docker 環境では`src/config_local.py`の設定を使用
- データセットは`data/`ディレクトリに配置
- GPU 使用時は NVIDIA Docker Runtime が必要
- ローカル環境で使用する場合は、Config クラス内のパスを適切に変更してください
