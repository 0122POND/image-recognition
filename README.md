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

## 注意事項

- このプロジェクトは Google Colab 環境での使用を前提としています
- パス設定は Google Drive のマウントを前提としています
- ローカル環境で使用する場合は、Config クラス内のパスを適切に変更してください
