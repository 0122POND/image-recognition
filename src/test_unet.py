#!/usr/bin/env python3
"""
U-Net++モデルを使った画像セグメンテーション推論スクリプト

このスクリプトは、Test_Unet.ipynbから変換されたPythonソースコードです。
学習済みU-Net++モデルを使って画像セグメンテーションの推論を行います。
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from glob import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


class InferenceConfig:
    """推論用設定クラス"""
    test_img_dir = '/content/drive/MyDrive/dataset_split/test/images20250806'
    output_mask_dir = '/content/drive/MyDrive/dataset_split/test/predicted_masks20250806'
    model_path = '/content/drive/MyDrive/shioda-lab-unet/best_model.pth'
    target_size = (1024, 1248)
    encoder_name = 'resnet18'
    encoder_weights = None  # 推論では不要
    in_channels = 3
    classes = 1
    threshold = 0.5


def load_model(device):
    """学習済みモデルを読み込み"""
    model = smp.UnetPlusPlus(
        encoder_name=InferenceConfig.encoder_name,
        encoder_weights=InferenceConfig.encoder_weights,
        in_channels=InferenceConfig.in_channels,
        classes=InferenceConfig.classes,
    ).to(device)
    
    model.load_state_dict(torch.load(InferenceConfig.model_path, map_location=device))
    model.eval()
    return model


def create_transform():
    """推論用の前処理変換を作成"""
    return A.Compose([
        A.Resize(InferenceConfig.target_size[0], InferenceConfig.target_size[1]),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])


def predict_masks(model, device, transform):
    """画像に対してマスクを予測し保存"""
    # 出力ディレクトリ作成
    os.makedirs(InferenceConfig.output_mask_dir, exist_ok=True)
    
    # テスト画像パス取得
    test_image_paths = sorted(glob(os.path.join(InferenceConfig.test_img_dir, '*.bmp')))
    
    if not test_image_paths:
        print(f"警告: {InferenceConfig.test_img_dir} に画像が見つかりません")
        return
    
    print(f"推論開始: {len(test_image_paths)}枚の画像を処理します")
    
    with torch.no_grad():
        for img_path in tqdm(test_image_paths, desc="推論中"):
            # 画像読み込み
            image = np.array(Image.open(img_path).convert('RGB'))
            transformed = transform(image=image)
            image_tensor = transformed['image'].unsqueeze(0).to(device)

            # 推論実行
            output = model(image_tensor)
            pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()

            # バイナリマスクに変換
            binary_mask = (pred_mask > InferenceConfig.threshold).astype(np.uint8) * 255
            
            # 結果保存
            out_name = os.path.basename(img_path).replace('.bmp', '_mask.png')
            out_path = os.path.join(InferenceConfig.output_mask_dir, out_name)
            Image.fromarray(binary_mask).save(out_path)
    
    print(f"推論完了！結果は {InferenceConfig.output_mask_dir} に保存されました")


def main():
    """メイン関数"""
    print("U-Net++画像セグメンテーション推論スクリプト")
    print("=" * 50)
    
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    
    # モデル読み込み
    print("学習済みモデルを読み込み中...")
    model = load_model(device)
    print("モデル読み込み完了")
    
    # 前処理変換作成
    transform = create_transform()
    
    # 推論実行
    predict_masks(model, device, transform)


if __name__ == "__main__":
    main()
