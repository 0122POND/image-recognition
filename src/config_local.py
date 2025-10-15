#!/usr/bin/env python3
"""
ローカル環境用設定ファイル

Google Colab用の設定をローカル環境用に調整したバージョン
"""

import os


class LocalConfig:
    """ローカル環境用設定クラス"""
    # データパス（ローカル環境用）
    train_img_dir = './data/train/images'
    train_mask_dir = './data/train/masks'
    val_img_dir = './data/val/images'
    val_mask_dir = './data/val/masks'
    test_img_dir = './data/test/images'
    output_mask_dir = './data/test/predicted_masks'
    
    # モデル保存パス
    save_dir = './models'
    best_model_path = f'{save_dir}/best_model.pth'
    
    # ハイパーパラメータ
    target_size = (1024, 1248)
    encoder_name = 'resnet18'
    encoder_weights = 'imagenet'
    in_channels = 3
    classes = 1
    batch_size = 4  # ローカル環境では少し大きく設定
    num_epochs = 50
    lr = 1e-3
    seed = 42
    
    # 推論用設定
    threshold = 0.5
    
    # ログ設定
    log_dir = './logs'
    tensorboard_dir = './runs'
    
    @classmethod
    def create_directories(cls):
        """必要なディレクトリを作成"""
        directories = [
            cls.save_dir,
            cls.output_mask_dir,
            cls.log_dir,
            cls.tensorboard_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"ディレクトリ作成: {directory}")


class InferenceConfig:
    """推論用設定クラス（ローカル環境用）"""
    test_img_dir = './data/test/images'
    output_mask_dir = './data/test/predicted_masks'
    model_path = './models/best_model.pth'
    target_size = (1024, 1248)
    encoder_name = 'resnet18'
    encoder_weights = None  # 推論では不要
    in_channels = 3
    classes = 1
    threshold = 0.5
