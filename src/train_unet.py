import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


class Config:
    """設定クラス"""
    # データパス
    train_img_dir = '/content/drive/MyDrive/dataset_split/train/images'
    train_mask_dir = '/content/drive/MyDrive/dataset_split/train/masks'
    val_img_dir = '/content/drive/MyDrive/dataset_split/val/images'
    val_mask_dir = '/content/drive/MyDrive/dataset_split/val/masks'
    save_dir = '/content/drive/MyDrive/shioda-lab-unet'
    best_model_path = f'{save_dir}/best_model.pth'

    # ハイパーパラメータ
    target_size = (1024, 1248)
    encoder_name = 'resnet18'
    encoder_weights = 'imagenet'
    in_channels = 3
    classes = 1
    batch_size = 2
    num_epochs = 50
    lr = 1e-3
    seed = 42


class SegmentationDataset(Dataset):
    """画像セグメンテーション用データセットクラス"""
    
    def __init__(self, image_dir, mask_dir, target_size=Config.target_size):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_names = sorted([f for f in os.listdir(image_dir) if f.endswith('.bmp')])
        self.target_size = target_size

        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.Resize(target_size[0], target_size[1]),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('.bmp', '.png'))

        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))  # バイナリマスク前提
        mask = (mask > 0).astype(np.uint8)

        augmented = self.transform(image=image, mask=mask)
        return augmented['image'], augmented['mask'].float()


def dice_loss(pred, target, smooth=1.):
    """Dice損失関数"""
    pred_prob = torch.sigmoid(pred).squeeze(1)
    intersection = (pred_prob * target).sum(dim=(1, 2))
    union = pred_prob.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def create_data_loaders():
    """データローダーを作成"""
    train_ds = SegmentationDataset(Config.train_img_dir, Config.train_mask_dir)
    val_ds = SegmentationDataset(Config.val_img_dir, Config.val_mask_dir)
    train_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=Config.batch_size)
    return train_loader, val_loader


def train_model():
    """モデルの学習を実行"""
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    
    # モデル初期化
    model = smp.UnetPlusPlus(
        encoder_name=Config.encoder_name,
        encoder_weights=Config.encoder_weights,
        in_channels=Config.in_channels,
        classes=Config.classes,
    ).to(device)
    
    # 損失関数とオプティマイザー
    criterion = dice_loss
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)
    
    # データローダー作成
    train_loader, val_loader = create_data_loaders()
    
    # 保存ディレクトリ作成
    os.makedirs(Config.save_dir, exist_ok=True)
    best_loss = float('inf')
    
    print(f"学習開始: {Config.num_epochs}エポック")
    print(f"バッチサイズ: {Config.batch_size}")
    print(f"学習率: {Config.lr}")
    
    # 学習ループ
    for epoch in range(Config.num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.num_epochs}"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{Config.num_epochs}, Loss: {avg_loss:.4f}")

        # 最良モデルの保存
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), Config.best_model_path)
            print("モデルを保存しました（新しい最良）")

    print("学習完了！")
    print(f"最良モデルは {Config.best_model_path} に保存されました")


def main():
    """メイン関数"""
    print("U-Net++画像セグメンテーション学習スクリプト")
    print("=" * 50)
    
    # シード設定
    torch.manual_seed(Config.seed)
    np.random.seed(Config.seed)
    
    # 学習実行
    train_model()


if __name__ == "__main__":
    main()
