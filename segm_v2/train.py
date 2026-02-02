"""
SEGM v2 訓練腳本
===============

這是 SEGM v2 的主要訓練腳本，實現自監督訓練流程。

訓練流程：
---------
1. 載入 DINOv3 預訓練模型
2. 包裝成 SEGMDinoVisionTransformer（凍結 DINO，只訓練 SEGM）
3. 使用 UnsupervisedSEGMLoss 進行自監督訓練
4. 定期視覺化 Grid 和頻率估計結果

使用方式：
---------
```bash
# 基本訓練
python train.py --data_dir /path/to/images --epochs 50

# 指定配置
python train.py --data_dir /path/to/images \
                --model vitb16 \
                --batch_size 16 \
                --lr 1e-4 \
                --segm_blocks 10 \
                --output_dir ./outputs
```

注意事項：
---------
1. 只需要未標註的氣孔圖片
2. 圖片會自動 resize 到模型輸入尺寸
3. 建議先執行驗證實驗確認假設成立
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# 將 segm_v2 加入 path
SEGM_PATH = Path(__file__).parent
sys.path.insert(0, str(SEGM_PATH.parent))

from segm_v2.models import SEGMDinoVisionTransformer, create_segm_dino
from segm_v2.losses import UnsupervisedSEGMLoss


# =============================================================================
# Dataset
# =============================================================================

class StomataDataset(Dataset):
    """
    氣孔圖片資料集（無標註）

    只需要圖片資料夾，會自動載入所有支援的圖片格式。
    """

    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    def __init__(
        self,
        data_dir: str,
        img_size: int = 224,
        augment: bool = True,
    ):
        """
        Args:
            data_dir: 圖片資料夾路徑
            img_size: 輸入尺寸
            augment: 是否使用資料增強
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size

        # 收集所有圖片
        self.image_paths = []
        for ext in self.SUPPORTED_EXTENSIONS:
            self.image_paths.extend(self.data_dir.glob(f"*{ext}"))
            self.image_paths.extend(self.data_dir.glob(f"*{ext.upper()}"))

        if len(self.image_paths) == 0:
            raise ValueError(f"在 {data_dir} 找不到任何圖片")

        print(f"找到 {len(self.image_paths)} 張圖片")

        # 建立 transform
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, str(img_path)


# =============================================================================
# Trainer
# =============================================================================

class SEGMTrainer:
    """
    SEGM v2 訓練器
    """

    def __init__(
        self,
        model: SEGMDinoVisionTransformer,
        loss_fn: UnsupervisedSEGMLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        output_dir: str = "./outputs",
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)

        # 建立輸出目錄
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)

        # 訓練紀錄
        self.history = {
            "epoch": [],
            "train_loss": [],
            "loss_components": [],
            "gate_value": [],
            "lr": [],
        }

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> dict:
        """
        訓練一個 epoch
        """
        self.model.train()

        # 只有 SEGM 參數需要訓練
        # DINO 參數已經凍結，但 model.train() 不影響它們

        total_loss = 0.0
        loss_components = {}
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch_idx, (images, paths) in enumerate(pbar):
            images = images.to(self.device)

            # Forward
            output = self.model(images)

            # 取得 SEGM 中間結果
            intermediates = self.model.get_intermediates()

            if not intermediates:
                print("警告: 沒有 SEGM 中間結果，跳過此 batch")
                continue

            # 取第一個（假設只有一個插入點）
            key = list(intermediates.keys())[0]
            segm_data = intermediates[key]

            # 計算損失
            loss, loss_dict = self.loss_fn(
                grid=segm_data["grid"],
                features=segm_data["spatial_features"],
                dominant_freq=segm_data["freq_info"]["dominant_freq"],
                freq_confidence=segm_data["freq_info"]["freq_confidence"],
            )

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.get_segm_parameters(), max_norm=1.0
            )

            self.optimizer.step()

            # 累積統計
            total_loss += loss.item()
            for k, v in loss_dict.items():
                if k not in loss_components:
                    loss_components[k] = 0.0
                loss_components[k] += v
            n_batches += 1

            # 更新進度條
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "gate": f"{segm_data['gate_value'].item():.4f}",
            })

        # 計算平均
        avg_loss = total_loss / max(n_batches, 1)
        avg_components = {k: v / max(n_batches, 1) for k, v in loss_components.items()}

        return {
            "loss": avg_loss,
            "components": avg_components,
        }

    def train(
        self,
        train_dataloader: DataLoader,
        epochs: int,
        val_dataloader: Optional[DataLoader] = None,
        save_every: int = 10,
        visualize_every: int = 5,
    ):
        """
        完整訓練流程
        """
        print("=" * 60)
        print("開始 SEGM v2 訓練")
        print("=" * 60)
        print(f"輸出目錄: {self.output_dir}")
        print(f"訓練 epochs: {epochs}")
        print(f"批次數: {len(train_dataloader)}")
        print("=" * 60)

        best_loss = float("inf")

        for epoch in range(1, epochs + 1):
            # 訓練一個 epoch
            train_result = self.train_epoch(train_dataloader, epoch)

            # 取得當前 gate 值
            gate_value = 0.0
            for adapter in self.model.segm_adapters.values():
                gate_value = adapter.get_gate_value()
                break

            # 取得當前 learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # 記錄
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_result["loss"])
            self.history["loss_components"].append(train_result["components"])
            self.history["gate_value"].append(gate_value)
            self.history["lr"].append(current_lr)

            # 印出結果
            print(f"\nEpoch {epoch}/{epochs}")
            print(f"  Loss: {train_result['loss']:.4f}")
            print(f"  Gate: {gate_value:.4f}")
            print(f"  LR: {current_lr:.6f}")
            print(f"  Components: ", end="")
            for k, v in train_result["components"].items():
                if k != "total":
                    print(f"{k}={v:.4f} ", end="")
            print()

            # 更新 scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # 儲存 checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(epoch, train_result["loss"])

            # 儲存最佳模型
            if train_result["loss"] < best_loss:
                best_loss = train_result["loss"]
                self.save_checkpoint(epoch, train_result["loss"], is_best=True)

            # 視覺化
            if epoch % visualize_every == 0:
                self.visualize(train_dataloader, epoch)

        # 儲存訓練歷史
        self.save_history()

        print("\n" + "=" * 60)
        print("訓練完成！")
        print(f"最佳 Loss: {best_loss:.4f}")
        print(f"輸出目錄: {self.output_dir}")
        print("=" * 60)

    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """
        儲存 checkpoint
        """
        checkpoint = {
            "epoch": epoch,
            "loss": loss,
            "segm_state_dict": {
                name: adapter.state_dict()
                for name, adapter in self.model.segm_adapters.items()
            },
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        if is_best:
            path = self.output_dir / "checkpoints" / "best_model.pth"
        else:
            path = self.output_dir / "checkpoints" / f"epoch_{epoch:04d}.pth"

        torch.save(checkpoint, path)
        print(f"  儲存 checkpoint: {path}")

    def save_history(self):
        """
        儲存訓練歷史
        """
        path = self.output_dir / "training_history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"儲存訓練歷史: {path}")

    def visualize(self, dataloader: DataLoader, epoch: int):
        """
        視覺化 Grid 和頻率估計結果
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib 未安裝，跳過視覺化")
            return

        self.model.eval()

        # 取一個 batch
        images, paths = next(iter(dataloader))
        images = images[:4].to(self.device)  # 只取前 4 張

        with torch.no_grad():
            output = self.model(images)
            intermediates = self.model.get_intermediates()

        if not intermediates:
            return

        key = list(intermediates.keys())[0]
        segm_data = intermediates[key]

        # 建立視覺化
        fig, axes = plt.subplots(4, 3, figsize=(12, 16))

        for i in range(min(4, len(images))):
            # 原圖（反正規化）
            img = images[i].cpu()
            img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            img = img.permute(1, 2, 0).clamp(0, 1).numpy()

            axes[i, 0].imshow(img)
            axes[i, 0].set_title("Input Image")
            axes[i, 0].axis("off")

            # Grid
            grid = segm_data["grid"][i, 0].cpu().numpy()
            axes[i, 1].imshow(grid, cmap="hot", vmin=0, vmax=1)
            axes[i, 1].set_title(f"Grid (mean={grid.mean():.3f})")
            axes[i, 1].axis("off")

            # 頻率估計
            freq = segm_data["freq_info"]["dominant_freq"][i].cpu().numpy()
            conf = segm_data["freq_info"]["freq_confidence"][i].cpu().numpy()
            axes[i, 2].barh(range(len(freq)), freq, color="blue", alpha=0.7, label="freq")
            axes[i, 2].barh(range(len(conf)), conf, color="red", alpha=0.3, label="conf")
            axes[i, 2].set_title("Freq & Confidence per Row")
            axes[i, 2].set_xlabel("Value")
            axes[i, 2].set_ylabel("Row")
            axes[i, 2].legend(loc="upper right")

        plt.suptitle(f"Epoch {epoch} - Gate: {segm_data['gate_value'].item():.4f}")
        plt.tight_layout()

        save_path = self.output_dir / "visualizations" / f"epoch_{epoch:04d}.png"
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()

        print(f"  儲存視覺化: {save_path}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="SEGM v2 Training")

    # 資料
    parser.add_argument("--data_dir", type=str, required=True,
                        help="圖片資料夾路徑")

    # 模型
    parser.add_argument("--model", type=str, default="vitb16",
                        choices=["vits16", "vitb16", "vitl16"],
                        help="DINO 模型類型")
    parser.add_argument("--segm_blocks", type=int, nargs="+", default=[10],
                        help="在哪些 Block 後插入 SEGM")

    # 訓練
    parser.add_argument("--epochs", type=int, default=50,
                        help="訓練 epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")

    # 輸出
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="輸出目錄")
    parser.add_argument("--save_every", type=int, default=10,
                        help="每幾個 epoch 儲存一次")
    parser.add_argument("--visualize_every", type=int, default=5,
                        help="每幾個 epoch 視覺化一次")

    # 其他
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    return parser.parse_args()


def main():
    args = parse_args()

    # 設定 seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 設備
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用設備: {device}")

    # 建立資料集
    print("\n載入資料集...")
    dataset = StomataDataset(
        data_dir=args.data_dir,
        img_size=224,
        augment=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 建立模型
    print("\n建立模型...")
    model = create_segm_dino(
        model_name=args.model,
        pretrained=True,
        segm_after_blocks=args.segm_blocks,
        device=device,
    )

    # 建立損失函數
    loss_fn = UnsupervisedSEGMLoss(
        intra_weight=1.0,
        inter_weight=0.5,
        period_weight=0.5,
        sparse_weight=0.3,
        freq_smooth_weight=0.2,
    )

    # 建立優化器（只優化 SEGM 參數）
    segm_params = model.get_segm_parameters()
    optimizer = AdamW(
        segm_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # 建立 scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )

    # 建立 Trainer
    trainer = SEGMTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=args.output_dir,
    )

    # 開始訓練
    trainer.train(
        train_dataloader=dataloader,
        epochs=args.epochs,
        save_every=args.save_every,
        visualize_every=args.visualize_every,
    )


if __name__ == "__main__":
    main()
