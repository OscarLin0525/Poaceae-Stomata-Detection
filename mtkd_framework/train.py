"""
MTKD Training Script

Multi-Teacher Knowledge Distillation 訓練腳本

Usage:
    python -m mtkd_framework.train --config config.json
    python -m mtkd_framework.train --config config.json --resume checkpoint.pth
"""

import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any
import logging

from .models import MTKDModel, build_mtkd_model
from .utils import (
    setup_logger,
    save_checkpoint,
    load_checkpoint,
    AverageMeterDict,
    EarlyStopping,
    GradientClipper,
    seed_everything,
    save_config,
    load_config,
    format_time,
    count_parameters,
)


def get_default_config() -> Dict[str, Any]:
    """
    獲取默認配置

    Returns:
        config: 默認配置字典
    """
    return {
        # 模型配置
        "model": {
            "num_classes": 1,  # 氣孔檢測為單類
            "student_config": {
                "backbone_config": {
                    "backbone_type": "resnet50",
                    "pretrained": True,
                },
                "head_config": {
                    "num_classes": 1,
                    "num_queries": 100,
                    "hidden_dim": 256,
                    "num_heads": 8,
                    "num_layers": 6,
                },
                "adapter_config": {
                    "adapter_type": "mlp",
                },
            },
            "dino_teacher_config": {
                "model_name": "vit_base",
                "patch_size": 16,
                "embed_dim": 768,
            },
            "ensemble_config": {
                "fusion_method": "wbf",
                "fusion_config": {
                    "iou_threshold": 0.55,
                    "skip_box_threshold": 0.0,
                },
            },
            "loss_config": {
                "feature_weight": 1.0,
                "prediction_weight": 2.0,
                "detection_weight": 1.0,
                "feature_loss_config": {
                    "loss_type": "cosine",
                    "normalize": True,
                },
                "prediction_loss_config": {
                    "box_loss_type": "giou",
                    "class_loss_type": "kl",
                    "temperature": 4.0,
                },
                "warmup_epochs": 5,
                "weight_schedule": "cosine",
            },
        },
        # 訓練配置
        "training": {
            "epochs": 100,
            "batch_size": 8,
            "num_workers": 4,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "lr_scheduler": "cosine",
            "warmup_epochs": 5,
            "min_lr": 1e-6,
            "gradient_clip_max_norm": 1.0,
            "accumulation_steps": 1,
            "mixed_precision": True,
            "early_stopping_patience": 20,
        },
        # 數據配置
        "data": {
            "train_path": "data/train",
            "val_path": "data/val",
            "image_size": 640,
            "augmentation": True,
        },
        # 輸出配置
        "output": {
            "save_dir": "outputs/mtkd",
            "save_freq": 5,
            "log_freq": 10,
        },
        # Checkpoints
        "checkpoints": {
            "dino_checkpoint": None,
            "teacher_checkpoints": [],
            "resume": None,
        },
        # 其他
        "seed": 42,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


class MTKDTrainer:
    """
    MTKD 訓練器

    封裝完整的訓練流程。

    Args:
        config: 訓練配置
        model: MTKD 模型（可選，如果不提供則根據配置創建）
        train_loader: 訓練數據載入器
        val_loader: 驗證數據載入器

    Example:
        >>> config = get_default_config()
        >>> trainer = MTKDTrainer(config, train_loader=train_loader, val_loader=val_loader)
        >>> trainer.train()
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model: Optional[MTKDModel] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
    ):
        self.config = config
        self.device = torch.device(config.get("device", "cuda"))

        # 設置日誌
        os.makedirs(config["output"]["save_dir"], exist_ok=True)
        log_file = os.path.join(config["output"]["save_dir"], "training.log")
        self.logger = setup_logger("mtkd_trainer", log_file=log_file)

        # 設置隨機種子
        seed_everything(config.get("seed", 42))

        # 保存配置
        config_path = os.path.join(config["output"]["save_dir"], "config.json")
        save_config(config, config_path)

        # 初始化模型
        if model is not None:
            self.model = model
        else:
            self.model = self._build_model()

        self.model = self.model.to(self.device)

        # 記錄模型信息
        total_params = count_parameters(self.model)
        trainable_params = count_parameters(self.model, trainable_only=True)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")

        # 數據載入器
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 初始化優化器和調度器
        self._setup_optimizer()
        self._setup_scheduler()

        # 混合精度訓練
        self.scaler = None
        if config["training"].get("mixed_precision", False) and self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
            self.logger.info("Mixed precision training enabled")

        # 梯度裁剪
        self.gradient_clipper = GradientClipper(
            max_norm=config["training"].get("gradient_clip_max_norm", 1.0)
        )

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config["training"].get("early_stopping_patience", 20),
            mode="min",
        )

        # 訓練狀態
        self.start_epoch = 0
        self.best_loss = float("inf")

        # 載入 checkpoint（如果有）
        if config["checkpoints"].get("resume"):
            self._load_checkpoint(config["checkpoints"]["resume"])

    def _build_model(self) -> MTKDModel:
        """構建模型"""
        model_config = self.config["model"]
        checkpoints = self.config["checkpoints"]

        return build_mtkd_model(
            config=model_config,
            dino_checkpoint=checkpoints.get("dino_checkpoint"),
            teacher_checkpoints=checkpoints.get("teacher_checkpoints"),
        )

    def _setup_optimizer(self):
        """設置優化器"""
        training_config = self.config["training"]

        # 獲取可訓練參數
        params = self.model.get_trainable_parameters()

        self.optimizer = torch.optim.AdamW(
            params,
            lr=training_config["learning_rate"],
            weight_decay=training_config["weight_decay"],
        )

    def _setup_scheduler(self):
        """設置學習率調度器"""
        training_config = self.config["training"]
        scheduler_type = training_config.get("lr_scheduler", "cosine")

        if scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_config["epochs"],
                eta_min=training_config.get("min_lr", 1e-6),
            )
        elif scheduler_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1,
            )
        elif scheduler_type == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=10,
            )
        else:
            self.scheduler = None

    def _load_checkpoint(self, checkpoint_path: str):
        """載入 checkpoint"""
        checkpoint = load_checkpoint(
            self.model,
            checkpoint_path,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        self.start_epoch = checkpoint.get("epoch", 0) + 1
        self.best_loss = checkpoint.get("loss", float("inf"))
        self.logger.info(f"Resumed from epoch {self.start_epoch}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        訓練一個 epoch

        Args:
            epoch: 當前 epoch

        Returns:
            metrics: 訓練指標字典
        """
        self.model.train()
        meters = AverageMeterDict()
        accumulation_steps = self.config["training"].get("accumulation_steps", 1)

        start_time = time.time()
        num_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            # 準備數據
            images = batch["images"].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch.get("targets", {}).items()}

            # 前向傳播
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    loss, loss_dict = self.model.training_step(images, targets, epoch=epoch)
                    loss = loss / accumulation_steps
            else:
                loss, loss_dict = self.model.training_step(images, targets, epoch=epoch)
                loss = loss / accumulation_steps

            # 反向傳播
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # 梯度累積
            if (batch_idx + 1) % accumulation_steps == 0:
                # 梯度裁剪
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                grad_norm = self.gradient_clipper(self.model.get_trainable_parameters())

                # 優化器步驟
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

            # 更新指標
            meters.update(loss_dict, n=images.size(0))

            # 日誌
            if (batch_idx + 1) % self.config["output"]["log_freq"] == 0:
                elapsed = time.time() - start_time
                eta = elapsed / (batch_idx + 1) * (num_batches - batch_idx - 1)
                self.logger.info(
                    f"Epoch [{epoch}][{batch_idx + 1}/{num_batches}] "
                    f"Loss: {loss_dict['total_loss']:.4f} "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.6f} "
                    f"ETA: {format_time(eta)}"
                )

        return meters.get_averages()

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        驗證

        Args:
            epoch: 當前 epoch

        Returns:
            metrics: 驗證指標字典
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        meters = AverageMeterDict()

        for batch in self.val_loader:
            images = batch["images"].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch.get("targets", {}).items()}

            # 前向傳播
            _, loss_dict = self.model(images, targets, return_loss=True, epoch=epoch)

            meters.update(loss_dict, n=images.size(0))

        return meters.get_averages()

    def train(self):
        """
        完整訓練流程
        """
        training_config = self.config["training"]
        output_config = self.config["output"]

        self.logger.info("=" * 50)
        self.logger.info("Starting MTKD Training")
        self.logger.info("=" * 50)

        total_start_time = time.time()

        for epoch in range(self.start_epoch, training_config["epochs"]):
            epoch_start_time = time.time()

            # 訓練
            train_metrics = self.train_epoch(epoch)

            # 驗證
            val_metrics = self.validate(epoch)

            # 更新學習率
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get("total_loss", train_metrics["total_loss"]))
                else:
                    self.scheduler.step()

            # 記錄
            epoch_time = time.time() - epoch_start_time
            self.logger.info(
                f"Epoch {epoch} completed in {format_time(epoch_time)} | "
                f"Train Loss: {train_metrics['total_loss']:.4f} | "
                f"Val Loss: {val_metrics.get('total_loss', 'N/A')}"
            )

            # 保存 checkpoint
            current_loss = val_metrics.get("total_loss", train_metrics["total_loss"])

            if (epoch + 1) % output_config["save_freq"] == 0:
                save_path = os.path.join(output_config["save_dir"], f"checkpoint_epoch_{epoch}.pth")
                save_checkpoint(
                    self.model, self.optimizer, epoch, current_loss, save_path,
                    scheduler=self.scheduler,
                    extra_info={"train_metrics": train_metrics, "val_metrics": val_metrics},
                )

            # 保存最佳模型
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                best_path = os.path.join(output_config["save_dir"], "best_model.pth")
                save_checkpoint(
                    self.model, self.optimizer, epoch, current_loss, best_path,
                    scheduler=self.scheduler,
                )
                self.logger.info(f"New best model saved with loss: {self.best_loss:.4f}")

            # Early stopping
            if self.early_stopping(current_loss):
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        total_time = time.time() - total_start_time
        self.logger.info("=" * 50)
        self.logger.info(f"Training completed in {format_time(total_time)}")
        self.logger.info(f"Best validation loss: {self.best_loss:.4f}")
        self.logger.info("=" * 50)


def create_dummy_dataloader(batch_size: int = 4, num_samples: int = 100) -> DataLoader:
    """
    創建用於測試的 dummy DataLoader

    Args:
        batch_size: 批次大小
        num_samples: 樣本數量

    Returns:
        dataloader: DataLoader
    """
    from torch.utils.data import Dataset

    class DummyDataset(Dataset):
        def __init__(self, num_samples: int, image_size: int = 640):
            self.num_samples = num_samples
            self.image_size = image_size

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            return {
                "images": torch.randn(3, self.image_size, self.image_size),
                "targets": {
                    "boxes": torch.rand(10, 4),  # 10 objects
                    "labels": torch.zeros(10, dtype=torch.long),
                },
            }

    dataset = DummyDataset(num_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="MTKD Training")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--dino_checkpoint", type=str, default=None, help="DINO checkpoint path")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--test_run", action="store_true", help="Run with dummy data for testing")
    args = parser.parse_args()

    # 載入或創建配置
    if args.config:
        config = load_config(args.config)
    else:
        config = get_default_config()

    # 覆蓋配置
    if args.resume:
        config["checkpoints"]["resume"] = args.resume
    if args.dino_checkpoint:
        config["checkpoints"]["dino_checkpoint"] = args.dino_checkpoint
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.output_dir:
        config["output"]["save_dir"] = args.output_dir
    if args.device:
        config["device"] = args.device
    if args.seed:
        config["seed"] = args.seed

    # 創建數據載入器
    if args.test_run:
        print("Running test with dummy data...")
        train_loader = create_dummy_dataloader(
            batch_size=config["training"]["batch_size"],
            num_samples=50,
        )
        val_loader = create_dummy_dataloader(
            batch_size=config["training"]["batch_size"],
            num_samples=20,
        )
        config["training"]["epochs"] = 2
    else:
        # TODO: 實現真實數據載入器
        # train_loader = create_real_dataloader(config["data"]["train_path"], ...)
        # val_loader = create_real_dataloader(config["data"]["val_path"], ...)
        raise NotImplementedError(
            "Real data loader not implemented. Use --test_run for testing or implement your own data loader."
        )

    # 創建訓練器並開始訓練
    trainer = MTKDTrainer(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    trainer.train()


if __name__ == "__main__":
    main()
