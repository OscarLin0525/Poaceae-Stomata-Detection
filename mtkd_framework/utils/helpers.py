"""
MTKD Utility Functions

輔助函數，包括:
- Checkpoint 管理
- 日誌設置
- 訓練工具
"""

import torch
import torch.nn as nn
import logging
import os
from typing import Dict, Optional, Any, List
from datetime import datetime
import json


def setup_logger(
    name: str = "mtkd",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_str: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> logging.Logger:
    """
    設置日誌記錄器

    Args:
        name: Logger 名稱
        log_file: 日誌文件路徑（可選）
        level: 日誌等級
        format_str: 日誌格式

    Returns:
        logger: 配置好的 logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 清除現有 handlers
    logger.handlers.clear()

    formatter = logging.Formatter(format_str)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    save_path: str,
    scheduler: Optional[Any] = None,
    extra_info: Optional[Dict] = None,
):
    """
    保存訓練 checkpoint

    Args:
        model: 模型
        optimizer: 優化器
        epoch: 當前 epoch
        loss: 當前損失
        save_path: 保存路徑
        scheduler: 學習率調度器（可選）
        extra_info: 額外信息（可選）
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "timestamp": datetime.now().isoformat(),
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if extra_info is not None:
        checkpoint["extra_info"] = extra_info

    torch.save(checkpoint, save_path)
    logging.info(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    strict: bool = True,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """
    載入訓練 checkpoint

    Args:
        model: 模型
        checkpoint_path: Checkpoint 路徑
        optimizer: 優化器（可選）
        scheduler: 學習率調度器（可選）
        strict: 是否嚴格匹配 state_dict
        map_location: 設備映射

    Returns:
        checkpoint: 載入的 checkpoint 字典
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    # 載入模型權重
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=strict)
    else:
        model.load_state_dict(checkpoint, strict=strict)

    # 載入優化器狀態
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # 載入調度器狀態
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    logging.info(f"Checkpoint loaded from {checkpoint_path}")
    return checkpoint


class AverageMeter:
    """
    追蹤和計算平均值

    Example:
        >>> meter = AverageMeter("loss")
        >>> meter.update(0.5)
        >>> meter.update(0.3)
        >>> print(meter.avg)  # 0.4
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0

    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


class AverageMeterDict:
    """
    追蹤多個指標的平均值

    Example:
        >>> meters = AverageMeterDict()
        >>> meters.update({"loss": 0.5, "acc": 0.8})
        >>> meters.update({"loss": 0.3, "acc": 0.9})
        >>> print(meters.get_averages())  # {"loss": 0.4, "acc": 0.85}
    """

    def __init__(self):
        self.meters: Dict[str, AverageMeter] = {}

    def update(self, values: Dict[str, float], n: int = 1):
        for key, val in values.items():
            if key not in self.meters:
                self.meters[key] = AverageMeter(key)

            # 處理 tensor
            if isinstance(val, torch.Tensor):
                val = val.item()

            self.meters[key].update(val, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def get_averages(self) -> Dict[str, float]:
        return {key: meter.avg for key, meter in self.meters.items()}

    def __str__(self) -> str:
        return " | ".join(str(meter) for meter in self.meters.values())


class EarlyStopping:
    """
    Early Stopping 機制

    Args:
        patience: 容忍的 epoch 數
        min_delta: 最小改善閾值
        mode: "min" 或 "max"

    Example:
        >>> early_stopping = EarlyStopping(patience=10, mode="min")
        >>> for epoch in range(100):
        ...     val_loss = train_epoch()
        ...     if early_stopping(val_loss):
        ...         print("Early stopping!")
        ...         break
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def reset(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class GradientClipper:
    """
    梯度裁剪器

    Args:
        max_norm: 最大梯度範數
        norm_type: 範數類型（2 為 L2 範數）
    """

    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type

    def __call__(self, parameters) -> float:
        """返回裁剪前的梯度範數"""
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = list(filter(lambda p: p.grad is not None, parameters))

        total_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            self.max_norm,
            norm_type=self.norm_type,
        )
        return total_norm.item()


def compute_flops(model: nn.Module, input_size: tuple = (1, 3, 224, 224)) -> int:
    """
    計算模型的 FLOPs

    Args:
        model: 模型
        input_size: 輸入大小

    Returns:
        flops: FLOPs 數量
    """
    try:
        from fvcore.nn import FlopCountAnalysis, flop_count_str

        input_tensor = torch.randn(input_size)
        flops = FlopCountAnalysis(model, input_tensor)
        return flops.total()
    except ImportError:
        logging.warning("fvcore not installed. Cannot compute FLOPs.")
        return -1


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    計算模型參數數量

    Args:
        model: 模型
        trainable_only: 是否只計算可訓練參數

    Returns:
        count: 參數數量
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_time(seconds: float) -> str:
    """
    格式化時間

    Args:
        seconds: 秒數

    Returns:
        formatted: 格式化的時間字符串
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def seed_everything(seed: int = 42):
    """
    設置隨機種子

    Args:
        seed: 隨機種子
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 確保確定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logging.info(f"Random seed set to {seed}")


def save_config(config: Dict, save_path: str):
    """保存配置到 JSON 文件"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(config, f, indent=2)
    logging.info(f"Config saved to {save_path}")


def load_config(config_path: str) -> Dict:
    """從 JSON 文件載入配置"""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config
