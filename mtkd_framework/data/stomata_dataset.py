"""
Stomata dataset loader for MTKD training.
"""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class DatasetPaths:
    image_root: Path
    label_root: Optional[Path]


def _list_images(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS])


def _find_label_path(image_path: Path, label_root: Optional[Path]) -> Optional[Path]:
    stem = image_path.stem

    # 1) Explicit label root mapping (fast path for barley dataset layout).
    if label_root is not None:
        candidate = label_root / f"{stem}.xml"
        if candidate.exists():
            return candidate

    # 2) Replace *_image_* with *_label_* in path.
    replaced = Path(str(image_path).replace("_image_", "_label_")).with_suffix(".xml")
    if replaced.exists():
        return replaced

    # 3) Same directory fallback.
    same_dir = image_path.with_suffix(".xml")
    if same_dir.exists():
        return same_dir

    return None


def _parse_xml_boxes(xml_path: Optional[Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse XML annotations and return normalized cxcywh boxes and labels.
    """
    if xml_path is None or not xml_path.exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    width = float(size.findtext("width", default="1"))
    height = float(size.findtext("height", default="1"))
    width = max(width, 1.0)
    height = max(height, 1.0)

    boxes: List[List[float]] = []
    labels: List[int] = []

    for obj in root.findall("object"):
        robnd = obj.find("robndbox")
        if robnd is not None:
            cx = float(robnd.findtext("cx", default="0")) / width
            cy = float(robnd.findtext("cy", default="0")) / height
            w = float(robnd.findtext("w", default="0")) / width
            h = float(robnd.findtext("h", default="0")) / height
            boxes.append([cx, cy, w, h])
            labels.append(0)
            continue

        bnd = obj.find("bndbox")
        if bnd is not None:
            x1 = float(bnd.findtext("xmin", default="0")) / width
            y1 = float(bnd.findtext("ymin", default="0")) / height
            x2 = float(bnd.findtext("xmax", default="0")) / width
            y2 = float(bnd.findtext("ymax", default="0")) / height
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            boxes.append([cx, cy, bw, bh])
            labels.append(0)

    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)


class StomataBarleyDataset(Dataset):
    """
    Real dataset loader for barley stomata images.
    """

    def __init__(
        self,
        image_paths: Sequence[Path],
        label_root: Optional[Path] = None,
        image_size: int = 640,
        augment: bool = False,
    ):
        self.image_paths = list(image_paths)
        self.label_root = label_root
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_path = self.image_paths[idx]
        label_path = _find_label_path(image_path, self.label_root)

        image = Image.open(image_path).convert("RGB")
        boxes_np, labels_np = _parse_xml_boxes(label_path)

        # Dual-view pipeline:
        # - weak view: no geometric augmentation (for teacher alignment)
        # - strong view: geometric augmentation (for student detection)
        image_weak = image.copy()
        image_strong = image.copy()
        strong_hflip = False

        if self.augment and random.random() < 0.5:
            image_strong = ImageOps.mirror(image_strong)
            strong_hflip = True
            if boxes_np.shape[0] > 0:
                boxes_np[:, 0] = 1.0 - boxes_np[:, 0]

        image_weak = image_weak.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        image_strong = image_strong.resize((self.image_size, self.image_size), resample=Image.BILINEAR)

        image_weak_np = np.asarray(image_weak, dtype=np.float32) / 255.0
        image_strong_np = np.asarray(image_strong, dtype=np.float32) / 255.0

        image_weak_tensor = torch.from_numpy(image_weak_np).permute(2, 0, 1).contiguous()
        image_strong_tensor = torch.from_numpy(image_strong_np).permute(2, 0, 1).contiguous()

        boxes = torch.from_numpy(boxes_np)
        labels = torch.from_numpy(labels_np)

        return {
            "images": image_strong_tensor,
            "images_weak": image_weak_tensor,
            "strong_hflip": strong_hflip,
            "targets": {
                "boxes": boxes,
                "labels": labels,
            },
            "image_path": str(image_path),
        }


def collate_stomata_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    images = torch.stack([item["images"] for item in batch], dim=0)
    images_weak = torch.stack([item["images_weak"] for item in batch], dim=0)
    strong_hflip = torch.tensor(
        [bool(item.get("strong_hflip", False)) for item in batch],
        dtype=torch.bool,
    )

    max_targets = 0
    for item in batch:
        max_targets = max(max_targets, item["targets"]["boxes"].shape[0])
    max_targets = max(max_targets, 1)

    boxes = torch.zeros((len(batch), max_targets, 4), dtype=torch.float32)
    labels = torch.zeros((len(batch), max_targets), dtype=torch.long)
    valid_mask = torch.zeros((len(batch), max_targets), dtype=torch.bool)

    for i, item in enumerate(batch):
        n = item["targets"]["boxes"].shape[0]
        if n == 0:
            continue
        boxes[i, :n] = item["targets"]["boxes"]
        labels[i, :n] = item["targets"]["labels"]
        valid_mask[i, :n] = True

    return {
        "images": images,
        "images_weak": images_weak,
        "strong_hflip": strong_hflip,
        "targets": {
            "boxes": boxes,
            "labels": labels,
            "valid_mask": valid_mask,
        },
        "image_paths": [item["image_path"] for item in batch],
    }


def build_stomata_dataloaders(
    dataset_root: str,
    image_subdir: str = "barley_category/barley_image_fresh-leaf",
    label_subdir: str = "barley_category/barley_label_fresh-leaf",
    image_size: int = 640,
    val_ratio: float = 0.1,
    batch_size: int = 8,
    num_workers: int = 4,
    seed: int = 42,
    augmentation: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    root = Path(dataset_root)
    image_root = root / image_subdir if image_subdir else root
    label_root = root / label_subdir if label_subdir else None
    if label_root is not None and not label_root.exists():
        label_root = None

    image_paths = _list_images(image_root)
    if not image_paths:
        raise FileNotFoundError(f"No images found under: {image_root}")

    rng = random.Random(seed)
    indices = list(range(len(image_paths)))
    rng.shuffle(indices)

    # Keep training split non-empty for tiny datasets (e.g., BARLEY/1%).
    n_samples = len(indices)
    if n_samples == 1:
        val_size = 0
    else:
        raw_val = int(n_samples * val_ratio)
        # Ensure at least 1 val sample and at least 1 train sample.
        val_size = max(1, min(raw_val, n_samples - 1))

    val_indices = set(indices[:val_size])
    train_paths = [image_paths[i] for i in indices if i not in val_indices]
    val_paths = [image_paths[i] for i in indices if i in val_indices]

    if not train_paths:
        raise ValueError(
            f"No training samples after split. dataset_root={dataset_root}, "
            f"image_subdir={image_subdir}, total_images={n_samples}, val_ratio={val_ratio}"
        )

    train_dataset = StomataBarleyDataset(
        image_paths=train_paths,
        label_root=label_root,
        image_size=image_size,
        augment=augmentation,
    )
    val_dataset = StomataBarleyDataset(
        image_paths=val_paths,
        label_root=label_root,
        image_size=image_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_stomata_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_stomata_batch,
    )
    return train_loader, val_loader
