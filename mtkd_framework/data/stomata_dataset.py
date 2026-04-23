"""
Stomata dataset loader for MTKD training.
"""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from ultralytics.utils.ops import xyxyxyxy2xywhr


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class DatasetPaths:
    image_root: Path
    label_root: Optional[Path]


def _list_images(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS])


def _find_label_path(image_path: Path, label_root: Optional[Path]) -> Optional[Path]:
    stem = image_path.stem

    # 1) Explicit label root mapping (fast path for dataset layout).
    if label_root is not None:
        for ext in (".xml", ".txt"):
            candidate = label_root / f"{stem}{ext}"
            if candidate.exists():
                return candidate

        # 1.5) Preserve nested split structure when images live under
        # ``.../images/<split>/...`` and labels live under
        # ``.../labels/<split>/...``.
        for parent in image_path.parents:
            if parent.name != "images":
                continue
            try:
                rel = image_path.relative_to(parent)
            except ValueError:
                break
            for ext in (".xml", ".txt"):
                candidate = (label_root / rel).with_suffix(ext)
                if candidate.exists():
                    return candidate
            break

    # 1.6) Generic ``images -> labels`` sibling mapping and the project-
    # specific ``ALL_DATA_IMAGE -> ALL_DATA_LABEL`` fallback.
    for parent in image_path.parents:
        if parent.name != "images":
            continue
        try:
            rel = image_path.relative_to(parent)
        except ValueError:
            break

        candidate_roots = [parent.parent / "labels"]
        try:
            candidate_roots.append(parent.parent.with_name("ALL_DATA_LABEL") / "labels")
        except ValueError:
            pass

        for candidate_root in candidate_roots:
            for ext in (".xml", ".txt"):
                candidate = (candidate_root / rel).with_suffix(ext)
                if candidate.exists():
                    return candidate
        break

    # 2) Replace *_image_* with *_label_* in path.
    replaced_base = Path(str(image_path).replace("_image_", "_label_"))
    for ext in (".xml", ".txt"):
        replaced = replaced_base.with_suffix(ext)
        if replaced.exists():
            return replaced

    # 3) Same directory fallback.
    for ext in (".xml", ".txt"):
        same_dir = image_path.with_suffix(ext)
        if same_dir.exists():
            return same_dir

    return None


def _parse_obj_label(raw_name: Optional[str]) -> int:
    """Best-effort conversion from XML object name to class id."""
    if raw_name is None:
        return 0

    name = raw_name.strip()
    if not name:
        return 0

    try:
        return int(float(name))
    except ValueError:
        pass

    lowered = name.lower()
    known = {
        "complete": 0,
        "incomplete": 1,
        "stomata": 0,
    }
    return known.get(lowered, 0)


def _polygon_to_xywhr(coords: np.ndarray) -> np.ndarray:
    """Convert normalized polygon corners ``[N, 8]`` to normalized ``xywhr``."""
    xywhr = xyxyxyxy2xywhr(coords)
    if isinstance(xywhr, torch.Tensor):
        xywhr = xywhr.detach().cpu().numpy()
    return np.asarray(xywhr, dtype=np.float32)


def _horizontal_flip_boxes(bboxes: np.ndarray) -> np.ndarray:
    """Flip normalized ``xywhr`` boxes horizontally."""
    if bboxes.size == 0:
        return bboxes

    out = np.asarray(bboxes, dtype=np.float32).copy()
    out[:, 0] = 1.0 - out[:, 0]
    if out.shape[1] >= 5:
        out[:, 4] = -out[:, 4]
    return out


def _parse_xml_boxes(xml_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse XML annotations and return normalized ``xywhr`` boxes and labels.
    """
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        return np.zeros((0, 5), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    width = float(size.findtext("width", default="1"))
    height = float(size.findtext("height", default="1"))
    width = max(width, 1.0)
    height = max(height, 1.0)

    boxes: List[List[float]] = []
    labels: List[int] = []

    for obj in root.findall("object"):
        cls = _parse_obj_label(obj.findtext("name", default="0"))
        robnd = obj.find("robndbox")
        if robnd is not None:
            cx = float(robnd.findtext("cx", default="0")) / width
            cy = float(robnd.findtext("cy", default="0")) / height
            w = float(robnd.findtext("w", default="0")) / width
            h = float(robnd.findtext("h", default="0")) / height
            angle = float(robnd.findtext("angle", default="0"))
            boxes.append([cx, cy, w, h, angle])
            labels.append(cls)
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
            boxes.append([cx, cy, bw, bh, 0.0])
            labels.append(cls)

    if not boxes:
        return np.zeros((0, 5), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)


def _parse_txt_boxes(txt_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse YOLO txt labels.

    Supported row formats:
    - class cx cy w h
    - class cx cy w h angle  (xywhr)
    - class x1 y1 x2 y2 x3 y3 x4 y4  (OBB polygon)

    Returns normalized ``xywhr`` boxes so OBB supervision is preserved.
    """
    boxes: List[List[float]] = []
    labels: List[int] = []

    with txt_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            try:
                cls = int(float(parts[0]))
                values = [float(v) for v in parts[1:]]
            except ValueError:
                continue

            cx: float
            cy: float
            bw: float
            bh: float

            if len(values) >= 8:
                polygon = np.asarray(values[:8], dtype=np.float32).reshape(1, 8)
                cx, cy, bw, bh, angle = _polygon_to_xywhr(polygon)[0].tolist()
            elif len(values) >= 5:
                cx, cy, bw, bh, angle = values[:5]
            else:
                cx, cy, bw, bh = values[:4]
                angle = 0.0

            boxes.append([
                float(np.clip(cx, 0.0, 1.0)),
                float(np.clip(cy, 0.0, 1.0)),
                float(np.clip(bw, 0.0, 1.0)),
                float(np.clip(bh, 0.0, 1.0)),
                float(angle),
            ])
            labels.append(cls)

    if not boxes:
        return np.zeros((0, 5), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)


def _parse_label_boxes(label_path: Optional[Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse annotation file and return normalized ``xywhr`` boxes + labels.
    """
    if label_path is None or not label_path.exists():
        return np.zeros((0, 5), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    suffix = label_path.suffix.lower()
    if suffix == ".xml":
        return _parse_xml_boxes(label_path)
    if suffix == ".txt":
        return _parse_txt_boxes(label_path)

    return np.zeros((0, 5), dtype=np.float32), np.zeros((0,), dtype=np.int64)


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
        boxes_np, labels_np = _parse_label_boxes(label_path)

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
                boxes_np = _horizontal_flip_boxes(boxes_np)

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
    box_dim = 5
    for item in batch:
        max_targets = max(max_targets, item["targets"]["boxes"].shape[0])
        if item["targets"]["boxes"].ndim == 2:
            box_dim = max(box_dim, int(item["targets"]["boxes"].shape[1]))
    max_targets = max(max_targets, 1)

    boxes = torch.zeros((len(batch), max_targets, box_dim), dtype=torch.float32)
    labels = torch.zeros((len(batch), max_targets), dtype=torch.long)
    valid_mask = torch.zeros((len(batch), max_targets), dtype=torch.bool)

    for i, item in enumerate(batch):
        item_boxes = item["targets"]["boxes"]
        n = item_boxes.shape[0]
        if n == 0:
            continue
        copy_dim = min(box_dim, int(item_boxes.shape[1]))
        boxes[i, :n, :copy_dim] = item_boxes[:, :copy_dim]
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


class DualStreamBatchLoader:
    """
    Simple DINO-style dual-stream iterator.

    Each iteration yields a tuple:
    ``(labeled_batch, unlabeled_batch)``.

    If one stream is shorter, it is cycled so both streams are always present.
    """

    def __init__(self, labeled_loader: DataLoader, unlabeled_loader: DataLoader):
        if len(labeled_loader) == 0:
            raise ValueError("Labeled loader is empty")
        if len(unlabeled_loader) == 0:
            raise ValueError("Unlabeled loader is empty")

        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader

    def __len__(self) -> int:
        return max(len(self.labeled_loader), len(self.unlabeled_loader))

    def __iter__(self) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any]]]:
        labeled_iter = iter(self.labeled_loader)
        unlabeled_iter = iter(self.unlabeled_loader)

        for _ in range(len(self)):
            try:
                labeled_batch = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(self.labeled_loader)
                labeled_batch = next(labeled_iter)

            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(self.unlabeled_loader)
                unlabeled_batch = next(unlabeled_iter)

            yield labeled_batch, unlabeled_batch


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


def build_stomata_semisup_dataloaders(
    dataset_root: str,
    image_subdir: str,
    label_subdir: Optional[str],
    unlabeled_dataset_root: Optional[str],
    unlabeled_image_subdir: str,
    unlabeled_label_subdir: Optional[str],
    image_size: int = 640,
    val_ratio: float = 0.1,
    batch_size_label: int = 8,
    batch_size_unlabel: int = 8,
    num_workers: int = 4,
    seed: int = 42,
    augmentation: bool = True,
) -> Tuple[DualStreamBatchLoader, DataLoader]:
    """
    Build DINO-style dual-stream loaders.

    Labeled stream:
        uses ``dataset_root / image_subdir`` and ``dataset_root / label_subdir``
        with train/val split.

    Unlabeled stream:
        uses ``unlabeled_dataset_root / unlabeled_image_subdir``
        and optional ``unlabeled_label_subdir`` (can be missing).
    """
    labeled_train_loader, val_loader = build_stomata_dataloaders(
        dataset_root=dataset_root,
        image_subdir=image_subdir,
        label_subdir=label_subdir or "",
        image_size=image_size,
        val_ratio=val_ratio,
        batch_size=batch_size_label,
        num_workers=num_workers,
        seed=seed,
        augmentation=augmentation,
    )

    unlabel_root = Path(unlabeled_dataset_root) if unlabeled_dataset_root else Path(dataset_root)
    unlabel_image_root = (
        unlabel_root / unlabeled_image_subdir if unlabeled_image_subdir else unlabel_root
    )
    unlabel_label_root = (
        unlabel_root / unlabeled_label_subdir if unlabeled_label_subdir else None
    )
    if unlabel_label_root is not None and not unlabel_label_root.exists():
        unlabel_label_root = None

    unlabel_image_paths = _list_images(unlabel_image_root)
    if not unlabel_image_paths:
        raise FileNotFoundError(f"No unlabeled images found under: {unlabel_image_root}")

    unlabel_dataset = StomataBarleyDataset(
        image_paths=unlabel_image_paths,
        label_root=unlabel_label_root,
        image_size=image_size,
        augment=augmentation,
    )
    unlabel_loader = DataLoader(
        unlabel_dataset,
        batch_size=batch_size_unlabel,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_stomata_batch,
    )

    train_loader = DualStreamBatchLoader(labeled_train_loader, unlabel_loader)
    return train_loader, val_loader
