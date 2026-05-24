#!/usr/bin/env python3
"""Evaluate DINO-bypass OBB predictions exported by train_dino_bypass_offline.py."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent
ULTRALYTICS_ROOT = REPO_ROOT / "ultralytics"
for candidate in (REPO_ROOT, ULTRALYTICS_ROOT):
    if candidate.is_dir():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

from ultralytics.utils import ops
from ultralytics.utils.metrics import ap_per_class, batch_probiou
from ultralytics.utils.nms import TorchNMS


def _read_data_yaml(path: Path) -> Tuple[List[Path], Dict[int, str]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Dataset YAML must be a mapping: {path}")

    base = Path(data.get("path", "."))
    val = Path(data.get("val") or data.get("test") or data.get("images"))
    if not val.is_absolute():
        val = base / val

    if val.is_file():
        image_paths = [Path(line.strip()) for line in val.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        image_paths = sorted(
            p for p in val.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        )
    if not image_paths:
        raise FileNotFoundError(f"No validation images found from {path}")

    names_raw = data.get("names", {})
    if isinstance(names_raw, dict):
        names = {int(k): str(v) for k, v in names_raw.items()}
    elif isinstance(names_raw, list):
        names = {idx: str(v) for idx, v in enumerate(names_raw)}
    else:
        names = {}
    return image_paths, names


def _label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def _read_label_file(label_path: Path, width: int, height: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not label_path.is_file():
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 5), dtype=np.float32),
        )
    boxes: List[List[float]] = []
    classes: List[int] = []
    rboxes: List[List[float]] = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        vals = [float(x) for x in raw.split()]
        cls = int(vals[0])
        if len(vals) == 9:
            poly_norm = np.asarray(vals[1:], dtype=np.float32).reshape(4, 2)
            poly = poly_norm.copy()
            poly[:, 0] *= float(width)
            poly[:, 1] *= float(height)
        elif len(vals) == 5:
            _, cx, cy, bw, bh = vals
            x1 = (cx - 0.5 * bw) * width
            y1 = (cy - 0.5 * bh) * height
            x2 = (cx + 0.5 * bw) * width
            y2 = (cy + 0.5 * bh) * height
            poly = np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        else:
            continue
        x1, y1 = poly[:, 0].min(), poly[:, 1].min()
        x2, y2 = poly[:, 0].max(), poly[:, 1].max()
        boxes.append([float(x1), float(y1), float(x2), float(y2)])
        classes.append(cls)
        rbox = ops.xyxyxyxy2xywhr(torch.tensor(poly.reshape(1, 8), dtype=torch.float32))[0].cpu().numpy()
        rboxes.append(rbox.astype(float).tolist())
    if not boxes:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 5), dtype=np.float32),
        )
    return np.asarray(boxes, dtype=np.float32), np.asarray(classes, dtype=np.int64), np.asarray(rboxes, dtype=np.float32)


def _load_predictions(path: Path) -> Dict[str, Dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    images = data.get("images", []) if isinstance(data, dict) else []
    index: Dict[str, Dict[str, object]] = {}
    for item in images:
        if not isinstance(item, dict):
            continue
        image_path = Path(str(item.get("image_path", "")))
        if image_path.name:
            index[image_path.name] = item
            index[image_path.stem] = item
    return index


def _preds_for_image(item: Optional[Dict[str, object]]) -> Tuple[np.ndarray, np.ndarray]:
    if not item:
        return np.zeros((0, 6), dtype=np.float32), np.zeros((0, 5), dtype=np.float32)
    preds = item.get("predictions", [])
    boxes: List[List[float]] = []
    rboxes: List[List[float]] = []
    for pred in preds if isinstance(preds, list) else []:
        if not isinstance(pred, dict):
            continue
        poly = np.asarray(pred.get("poly", []), dtype=np.float32).reshape(-1, 2)
        if poly.shape != (4, 2):
            continue
        conf = float(pred.get("conf", 0.0) or 0.0)
        cls = int(pred.get("cls", 0))
        x1, y1 = poly[:, 0].min(), poly[:, 1].min()
        x2, y2 = poly[:, 0].max(), poly[:, 1].max()
        boxes.append([float(x1), float(y1), float(x2), float(y2), conf, float(cls)])
        rbox = ops.xyxyxyxy2xywhr(torch.tensor(poly.reshape(1, 8), dtype=torch.float32))[0].cpu().numpy()
        rboxes.append(rbox.astype(float).tolist())
    if not boxes:
        return np.zeros((0, 6), dtype=np.float32), np.zeros((0, 5), dtype=np.float32)
    order = np.argsort(-np.asarray([row[4] for row in boxes], dtype=np.float32))
    return np.asarray(boxes, dtype=np.float32)[order], np.asarray(rboxes, dtype=np.float32)[order]


def _nms_rotated(
    preds: np.ndarray,
    rboxes: np.ndarray,
    iou_thres: float,
    max_det: int,
    agnostic: bool,
    max_wh: float = 7680.0,
) -> Tuple[np.ndarray, np.ndarray]:
    if preds.size == 0:
        return preds, rboxes
    scores = torch.tensor(preds[:, 4], dtype=torch.float32)
    boxes = torch.tensor(rboxes[:, :5], dtype=torch.float32)
    if not agnostic:
        cls = torch.tensor(preds[:, 5], dtype=torch.float32).view(-1, 1)
        boxes = boxes.clone()
        boxes[:, :2] += cls * max_wh
    keep = TorchNMS.fast_nms(boxes, scores, iou_thres, iou_func=batch_probiou)
    keep = keep[:max_det].cpu().numpy()
    return preds[keep], rboxes[keep]


def _match_predictions(
    preds: np.ndarray,
    pred_rboxes: np.ndarray,
    gt_cls: np.ndarray,
    gt_rboxes: np.ndarray,
    iou_thrs: np.ndarray,
) -> np.ndarray:
    tp = np.zeros((preds.shape[0], len(iou_thrs)), dtype=bool)
    if preds.shape[0] == 0 or gt_cls.shape[0] == 0:
        return tp
    pred_cls = preds[:, 5].astype(int)
    ious_all = batch_probiou(
        torch.tensor(gt_rboxes, dtype=torch.float32),
        torch.tensor(pred_rboxes, dtype=torch.float32),
    ).cpu().numpy()
    order = np.argsort(-preds[:, 4])
    for ti, thr in enumerate(iou_thrs):
        assigned = np.zeros(gt_cls.shape[0], dtype=bool)
        for pred_idx in order:
            ious = ious_all[:, pred_idx]
            candidates = np.where((ious >= thr) & (gt_cls == pred_cls[pred_idx]) & (~assigned))[0]
            if candidates.size == 0:
                continue
            best = candidates[np.argmax(ious[candidates])]
            assigned[best] = True
            tp[pred_idx, ti] = True
    return tp


def evaluate(
    pred_json: Path,
    data_yaml: Path,
    save_dir: Path,
    conf_thres: float = 0.001,
    iou_thres: float = 0.7,
    max_det: int = 300,
    agnostic_nms: bool = False,
    skip_nms: bool = False,
) -> Dict[str, float]:
    save_dir.mkdir(parents=True, exist_ok=True)
    image_paths, names = _read_data_yaml(data_yaml)
    pred_index = _load_predictions(pred_json)
    iou_thrs = np.arange(0.5, 0.96, 0.05)

    all_tp: List[np.ndarray] = []
    all_conf: List[np.ndarray] = []
    all_pred_cls: List[np.ndarray] = []
    all_target_cls: List[np.ndarray] = []
    per_image_rows: List[Dict[str, object]] = []

    for image_path in image_paths:
        with Image.open(image_path) as im:
            width, height = im.size
        _gt_boxes, gt_cls, gt_rboxes = _read_label_file(_label_path_for_image(image_path), width, height)
        pred_item = pred_index.get(image_path.name) or pred_index.get(image_path.stem)
        preds, pred_rboxes = _preds_for_image(pred_item)
        if preds.size:
            keep = preds[:, 4] >= conf_thres
            preds = preds[keep]
            pred_rboxes = pred_rboxes[keep]
            if not skip_nms:
                preds, pred_rboxes = _nms_rotated(preds, pred_rboxes, iou_thres, max_det, agnostic_nms)
        tp = _match_predictions(preds, pred_rboxes, gt_cls, gt_rboxes, iou_thrs)
        all_tp.append(tp)
        all_conf.append(preds[:, 4] if preds.size else np.zeros((0,), dtype=np.float32))
        all_pred_cls.append(preds[:, 5].astype(int) if preds.size else np.zeros((0,), dtype=np.int64))
        all_target_cls.append(gt_cls.astype(int))
        per_image_rows.append(
            {
                "image": str(image_path),
                "gt_count": int(gt_cls.shape[0]),
                "pred_count": int(preds.shape[0]),
            }
        )

    tp_all = np.concatenate(all_tp, axis=0) if all_tp else np.zeros((0, len(iou_thrs)), dtype=bool)
    conf_all = np.concatenate(all_conf, axis=0) if all_conf else np.zeros((0,), dtype=np.float32)
    pred_cls_all = np.concatenate(all_pred_cls, axis=0) if all_pred_cls else np.zeros((0,), dtype=np.int64)
    target_cls_all = np.concatenate(all_target_cls, axis=0) if all_target_cls else np.zeros((0,), dtype=np.int64)

    if target_cls_all.size == 0:
        raise RuntimeError("No ground-truth labels found for evaluation")

    result = ap_per_class(tp_all, conf_all, pred_cls_all, target_cls_all, plot=False, save_dir=save_dir, names=names)
    # Current Ultralytics returns:
    # tp, fp, p, r, f1, ap, unique_classes, p_curve, r_curve, f1_curve, x, prec_values.
    # Keep a defensive fallback for older forks where only the leading values exist.
    ap = result[5] if len(result) > 5 else result[3]
    unique_raw = result[6] if len(result) > 6 else np.unique(target_cls_all)
    unique_classes = np.asarray(unique_raw, dtype=int).reshape(-1)
    ap = np.asarray(ap, dtype=np.float32)
    if ap.ndim == 1:
        ap = ap[:, None]

    map50_by_class = ap[:, 0] if ap.shape[1] else np.zeros((ap.shape[0],), dtype=np.float32)
    map_by_class = ap.mean(axis=1) if ap.size else np.zeros((0,), dtype=np.float32)
    summary = {
        "images": float(len(image_paths)),
        "gt_count": float(target_cls_all.shape[0]),
        "pred_count": float(pred_cls_all.shape[0]),
        "mAP50": float(map50_by_class.mean()) if map50_by_class.size else 0.0,
        "mAP50-95": float(map_by_class.mean()) if map_by_class.size else 0.0,
    }

    with (save_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            writer.writerow([key, value])
    with (save_dir / "class_map50-95.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "mAP50-95", "AP50"])
        for idx, cls_id in enumerate(unique_classes.tolist()):
            writer.writerow([cls_id, names.get(int(cls_id), str(cls_id)), float(map_by_class[idx]), float(map50_by_class[idx])])
    with (save_dir / "per_image_counts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "gt_count", "pred_count"])
        writer.writeheader()
        writer.writerows(per_image_rows)

    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-json", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--save-dir", required=True, type=Path)
    parser.add_argument("--conf", type=float, default=0.001, help="confidence threshold (Ultralytics default for val)")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold (Ultralytics default)")
    parser.add_argument("--max-det", type=int, default=300, help="max detections per image")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--skip-nms", action="store_true", help="skip NMS (use when predictions already include NMS)")
    args = parser.parse_args()

    summary = evaluate(
        args.pred_json,
        args.data,
        args.save_dir,
        conf_thres=args.conf,
        iou_thres=args.iou,
        max_det=args.max_det,
        agnostic_nms=args.agnostic_nms,
        skip_nms=args.skip_nms,
    )
    print("DINO bypass prediction evaluation")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
