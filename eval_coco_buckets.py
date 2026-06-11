#!/usr/bin/env python3
"""Evaluate a YOLO model and export COCO-style mAP by size buckets, F1@0.50:0.95,
PR & F1 curves (CSV), and confusion matrix CSVs.

Usage:
  python eval_coco_buckets.py --weights ./runs/detect/exp/weights/best.pt --data data.yaml --save_dir ./eval_out

Notes:
 - Requires `ultralytics` package available in your PYTHONPATH (this workspace contains one).
 - Reads dataset labels in YOLO format (supports bbox xywh or OBB with 8 coords). Box coordinates are evaluated in original image resolution.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

from ultralytics import YOLO
from ultralytics.models.yolo.obb import OBBValidator
from ultralytics.utils import ops
from ultralytics.utils.metrics import box_iou, ap_per_class, batch_probiou, ConfusionMatrix, compute_ap


def read_yolo_label_file(
    lbl_path: Path, img_w: int, img_h: int, class_filter: set[int] | None = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return boxes (xyxy absolute), classes, areas, and rboxes (xywhr) for a label file.
    Supports 5-value xywh or 9-value (class + 8 coords) per-line formats.
    Areas are computed in original image coordinates (not scaled).
    """
    if not lbl_path.exists():
        return (
            np.zeros((0, 4)),
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=float),
            np.zeros((0, 5), dtype=float),
        )
    
    boxes = []
    classes = []
    areas = []
    rboxes = []
    with lbl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [float(x) for x in line.split()]
            cls = int(parts[0])
            if class_filter is not None and cls not in class_filter:
                continue
            if len(parts) == 5:
                # class, x_center, y_center, w, h (normalized)
                _, xc, yc, w, h = parts
                x1 = (xc - w / 2) * img_w
                y1 = (yc - h / 2) * img_h
                x2 = (xc + w / 2) * img_w
                y2 = (yc + h / 2) * img_h
                w_px = w * img_w
                h_px = h * img_h
                area = max(0.0, w_px * h_px)
                rbox = [xc * img_w, yc * img_h, w_px, h_px, 0.0]
            elif len(parts) == 9:
                # class + 8 coords (x1 y1 x2 y2 x3 y3 x4 y4) normalized
                coords = np.array(parts[1:]).reshape(4, 2)
                xs = coords[:, 0] * img_w
                ys = coords[:, 1] * img_h
                pts = np.stack([xs, ys], axis=1)
                x1, x2 = xs.min(), xs.max()
                y1, y2 = ys.min(), ys.max()
                # OBB native-space area: edge length (p0->p1) * edge length (p1->p2)
                edge1 = float(np.linalg.norm(pts[1] - pts[0]))
                edge2 = float(np.linalg.norm(pts[2] - pts[1]))
                area = max(0.0, edge1 * edge2)
                corners = pts.reshape(1, 8).astype(np.float32)
                rbox = ops.xyxyxyxy2xywhr(corners)[0].tolist()
            else:
                # Unknown format: skip
                continue
            boxes.append([x1, y1, x2, y2])
            classes.append(cls)
            areas.append(area)
            rboxes.append(rbox)
    if not boxes:
        return (
            np.zeros((0, 4)),
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=float),
            np.zeros((0, 5), dtype=float),
        )
    return (
        np.array(boxes, dtype=float),
        np.array(classes, dtype=int),
        np.array(areas, dtype=float),
        np.array(rboxes, dtype=float),
    )


def compute_image_predictions(
    model: YOLO,
    image_paths: List[Path],
    imgsz: int = 640,
    batch: int = 16,
    conf_thres: float = 0.001,
    iou_nms: float = 0.6,
    max_det: int = 300,
    class_filter: set[int] | None = None,
):
    """Run model.predict on a list of images and return preds per image.
    Returns:
      - preds_all: list of arrays Nx6: x1,y1,x2,y2,conf,cls
      - rboxes_all: list of arrays Nx5 (xywhr) or None
    """
    preds_all = []
    rboxes_all = []
    # Use model.predict for each image in batches
    for i in range(0, len(image_paths), batch):
        batch_paths = [str(p) for p in image_paths[i : i + batch]]
        results = model.predict(
            source=batch_paths,
            conf=conf_thres,
            iou=iou_nms,
            imgsz=imgsz,
            max_det=max_det,
            classes=sorted(class_filter) if class_filter is not None else None,
            verbose=False,
        )
        for r in results:
            # Support standard bbox (`r.boxes`) and oriented bbox (`r.obb`) results
            if getattr(r, 'boxes', None) is not None and r.boxes is not None and len(r.boxes):
                xyxy = r.boxes.xyxy.cpu().numpy()  # (n,4)
                conf = r.boxes.conf.cpu().numpy().reshape(-1, 1)
                cls = r.boxes.cls.cpu().numpy().reshape(-1, 1).astype(int)
                arr = np.concatenate([xyxy, conf, cls], axis=1)
                if class_filter is not None and arr.size:
                    arr = arr[np.isin(arr[:, 5].astype(int), list(class_filter))]
                preds_all.append(arr)
                rboxes_all.append(None)
                continue
            if getattr(r, 'obb', None) is not None and r.obb is not None and getattr(r.obb, 'xyxy', None) is not None:
                xyxy = r.obb.xyxy.cpu().numpy()  # (n,4) xyxy bounding rect for obb
                conf = r.obb.conf.cpu().numpy().reshape(-1, 1)
                cls = r.obb.cls.cpu().numpy().reshape(-1, 1).astype(int)
                arr = np.concatenate([xyxy, conf, cls], axis=1)
                rboxes = r.obb.xywhr.cpu().numpy() if getattr(r.obb, 'xywhr', None) is not None else None
                if class_filter is not None and arr.size:
                    keep = np.isin(arr[:, 5].astype(int), list(class_filter))
                    arr = arr[keep]
                    if rboxes is not None:
                        rboxes = rboxes[keep]
                preds_all.append(arr)
                rboxes_all.append(rboxes)
                continue
            preds_all.append(np.zeros((0, 6)))
            rboxes_all.append(None)
    return preds_all, rboxes_all


def prediction_areas(preds: np.ndarray, pred_rboxes: np.ndarray | None = None) -> np.ndarray:
    """Return prediction areas in original image pixels."""
    if preds.shape[0] == 0:
        return np.zeros((0,), dtype=float)
    if pred_rboxes is not None and len(pred_rboxes):
        rboxes = np.asarray(pred_rboxes, dtype=float)
        return np.maximum(0.0, np.abs(rboxes[:, 2] * rboxes[:, 3]))
    wh = np.maximum(0.0, preds[:, 2:4] - preds[:, 0:2])
    return wh[:, 0] * wh[:, 1]


def match_image_preds_to_gts(
    preds: np.ndarray,
    gts: np.ndarray,
    gt_cls: np.ndarray,
    iou_thrs: np.ndarray,
    pred_rboxes: np.ndarray | None = None,
    gt_rboxes: np.ndarray | None = None,
) -> np.ndarray:
    """Given preds Nx6 and gts Mx4, gt_cls M, compute tp matrix (N, T) booleans per IoU threshold.
    preds: x1,y1,x2,y2,conf,cls
    gts: x1,y1,x2,y2
    pred_rboxes/gt_rboxes: xywhr (radians) if OBB is available
    """
    if preds.shape[0] == 0:
        return np.zeros((0, len(iou_thrs)), dtype=bool)
    if gts.shape[0] == 0:
        return np.zeros((preds.shape[0], len(iou_thrs)), dtype=bool)

    # convert to torch
    pred_boxes = torch.tensor(preds[:, :4], dtype=torch.float32)
    pred_cls = preds[:, 5].astype(int)
    gt_boxes = torch.tensor(gts, dtype=torch.float32)
    gt_cls = gt_cls.astype(int)

    T = len(iou_thrs)
    tp = np.zeros((preds.shape[0], T), dtype=bool)

    # For each IoU threshold, perform greedy matching following COCO: preds sorted by conf desc
    order = np.argsort(-preds[:, 4])
    for ti, thr in enumerate(iou_thrs):
        assigned = np.zeros(len(gt_boxes), dtype=bool)
        for idx in order:
            if pred_rboxes is not None and gt_rboxes is not None:
                pb = torch.tensor(pred_rboxes[idx : idx + 1], dtype=torch.float32)
                gb = torch.tensor(gt_rboxes, dtype=torch.float32)
                ious = batch_probiou(gb, pb).cpu().numpy().ravel()
            else:
                pb = pred_boxes[idx].unsqueeze(0)  # 1x4
                ious = box_iou(pb, gt_boxes).cpu().numpy().ravel()
            # get best matching gt that is not assigned and same class
            candidates = np.where((ious >= thr) & (gt_cls == pred_cls[idx]) & (~assigned))[0]
            if candidates.size > 0:
                # choose gt with highest iou
                best = candidates[ious[candidates].argmax()]
                assigned[best] = True
                tp[idx, ti] = True
            else:
                tp[idx, ti] = False
    return tp


def match_image_preds_to_bucket_gts(
    preds: np.ndarray,
    gts: np.ndarray,
    gt_cls: np.ndarray,
    gt_in_bucket: np.ndarray,
    pred_in_bucket: np.ndarray,
    iou_thrs: np.ndarray,
    pred_rboxes: np.ndarray | None = None,
    gt_rboxes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Match predictions to GT with COCO-style area-bucket ignore handling.

    COCO area ranges are applied primarily to GT objects. Detections outside
    the area range are ignored only when they do not match a non-ignored GT.
    Therefore a detection whose predicted area falls outside the bucket can
    still be a TP if it correctly matches a GT inside the bucket.
    """
    if preds.shape[0] == 0:
        return (
            np.zeros((0, len(iou_thrs)), dtype=bool),
            np.zeros((0, len(iou_thrs)), dtype=bool),
        )
    if gts.shape[0] == 0:
        ignore = np.tile(~pred_in_bucket.reshape(-1, 1), (1, len(iou_thrs)))
        return np.zeros((preds.shape[0], len(iou_thrs)), dtype=bool), ignore

    pred_boxes = torch.tensor(preds[:, :4], dtype=torch.float32)
    pred_cls = preds[:, 5].astype(int)
    gt_boxes = torch.tensor(gts, dtype=torch.float32)
    gt_cls = gt_cls.astype(int)
    gt_in_bucket = gt_in_bucket.astype(bool)
    pred_in_bucket = pred_in_bucket.astype(bool)

    tp = np.zeros((preds.shape[0], len(iou_thrs)), dtype=bool)
    ignore = np.zeros((preds.shape[0], len(iou_thrs)), dtype=bool)
    order = np.argsort(-preds[:, 4])

    for ti, thr in enumerate(iou_thrs):
        assigned = np.zeros(len(gt_boxes), dtype=bool)
        for idx in order:
            if pred_rboxes is not None and gt_rboxes is not None:
                pb = torch.tensor(pred_rboxes[idx : idx + 1], dtype=torch.float32)
                gb = torch.tensor(gt_rboxes, dtype=torch.float32)
                ious = batch_probiou(gb, pb).cpu().numpy().ravel()
            else:
                pb = pred_boxes[idx].unsqueeze(0)
                ious = box_iou(pb, gt_boxes).cpu().numpy().ravel()

            candidates = np.where((ious >= thr) & (gt_cls == pred_cls[idx]) & (~assigned))[0]
            if candidates.size:
                valid = candidates[gt_in_bucket[candidates]]
                if valid.size:
                    best = valid[ious[valid].argmax()]
                    assigned[best] = True
                    tp[idx, ti] = True
                    continue

                ignored = candidates[~gt_in_bucket[candidates]]
                if ignored.size:
                    best = ignored[ious[ignored].argmax()]
                    assigned[best] = True
                    ignore[idx, ti] = True
                    continue

            # COCO ignores unmatched detections outside the active area range.
            ignore[idx, ti] = not pred_in_bucket[idx]

    return tp, ignore


def ap_per_class_with_ignore(
    tp: np.ndarray,
    ignore: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    eps: float = 1e-16,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute AP per class while ignoring detections per IoU threshold."""
    if target_cls.size == 0:
        return np.zeros((0, tp.shape[1])), np.zeros((0,), dtype=int)

    order = np.argsort(-conf)
    tp, ignore, pred_cls = tp[order], ignore[order], pred_cls[order]
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    ap = np.zeros((unique_classes.shape[0], tp.shape[1]))

    for ci, c in enumerate(unique_classes):
        class_mask = pred_cls == c
        n_l = nt[ci]
        if n_l == 0 or not class_mask.any():
            continue
        for j in range(tp.shape[1]):
            valid = class_mask & (~ignore[:, j])
            if not valid.any():
                continue
            tp_j = tp[valid, j].astype(float)
            fp_j = 1.0 - tp_j
            tpc = tp_j.cumsum()
            fpc = fp_j.cumsum()
            recall = tpc / (n_l + eps)
            precision = tpc / (tpc + fpc + eps)
            ap[ci, j], _, _ = compute_ap(recall, precision)

    return ap, unique_classes.astype(int)


class BucketOBBValidator(OBBValidator):
    """OBB validator that computes size-bucket AP from Ultralytics-native matches.

    The regular headline metrics already come from Ultralytics. This validator
    adds small/medium/large AP using the same scaled boxes and OBB matching
    path as the official validator, avoiding a second external label parser and
    matcher.
    """

    bucket_defs = {
        "small": (0.0, 1024.0),
        "medium": (1024.0, 9216.0),
        "large": (9216.0, float("inf")),
    }

    def init_metrics(self, model):
        super().init_metrics(model)
        self.bucket_stats = {
            name: {"tp": [], "ignore": [], "conf": [], "pred_cls": [], "target_cls": []}
            for name in self.bucket_defs
        }

    @staticmethod
    def _area_mask(area: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        if lo == 0.0:
            return area <= hi
        if hi == float("inf"):
            return area > lo
        return (area > lo) & (area <= hi)

    def _match_predictions_with_indices(self, pred_classes, true_classes, iou):
        """Return Ultralytics-style correct matrix plus matched GT index."""
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0]), dtype=bool)
        matched = np.full((pred_classes.shape[0], self.iouv.shape[0]), -1, dtype=int)
        correct_class = true_classes[:, None] == pred_classes
        iou = (iou * correct_class).cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            matches = np.nonzero(iou >= threshold)
            matches = np.array(matches).T
            if matches.shape[0]:
                if matches.shape[0] > 1:
                    matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                det_idx = matches[:, 1].astype(int)
                gt_idx = matches[:, 0].astype(int)
                correct[det_idx, i] = True
                matched[det_idx, i] = gt_idx
        return (
            torch.tensor(correct, dtype=torch.bool, device=pred_classes.device),
            torch.tensor(matched, dtype=torch.long, device=pred_classes.device),
        )

    def _process_batch_with_indices(self, detections, gt_bboxes, gt_cls):
        pred_bboxes = torch.cat([detections[:, :4], detections[:, -1:]], dim=-1)
        iou = batch_probiou(gt_bboxes, pred_bboxes)
        return self._match_predictions_with_indices(detections[:, 5], gt_cls, iou)

    def _process_dict_batch_with_indices(self, preds, batch):
        if batch["cls"].shape[0] == 0 or preds["cls"].shape[0] == 0:
            correct = torch.zeros((preds["cls"].shape[0], self.niou), dtype=torch.bool, device=self.device)
            matched = torch.full((preds["cls"].shape[0], self.niou), -1, dtype=torch.long, device=self.device)
            return correct, matched
        iou = batch_probiou(batch["bboxes"], preds["bboxes"])
        return self._match_predictions_with_indices(preds["cls"], batch["cls"], iou)

    def _append_bucket_stats_from_tensors(self, pred_bboxes, pred_conf, pred_cls, bbox, cls, matched):
        niou = self.iouv.shape[0]
        npr = len(pred_bboxes)
        pred_area = pred_bboxes[:, 2].clamp(min=0) * pred_bboxes[:, 3].clamp(min=0) if npr else pred_bboxes.new_zeros((0,))
        gt_area = bbox[:, 2].clamp(min=0) * bbox[:, 3].clamp(min=0) if len(cls) else bbox.new_zeros((0,))

        for name, (lo, hi) in self.bucket_defs.items():
            gt_in_bucket = self._area_mask(gt_area, lo, hi) if len(cls) else torch.zeros(0, dtype=torch.bool, device=self.device)
            pred_in_bucket = self._area_mask(pred_area, lo, hi) if npr else torch.zeros(0, dtype=torch.bool, device=self.device)
            target_cls = cls[gt_in_bucket] if len(cls) else cls

            if npr:
                tp = torch.zeros((npr, niou), dtype=torch.bool, device=self.device)
                ignore = torch.zeros((npr, niou), dtype=torch.bool, device=self.device)
                for ti in range(niou):
                    matched_t = matched[:, ti] if matched.numel() else torch.full((npr,), -1, dtype=torch.long, device=self.device)
                    has_match = matched_t >= 0
                    if has_match.any():
                        matched_in_bucket = torch.zeros(npr, dtype=torch.bool, device=self.device)
                        matched_in_bucket[has_match] = gt_in_bucket[matched_t[has_match]]
                        tp[:, ti] = has_match & matched_in_bucket
                        ignore[:, ti] = has_match & (~matched_in_bucket)
                    ignore[:, ti] |= (~has_match) & (~pred_in_bucket)

                self.bucket_stats[name]["tp"].append(tp)
                self.bucket_stats[name]["ignore"].append(ignore)
                self.bucket_stats[name]["conf"].append(pred_conf)
                self.bucket_stats[name]["pred_cls"].append(pred_cls)
            self.bucket_stats[name]["target_cls"].append(target_cls)

    def _append_bucket_stats(self, predn, bbox, cls, correct, matched):
        self._append_bucket_stats_from_tensors(predn[:, :4], predn[:, 4], predn[:, 5], bbox, cls, matched)

    def _update_metrics_dict_preds(self, preds, batch):
        """Bucket-aware path for newer Ultralytics validators that pass dict predictions."""
        for si, pred in enumerate(preds):
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)

            cls = pbatch["cls"]
            no_pred = predn["cls"].shape[0] == 0
            correct, matched = self._process_dict_batch_with_indices(predn, pbatch)
            self.metrics.update_stats(
                {
                    "tp": correct.cpu().numpy(),
                    "target_cls": cls.cpu().numpy(),
                    "target_img": np.unique(cls.cpu().numpy()),
                    "conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                    "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
                }
            )

            bucket_pred_bboxes = (
                self.scale_preds(predn, pbatch)["bboxes"] if not no_pred else predn["bboxes"]
            )
            bucket_gt_bboxes = (
                ops.scale_boxes(
                    pbatch["imgsz"],
                    pbatch["bboxes"].clone(),
                    pbatch["ori_shape"],
                    ratio_pad=pbatch["ratio_pad"],
                    xywh=True,
                )
                if len(cls)
                else pbatch["bboxes"]
            )
            self._append_bucket_stats_from_tensors(
                bucket_pred_bboxes,
                predn["conf"] if not no_pred else predn["bboxes"].new_zeros((0,)),
                predn["cls"] if not no_pred else predn["bboxes"].new_zeros((0,)),
                bucket_gt_bboxes,
                cls,
                matched,
            )

            if self.args.plots:
                self.confusion_matrix.process_batch(predn, pbatch, conf=self.args.conf)
                if getattr(self.args, "visualize", False):
                    self.confusion_matrix.plot_matches(batch["img"][si], pbatch["im_file"], self.save_dir)

            if no_pred:
                continue

            if self.args.save_json or self.args.save_txt:
                predn_scaled = self.scale_preds(predn, pbatch)
            if self.args.save_json:
                self.pred_to_json(predn_scaled, pbatch)
            if self.args.save_txt:
                self.save_one_txt(
                    predn_scaled,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(pbatch['im_file']).stem}.txt",
                )

    def update_metrics(self, preds, batch):
        """Same as OBBValidator.update_metrics, with bucket stat collection."""
        if preds and isinstance(preds[0], dict):
            return self._update_metrics_dict_preds(preds, batch)

        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                self._append_bucket_stats(pred.new_zeros((0, 7)), bbox, cls, stat["tp"], pred.new_full((0, self.niou), -1, dtype=torch.long))
                continue

            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            matched = predn.new_full((npr, self.niou), -1, dtype=torch.long)
            if nl:
                stat["tp"], matched = self._process_batch_with_indices(predn, bbox, cls)
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, bbox, cls)
            self._append_bucket_stats(predn, bbox, cls, stat["tp"], matched)

            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                file = self.save_dir / "labels" / f'{Path(batch["im_file"][si]).stem}.txt'
                self.save_one_txt(predn, self.args.save_conf, pbatch["ori_shape"], file)

    def get_stats(self):
        result = super().get_stats()
        bucket_results = {}
        bucket_class_ap = {}
        bucket_gt_count_by_class = {}
        for name, stats in self.bucket_stats.items():
            target_cls = torch.cat(stats["target_cls"], 0).cpu().numpy().astype(int) if stats["target_cls"] else np.zeros((0,), dtype=int)
            gt_counts = {int(c): int((target_cls == c).sum()) for c in np.unique(target_cls)}
            bucket_gt_count_by_class[name] = gt_counts
            if not stats["conf"] or target_cls.size == 0:
                bucket_results[name] = {"ap": 0.0}
                bucket_class_ap[name] = {}
                continue
            tp = torch.cat(stats["tp"], 0).cpu().numpy().astype(bool)
            ignore = torch.cat(stats["ignore"], 0).cpu().numpy().astype(bool)
            conf = torch.cat(stats["conf"], 0).cpu().numpy()
            pred_cls = torch.cat(stats["pred_cls"], 0).cpu().numpy().astype(int)
            ap_array, unique_classes = ap_per_class_with_ignore(tp, ignore, conf, pred_cls, target_cls)
            ap_by_class = {int(c): float(ap_array[i].mean()) for i, c in enumerate(unique_classes.tolist())} if ap_array.size else {}
            ap50_by_class = {int(c): float(ap_array[i, 0]) for i, c in enumerate(unique_classes.tolist())} if ap_array.size else {}
            bucket_results[name] = {"ap": float(ap_array.mean()) if ap_array.size else 0.0}
            bucket_class_ap[name] = ap_by_class
            bucket_results[name]["ap50_by_class"] = ap50_by_class
        self.metrics.bucket_results = bucket_results
        self.metrics.bucket_class_ap = bucket_class_ap
        self.metrics.bucket_gt_count_by_class = bucket_gt_count_by_class
        return result


def evaluate_dataset(
    weights: str,
    data_yaml: Path,
    save_dir: Path,
    imgsz: int = 640,
    iou_thrs: np.ndarray = None,
    conf_thres: float = 0.001,
    iou_nms: float = 0.6,
    max_det: int = 300,
    class_filter: set[int] | None = None,
    plots: bool = False,
):
    save_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(weights)

    if iou_thrs is None:
        iou_thrs = np.arange(0.5, 0.96, 0.05)

    # read data yaml
    import yaml

    with open(data_yaml) as f:
        data = yaml.safe_load(f)
    if class_filter is not None:
        print(f"Evaluating class filter: {sorted(class_filter)}")

    names_data = data.get('names', {})

    def class_name(cls_id: int) -> str:
        if isinstance(names_data, dict):
            name = str(names_data.get(cls_id, names_data.get(str(cls_id), cls_id)))
            return name.lower() if name.lower() in {'complete', 'incomplete'} else name
        if isinstance(names_data, (list, tuple)) and 0 <= int(cls_id) < len(names_data):
            name = str(names_data[int(cls_id)])
            return name.lower() if name.lower() in {'complete', 'incomplete'} else name
        return str(cls_id)
    # Resolve base path if provided (Ultralytics-style `path:`)
    base = Path(data.get('path', '.'))
    val_images_dir = Path(data.get('val') or data.get('test') or data.get('images'))
    if not val_images_dir.is_absolute():
        val_images_dir = base / val_images_dir

    if val_images_dir.is_file() and val_images_dir.suffix in {'.txt', '.csv'}:
        # file list
        with open(val_images_dir) as f:
            image_paths = [Path(l.strip()) for l in f if l.strip()]
    else:
        image_paths = sorted([p for p in val_images_dir.rglob('*.jpg')] + [p for p in val_images_dir.rglob('*.png')])
    if len(image_paths) == 0:
        raise SystemExit(f'No images found in {val_images_dir}')

    # build image->label mapping
    # Common layouts:
    # - base/images/<split>/...  and base/labels/<split>/...
    # - base/images/<split>/...  and base/images/labels/<split>/...
    split = val_images_dir.name
    labels_dir_candidates = []
    if 'labels' in data:
        labels_dir_candidates.append(Path(data['labels']))
    labels_dir_candidates.append(base / 'labels' / split)
    labels_dir_candidates.append(base / 'labels')
    labels_dir_candidates.append(val_images_dir.parent / 'labels' / split)
    labels_dir_candidates.append(val_images_dir.parent / 'labels')
    labels_dir = None
    for cand in labels_dir_candidates:
        if cand and cand.exists():
            labels_dir = cand
            break
    if labels_dir is None:
        # fallback: sibling 'labels' next to images
        labels_dir = val_images_dir.parent / 'labels'

    # Run official Ultralytics validation once and use these as the headline metrics.
    # This keeps overall mAP/mAP50 consistent with `model.val(...)` in valid_oscar.py.
    try:
        val_kwargs = dict(
            data=str(data_yaml),
            imgsz=imgsz,
            conf=conf_thres,
            iou=iou_nms,
            max_det=max_det,
            classes=sorted(class_filter) if class_filter is not None else None,
            verbose=False,
            plots=plots,
        )
        if getattr(model, "task", "") == "obb":
            val_metrics = model.val(validator=BucketOBBValidator, **val_kwargs)
        else:
            val_metrics = model.val(**val_kwargs)
    except KeyError as exc:
        if not plots:
            raise
        print(f"Warning: Ultralytics plot export failed for missing class name {exc}; retrying validation with plots=False.")
        val_kwargs["plots"] = False
        if getattr(model, "task", "") == "obb":
            val_metrics = model.val(validator=BucketOBBValidator, **val_kwargs)
        else:
            val_metrics = model.val(**val_kwargs)

    # collect GT classes (flatten)
    all_gt_classes = []
    # store per-image GT boxes and classes
    gts_per_image = []
    for p in image_paths:
        img = Image.open(p)
        w, h = img.size
        rel_lbl = p.relative_to(val_images_dir).with_suffix('.txt')
        lbl = labels_dir / rel_lbl
        if not lbl.exists():
            # fallback for flat-label layouts
            lbl = labels_dir / f"{p.stem}.txt"
        boxes, classes, areas, rboxes = read_yolo_label_file(lbl, w, h, class_filter=class_filter)
        gts_per_image.append((boxes, classes, areas, rboxes))
        all_gt_classes.extend(classes.tolist())

    native_bucket_results = getattr(val_metrics, "bucket_results", None)
    native_bucket_class_ap = getattr(val_metrics, "bucket_class_ap", None)
    native_bucket_counts = getattr(val_metrics, "bucket_gt_count_by_class", None)

    gt_count_by_class = {int(c): int(all_gt_classes.count(c)) for c in set(all_gt_classes)}
    metric_by_class = {}
    report_classes = sorted(class_filter) if class_filter is not None else sorted(gt_count_by_class)

    # Export raw curve arrays directly from the box metric object.
    # This is more reliable than relying on the higher-level curves_results wrapper.
    box_metric = getattr(val_metrics, 'box', None)
    if box_metric is not None:
        # Avoid val_metrics.box.maps here: older Ultralytics versions index this
        # array by class id and can fail when the model predicts class ids not
        # present in the dataset names, e.g. a 3-class model evaluated on a
        # 2-class target yaml. ap_class_index/ap are already aligned.
        ap_class_index = np.asarray(getattr(box_metric, 'ap_class_index', []), dtype=int)
        ap_values = np.asarray(getattr(box_metric, 'ap', []), dtype=float)
        if ap_class_index.size and ap_values.size:
            with open(save_dir / 'class_map50-95.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['class', 'mAP50-95'])
                for cls_id, value in zip(ap_class_index.tolist(), ap_values.tolist()):
                    writer.writerow([int(cls_id), float(value)])

            p_values = np.asarray(getattr(box_metric, 'p', []), dtype=float)
            r_values = np.asarray(getattr(box_metric, 'r', []), dtype=float)
            ap50_values = np.asarray(getattr(box_metric, 'ap50', []), dtype=float)
            for idx, cls_id in enumerate(ap_class_index.tolist()):
                metric_by_class[int(cls_id)] = {
                    'P': float(p_values[idx]) if idx < p_values.size else 0.0,
                    'R': float(r_values[idx]) if idx < r_values.size else 0.0,
                    'mAP50': float(ap50_values[idx]) if idx < ap50_values.size else 0.0,
                    'mAP50-95': float(ap_values[idx]) if idx < ap_values.size else 0.0,
                }
            report_classes = sorted(class_filter) if class_filter is not None else sorted(set(gt_count_by_class) | set(metric_by_class))
            with open(save_dir / 'class_metrics.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['class', 'name', 'images', 'instances', 'P', 'R', 'F1', 'mAP50', 'mAP50-95'])
                for cls_id in report_classes:
                    metrics = metric_by_class.get(int(cls_id), {'P': 0.0, 'R': 0.0, 'mAP50': 0.0, 'mAP50-95': 0.0})
                    f1_value = 2.0 * metrics['P'] * metrics['R'] / (metrics['P'] + metrics['R'] + 1e-16)
                    writer.writerow([
                        int(cls_id),
                        class_name(int(cls_id)),
                        len(image_paths),
                        int(gt_count_by_class.get(int(cls_id), 0)),
                        metrics['P'],
                        metrics['R'],
                        float(f1_value),
                        metrics['mAP50'],
                        metrics['mAP50-95'],
                    ])

        curve_exports = [
            ('pr_curve.csv', getattr(box_metric, 'px', None), getattr(box_metric, 'prec_values', None), 'recall', 'precision'),
            ('f1_curve.csv', getattr(box_metric, 'px', None), getattr(box_metric, 'f1_curve', None), 'confidence', 'f1'),
            ('precision_curve.csv', getattr(box_metric, 'px', None), getattr(box_metric, 'p_curve', None), 'confidence', 'precision'),
            ('recall_curve.csv', getattr(box_metric, 'px', None), getattr(box_metric, 'r_curve', None), 'confidence', 'recall'),
        ]
        for filename, x, y, x_title, y_title in curve_exports:
            if x is None or y is None:
                continue
            x = np.asarray(x)
            y = np.asarray(y)
            y_mean = y.mean(axis=0) if y.ndim > 1 else y
            with open(save_dir / filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([x_title, y_title])
                for xi, yi in zip(x.tolist(), y_mean.tolist()):
                    writer.writerow([xi, yi])

    # Backward-compatible export from curves_results if present.
    curve_rows = getattr(val_metrics, 'curves_results', [])
    curve_names = getattr(val_metrics, 'curves', [])
    for curve_name, curve_values in zip(curve_names, curve_rows):
        x, y, x_title, y_title = curve_values
        x = np.asarray(x)
        y = np.asarray(y)
        y_mean = y.mean(axis=0) if y.ndim > 1 else y
        safe_name = curve_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        with open(save_dir / f'{safe_name}.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([x_title.lower(), y_title.lower()])
            for xi, yi in zip(x.tolist(), y_mean.tolist()):
                writer.writerow([xi, yi])

    confusion = getattr(val_metrics, 'confusion_matrix', None)
    if confusion is not None:
        np.savetxt(save_dir / 'confusion_matrix.csv', confusion.matrix, fmt='%d', delimiter=',')
    else:
        print('Warning: confusion_matrix is not available. Use --plots to enable it in model.val().')

    # Compute COCO-size bucket APs by re-evaluating per-bucket.
    # All boxes stay in original image resolution; the model predictions returned by
    # Ultralytics are already mapped back to original coordinates.
    image_areas_per_image = [areas for (_, _, areas, _) in gts_per_image]
    # Fixed COCO thresholds in original-resolution pixel area.
    # Do not rescale to 640; this keeps the evaluation aligned with COCO/AP size buckets.
    small_thr = 32 * 32
    med_thr = 96 * 96
    buckets = {
        'small': lambda a: a <= small_thr,
        'medium': lambda a: (a > small_thr) & (a <= med_thr),
        'large': lambda a: a > med_thr,
    }
    print(f"\n=== Area Thresholds (COCO standard, original resolution) ===")
    print(f"small: ≤ {small_thr} px | medium: ({small_thr}, {med_thr}] | large: > {med_thr} px")

    bucket_results = {}
    bucket_stats = {}  # for debug: track GT/pred counts per bucket
    bucket_class_ap = {}
    if native_bucket_results is not None:
        bucket_results = native_bucket_results
        bucket_class_ap = native_bucket_class_ap or {}
        for name in ["small", "medium", "large"]:
            counts = (native_bucket_counts or {}).get(name, {})
            bucket_stats[name] = {
                "gt_count": int(sum(counts.values())),
                "gt_count_by_class": counts,
                "pred_count": "",
            }
            with open(save_dir / f'bucket_{name}_ap_per_class.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['class', 'gt_count', 'mAP50-95', 'AP50'])
                bucket_report_classes = sorted(class_filter) if class_filter is not None else sorted(set(counts) | set(bucket_class_ap.get(name, {})))
                ap50_by_class = bucket_results.get(name, {}).get("ap50_by_class", {})
                for cls_id in bucket_report_classes:
                    writer.writerow([
                        int(cls_id),
                        int(counts.get(int(cls_id), 0)),
                        bucket_class_ap.get(name, {}).get(int(cls_id), ''),
                        ap50_by_class.get(int(cls_id), ''),
                    ])
    else:
        preds_per_image, pred_rboxes_per_image = compute_image_predictions(
            model,
            image_paths,
            imgsz=imgsz,
            conf_thres=conf_thres,
            iou_nms=iou_nms,
            max_det=max_det,
            class_filter=class_filter,
        )
        for name, fn in buckets.items():
        # COCO-style area range: GT outside this bucket are ignored. Unmatched
        # detections outside the active area range are ignored, but detections
        # that match a GT inside the bucket may still count as TP even if their
        # predicted area falls outside the bucket.
            all_bucket_gt_classes = []
            bucket_gt_count_by_class = {}
            confs_b = []
            pred_cls_b = []
            tp_rows_b = []
            ignore_rows_b = []
            bucket_gt_count = 0
            for preds, pred_rboxes, (gts, gcls, _, gt_rboxes), areas in zip(
                preds_per_image, pred_rboxes_per_image, gts_per_image, image_areas_per_image
            ):
                gt_in_bucket = fn(areas) if areas.size else np.array([], dtype=bool)
                bucket_gcls = gcls[gt_in_bucket] if gt_in_bucket.size else np.zeros((0,), dtype=int)
                bucket_gt_count += int(gt_in_bucket.sum())
                for cls_id in bucket_gcls.tolist():
                    bucket_gt_count_by_class[int(cls_id)] = bucket_gt_count_by_class.get(int(cls_id), 0) + 1

                pred_areas = prediction_areas(preds, pred_rboxes)
                pred_in_bucket = fn(pred_areas) if pred_areas.size else np.array([], dtype=bool)

                tp_b, ignore_b = match_image_preds_to_bucket_gts(
                    preds,
                    gts,
                    gcls,
                    gt_in_bucket,
                    pred_in_bucket,
                    iou_thrs,
                    pred_rboxes=pred_rboxes,
                    gt_rboxes=gt_rboxes,
                )
                if preds.shape[0] > 0:
                    confs_b.append(preds[:, 4])
                    pred_cls_b.append(preds[:, 5].astype(int))
                    tp_rows_b.append(tp_b)
                    ignore_rows_b.append(ignore_b)
                all_bucket_gt_classes.extend(bucket_gcls.tolist())

            bucket_stats[name] = {
                'gt_count': bucket_gt_count,
                'gt_count_by_class': bucket_gt_count_by_class,
                'pred_count': sum(p.shape[0] for p in preds_per_image) if preds_per_image else 0,
            }

            if len(confs_b) == 0:
                # no detections
                bucket_results[name] = {'ap': 0.0}
                bucket_class_ap[name] = {}
                continue
            conf_b = np.concatenate(confs_b, axis=0)
            pred_cls_b = np.concatenate(pred_cls_b, axis=0)
            tp_b = np.concatenate(tp_rows_b, axis=0).astype(int)
            ignore_b = np.concatenate(ignore_rows_b, axis=0).astype(bool)
            target_cls_b = np.array(all_bucket_gt_classes, dtype=int)
            if target_cls_b.size == 0:
                bucket_results[name] = {'ap': 0.0}
                bucket_class_ap[name] = {}
                continue
            tp_bool_b = tp_b.astype(bool)
            ap_array, unique_classes = ap_per_class_with_ignore(
                tp_bool_b, ignore_b, conf_b, pred_cls_b, target_cls_b
            )
            ap_by_class = {}
            # save per-class AP for this bucket for debugging
            if ap_array.size:
                # mean across IoU for mAP50-95, and AP50 is first column
                ap_mean_per_class = ap_array.mean(axis=1)
                ap50_per_class = ap_array[:, 0]
                with open(save_dir / f'bucket_{name}_ap_per_class.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['class', 'gt_count', 'mAP50-95', 'AP50'])
                    ap_by_class = {int(c): float(ap_mean_per_class[cls_idx]) for cls_idx, c in enumerate(unique_classes.tolist())}
                    ap50_by_class = {int(c): float(ap50_per_class[cls_idx]) for cls_idx, c in enumerate(unique_classes.tolist())}
                    bucket_report_classes = sorted(class_filter) if class_filter is not None else sorted(set(unique_classes.tolist()))
                    for cls_id in bucket_report_classes:
                        writer.writerow([
                            int(cls_id),
                            int(bucket_gt_count_by_class.get(int(cls_id), 0)),
                            ap_by_class.get(int(cls_id), ''),
                            ap50_by_class.get(int(cls_id), ''),
                        ])
            # mean across classes
            ap_mean = ap_array.mean() if ap_array.size else 0.0
            bucket_results[name] = {'ap': float(ap_mean), 'ap_per_class': ap_array.tolist()}
            bucket_class_ap[name] = ap_by_class

    # summary CSV
    map50 = float(val_metrics.box.map50)
    precision50 = float(val_metrics.box.mp)
    recall50 = float(val_metrics.box.mr)
    f1_50 = 2.0 * precision50 * recall50 / (precision50 + recall50 + 1e-16)

    summary = {
        'mAP@0.50:0.95': {'all': float(val_metrics.box.map)},
        'mAP@0.50': {'all': map50},
        'AP_small': {'all': bucket_results['small']['ap']},
        'AP_medium': {'all': bucket_results['medium']['ap']},
        'AP_large': {'all': bucket_results['large']['ap']},
        'F1@0.50': {'all': float(f1_50)},
        'Precision@0.50': {'all': precision50},
        'Recall@0.50': {'all': recall50},
        'TP': {'all': 0},
        'FP': {'all': 0},
        'FN': {'all': 0},
    }
    for cls_id in report_classes:
        cls_name = class_name(int(cls_id))
        cls_metrics = metric_by_class.get(int(cls_id), {'P': 0.0, 'R': 0.0, 'mAP50': 0.0, 'mAP50-95': 0.0})
        cls_p = float(cls_metrics.get('P', 0.0))
        cls_r = float(cls_metrics.get('R', 0.0))
        cls_f1 = 2.0 * cls_p * cls_r / (cls_p + cls_r + 1e-16)
        cls_gt = int(gt_count_by_class.get(int(cls_id), 0))
        cls_tp = int(round(cls_r * cls_gt))
        cls_fn = max(0, cls_gt - cls_tp)
        cls_fp = int(round(cls_tp / cls_p - cls_tp)) if cls_p > 0 else 0
        summary['mAP@0.50:0.95'][cls_name] = cls_metrics.get('mAP50-95', '')
        summary['mAP@0.50'][cls_name] = cls_metrics.get('mAP50', '')
        summary['AP_small'][cls_name] = bucket_class_ap.get('small', {}).get(int(cls_id), '')
        summary['AP_medium'][cls_name] = bucket_class_ap.get('medium', {}).get(int(cls_id), '')
        summary['AP_large'][cls_name] = bucket_class_ap.get('large', {}).get(int(cls_id), '')
        summary['F1@0.50'][cls_name] = float(cls_f1)
        summary['Precision@0.50'][cls_name] = cls_p
        summary['Recall@0.50'][cls_name] = cls_r
        summary['TP'][cls_name] = cls_tp
        summary['FP'][cls_name] = max(0, cls_fp)
        summary['FN'][cls_name] = cls_fn
        summary['TP']['all'] += cls_tp
        summary['FP']['all'] += max(0, cls_fp)
        summary['FN']['all'] += cls_fn
    with open(save_dir / 'summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        columns = ['metric', 'all'] + [class_name(int(cls_id)) for cls_id in report_classes]
        writer.writerow(columns)
        for metric_name, values in summary.items():
            writer.writerow([metric_name] + [values.get(col, '') for col in columns[1:]])

    # Debug: print bucket stats
    print('\n=== Bucket Statistics ===')
    for name in ['small', 'medium', 'large']:
        stats = bucket_stats.get(name, {})
        result = bucket_results.get(name, {})
        print(f"{name:8s}: GT_count={stats.get('gt_count', 0):4d}  AP={result.get('ap', 0):.4f}")

    print('Saved evaluation outputs to', save_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--data', required=True, help='data yaml pointing to dataset (val)')
    parser.add_argument('--save_dir', default='eval_out')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf', type=float, default=0.001, help='confidence threshold for custom prediction pass')
    parser.add_argument('--iou_nms', type=float, default=0.7, help='NMS IoU for custom prediction pass')
    parser.add_argument('--max_det', type=int, default=300, help='maximum detections per image')
    parser.add_argument('--classes', type=str, default=None, help='comma-separated class ids to evaluate, e.g. "0,1"')
    parser.add_argument('--plots', action='store_true', help='enable plots in model.val() (produces confusion matrix)')
    args = parser.parse_args()
    class_filter = None
    if args.classes:
        class_filter = {int(x.strip()) for x in str(args.classes).split(',') if x.strip()}
    evaluate_dataset(
        args.weights,
        Path(args.data),
        Path(args.save_dir),
        imgsz=args.imgsz,
        conf_thres=args.conf,
        iou_nms=args.iou_nms,
        max_det=args.max_det,
        class_filter=class_filter,
        plots=args.plots,
    )


if __name__ == '__main__':
    main()
