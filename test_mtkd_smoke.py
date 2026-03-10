#!/usr/bin/env python3
"""
MTKD v2 — Smoke Test
=====================
Tests each component incrementally to find issues before full training.

Usage:
    cd /home/oscar/Poaceae-Stomata-Detection
    source Thesis/bin/activate
    python test_mtkd_smoke.py
"""

import sys
import os
import torch
import traceback

sys.path.insert(0, os.path.dirname(__file__))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DINO_CKPT = "dinov3-main/weight folder/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
DATASET_ROOT = "Stomata_Dataset"
IMAGE_SUBDIR = "barley_category/barley_image_fresh-leaf"
LABEL_SUBDIR = "barley_category/barley_label_fresh-leaf"

passed = 0
failed = 0


def report(name, ok, msg=""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name}: {msg}")


# ===========================================================================
print("\n" + "=" * 60)
print("STEP 1: Data Pipeline")
print("=" * 60)

try:
    from mtkd_framework.data import (
        StomataBarleyDataset,
        build_stomata_dataloaders,
        collate_stomata_batch,
    )

    train_loader, val_loader = build_stomata_dataloaders(
        dataset_root=DATASET_ROOT,
        image_subdir=IMAGE_SUBDIR,
        label_subdir=LABEL_SUBDIR,
        image_size=640,
        val_ratio=0.15,
        batch_size=2,
        num_workers=0,
    )
    report("DataLoader created", True)

    batch = next(iter(train_loader))
    images = batch["images"]
    targets = batch["targets"]
    image_paths = batch["image_paths"]

    report(f"Batch loaded: images {images.shape}", True)
    report(f"Image range [{images.min():.2f}, {images.max():.2f}]",
           images.min() >= 0 and images.max() <= 1.01,
           f"Expected [0, 1], got [{images.min():.2f}, {images.max():.2f}]")
    report(f"Targets: boxes {targets['boxes'].shape}, labels {targets['labels'].shape}",
           targets["boxes"].ndim == 3)
    report(f"Valid mask present", "valid_mask" in targets)
    report(f"Image paths: {len(image_paths)} paths", len(image_paths) == images.shape[0])

    print(f"\n  Sample image path: {image_paths[0]}")
    n_boxes = targets["valid_mask"][0].sum().item()
    print(f"  Sample: {n_boxes} GT boxes in first image")
    if n_boxes > 0:
        print(f"  First box (cxcywh): {targets['boxes'][0, 0]}")
except Exception as e:
    report("Data pipeline", False, str(e))
    traceback.print_exc()

# ===========================================================================
print("\n" + "=" * 60)
print("STEP 2: Pseudo-Label Utilities")
print("=" * 60)

try:
    from mtkd_framework.engine.pseudo_labels import targets_to_yolo_batch

    gt_batch = targets_to_yolo_batch(targets, DEVICE)
    report(f"targets_to_yolo_batch OK", True)
    report(f"  batch_idx: {gt_batch['batch_idx'].shape}", True)
    report(f"  cls: {gt_batch['cls'].shape}", True)
    report(f"  bboxes: {gt_batch['bboxes'].shape}", True)
    n = gt_batch["bboxes"].shape[0]
    print(f"  Total GT boxes in batch: {n}")
    if n > 0:
        print(f"  First bbox: {gt_batch['bboxes'][0]}")
        print(f"  First cls:  {gt_batch['cls'][0]}")
except Exception as e:
    report("Pseudo-label utils", False, str(e))
    traceback.print_exc()

# ===========================================================================
print("\n" + "=" * 60)
print("STEP 3: YOLO Student — forward_train_raw + compute_loss")
print("=" * 60)

try:
    from mtkd_framework.models.yolo_wrappers import YOLOStudentDetector

    student = YOLOStudentDetector(
        weights="yolo11s.pt",
        dino_teacher_dim=768,
        feature_level="p4",
        num_classes=1,
    ).to(DEVICE)
    report("YOLOStudentDetector created", True)

    imgs = images.to(DEVICE)
    raw_preds, feats = student.forward_train_raw(imgs)
    report(f"forward_train_raw OK", True)
    print(f"  raw_preds type: {type(raw_preds)}")
    if isinstance(raw_preds, torch.Tensor):
        print(f"  raw_preds shape: {raw_preds.shape}")
    elif isinstance(raw_preds, (list, tuple)):
        for i, rp in enumerate(raw_preds):
            print(f"  raw_preds[{i}] shape: {rp.shape}")
    print(f"  Features captured: {list(feats.keys())}")
    for k, v in feats.items():
        print(f"    {k}: {v.shape}")

    # Compute loss
    det_loss, det_items = student.compute_loss(raw_preds, gt_batch)
    report(f"compute_loss OK: loss={det_loss.item():.4f}", True)
    print(f"  loss_items (box, cls, dfl): {det_items}")
    report("Loss has grad_fn", det_loss.requires_grad)

except Exception as e:
    report("YOLO Student", False, str(e))
    traceback.print_exc()

# ===========================================================================
print("\n" + "=" * 60)
print("STEP 4: DINO Feature Extractor (frozen)")
print("=" * 60)

try:
    from mtkd_framework.engine.build_dino import DinoFeatureExtractor

    dino = DinoFeatureExtractor(
        model_name="vit_base",
        patch_size=16,
        embed_dim=768,
        normalize_feature=True,
        pretrained_path=os.path.join(os.path.dirname(__file__), DINO_CKPT),
    ).to(DEVICE)
    report("DinoFeatureExtractor created", True)

    # NOTE: dataset outputs [0, 1], DINO preprocessing expects [0, 255]
    # Check if we need to scale
    dino_input = imgs  # [0, 1] range
    dino_feat = dino(dino_input)
    report(f"DINO forward OK: output {dino_feat.shape}", True)

    expected_h = 640 // 16  # = 40
    report(f"Spatial dims: {dino_feat.shape[2]}x{dino_feat.shape[3]} (expected {expected_h}x{expected_h})",
           dino_feat.shape[2] == expected_h)

    # Check feature quality: with wrong normalization features may be degenerate
    feat_mean = dino_feat.mean().item()
    feat_std = dino_feat.std().item()
    feat_norm = dino_feat.norm(dim=1).mean().item()
    print(f"  Feature stats: mean={feat_mean:.4f}, std={feat_std:.4f}, norm={feat_norm:.4f}")
    report("Feature std > 0.001 (not degenerate)", feat_std > 0.001,
           f"std={feat_std:.6f} — may indicate wrong input normalization")

except Exception as e:
    report("DINO Feature Extractor", False, str(e))
    traceback.print_exc()

# ===========================================================================
print("\n" + "=" * 60)
print("STEP 5: FFT Block injection")
print("=" * 60)

try:
    from mtkd_framework.engine.pluggable_fft_block import inject_fft_blocks, PluggableFFTBlock

    # Test on standalone DINO encoder
    fft_blocks = inject_fft_blocks(
        dino.encoder,
        after_blocks=[9],
        embed_dim=768,
        n_storage_tokens=0,
        num_freq_bins=32,
        hidden_dim=256,
        init_gate=-5.0,
    )
    report(f"FFT block injected: {len(fft_blocks)} block(s)", len(fft_blocks) == 1)
    print(f"  Total blocks after injection: {len(dino.encoder.blocks)}")
    print(f"  Gate value: {fft_blocks[0].get_gate_value():.4f}")

    # Test forward after injection
    dino_feat2 = dino(imgs)
    report(f"DINO forward after FFT injection OK: {dino_feat2.shape}", True)

    # Check that FFT block barely changed output (zero-init gate)
    diff = (dino_feat2 - dino_feat).abs().mean().item()
    print(f"  Feature diff (should be small due to zero-init gate): {diff:.6f}")
    report("Near-identity at init", diff < 0.1, f"diff={diff:.6f}")

except Exception as e:
    report("FFT Block", False, str(e))
    traceback.print_exc()

# ===========================================================================
print("\n" + "=" * 60)
print("STEP 6: Alignment Head")
print("=" * 60)

try:
    from mtkd_framework.engine.align_head import TeacherStudentAlignHead

    student_feat = feats["p4_features"].to(DEVICE)  # from YOLO
    print(f"  Student P4 feat: {student_feat.shape}")
    print(f"  DINO feat: {dino_feat2.shape}")

    align_head = TeacherStudentAlignHead(
        student_dim=student_feat.shape[1],
        teacher_dim=768,
        head_type="MLP",
        proj_dim=1024,
        normalize=True,
    ).to(DEVICE)
    report("AlignHead created", True)

    projected = align_head(student_feat, dino_feat2.shape[2:])
    report(f"Projection OK: {projected.shape}", projected.shape == dino_feat2.shape)

    align_loss = align_head.align_loss(projected, dino_feat2)
    report(f"Alignment loss: {align_loss.item():.4f}", True)
    report("Align loss has grad_fn", align_loss.requires_grad)

except Exception as e:
    report("Alignment Head", False, str(e))
    traceback.print_exc()

# ===========================================================================
print("\n" + "=" * 60)
print("STEP 7: Full MTKDModelV2 (integrated)")
print("=" * 60)

try:
    from mtkd_framework.models.mtkd_model_v2 import build_mtkd_model_v2

    model_config = {
        "num_classes": 1,
        "student_config": {
            "student_type": "yolo",
            "weights": "yolo11s.pt",
            "feature_level": "p4",
        },
        "dino_config": {
            "model_name": "vit_base",
            "patch_size": 16,
            "embed_dim": 768,
            "normalize_feature": True,
        },
        "dino_checkpoint": os.path.join(os.path.dirname(__file__), DINO_CKPT),
        "align_head_config": {
            "head_type": "MLP",
            "proj_dim": 1024,
            "normalize": True,
        },
        "student_align_layer": "p4",
        "fft_block_config": {
            "after_blocks": [9],
            "num_freq_bins": 32,
            "hidden_dim": 256,
            "init_gate": -5.0,
            "modulation_mode": "multiplicative",
        },
    }

    model = build_mtkd_model_v2(model_config)
    model = model.to(DEVICE)
    report("MTKDModelV2 built", True)

    # Test forward_train
    out = model.forward_train(imgs, gt_yolo_batch=gt_batch, compute_dino=True)
    report(f"forward_train OK", True)
    report(f"  det_loss: {out['det_loss'].item():.4f}", True)
    report(f"  raw_preds present", out["raw_preds"] is not None)
    report(f"  student_spatial_feat: {out['student_spatial_feat'].shape}",
           out["student_spatial_feat"] is not None)
    report(f"  dino_features: {out['dino_features'].shape}",
           out["dino_features"] is not None)

    # Test align loss
    align_loss = model.compute_align_loss(
        out["student_spatial_feat"], out["dino_features"]
    )
    report(f"  align_loss: {align_loss.item():.4f}", True)

    # Test backward
    total = out["det_loss"] + align_loss
    total.backward()
    report("Backward pass OK", True)

    # Check gradients
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.student.parameters() if p.requires_grad
    )
    report("Student has gradients", has_grad)

    fft_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for blk in model.fft_blocks for p in blk.parameters()
    )
    report("FFT blocks have gradients", fft_has_grad)

except Exception as e:
    report("MTKDModelV2", False, str(e))
    traceback.print_exc()

# ===========================================================================
print("\n" + "=" * 60)
print("STEP 8: Trainer (1-step smoke test)")
print("=" * 60)

try:
    from mtkd_framework.train_v2 import MTKDTrainerV2, get_default_config_v2

    config = get_default_config_v2()
    config["data"]["dataset_root"] = DATASET_ROOT
    config["data"]["image_subdir"] = IMAGE_SUBDIR
    config["data"]["label_subdir"] = LABEL_SUBDIR
    config["data"]["image_size"] = 640
    config["data"]["val_ratio"] = 0.15

    config["model"]["dino_checkpoint"] = os.path.join(
        os.path.dirname(__file__), DINO_CKPT
    )
    config["model"]["fft_block_config"]["after_blocks"] = [9]

    config["training"]["epochs"] = 1
    config["training"]["batch_size"] = 2
    config["training"]["num_workers"] = 0
    config["training"]["burn_up_epochs"] = 0       # skip burn-in → test all stages
    config["training"]["align_target_start_epoch"] = 0
    config["training"]["mixed_precision"] = False

    config["output"]["save_dir"] = "outputs/smoke_test"
    config["output"]["log_freq"] = 1
    config["output"]["save_freq"] = 999  # don't save during smoke test

    config["device"] = DEVICE

    trainer = MTKDTrainerV2(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    report("MTKDTrainerV2 created", True)

    # Run 1 epoch (all stages enabled)
    train_metrics = trainer.train_epoch(epoch=0)
    report(f"train_epoch(0) OK", True)
    print(f"  Metrics: {train_metrics}")

    val_metrics = trainer.validate(epoch=0)
    report(f"validate(0) OK", True)
    print(f"  Val metrics: {val_metrics}")

except Exception as e:
    report("Trainer smoke test", False, str(e))
    traceback.print_exc()

# ===========================================================================
print("\n" + "=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed")
print("=" * 60)

if failed > 0:
    print("\n⚠️  Some tests failed — fix the issues above before training.")
else:
    print("\n🎉 All tests passed! You can start training.")
    print("""
Next steps:
  1. Generate pseudo-labels from your pretrained wheat model:
     cd /home/oscar/Myultralytics_All
     python -c "
     from ultralytics import YOLO
     model = YOLO('path/to/wheat_model.pt')
     model.predict(
         source='../Poaceae-Stomata-Detection/Stomata_Dataset/barley_category/barley_image_fresh-leaf',
         save_txt=True,
         save_conf=True,
         conf=0.3,
     )
     "

  2. Run full training:
     cd /home/oscar/Poaceae-Stomata-Detection
     python -c "
     from mtkd_framework.train_v2 import MTKDTrainerV2, get_default_config_v2
     config = get_default_config_v2()
     config['model']['dino_checkpoint'] = 'dinov3-main/weight folder/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
     config['pseudo_labels']['label_dir'] = 'path/to/pseudo_label_txts/'
     trainer = MTKDTrainerV2(config)
     trainer.train()
     "
""")
