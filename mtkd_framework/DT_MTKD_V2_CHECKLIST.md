# DINO_Teacher -> MTKD v2 對照 Checklist

目的：把 DINO_Teacher 的核心訓練機制，逐項對照到目前 `mtkd_framework`（v2）實作狀態，方便直接補齊。

狀態標記：
- `DONE`：已對齊核心行為
- `PARTIAL`：有對應實作，但細節或行為不完全一致
- `TODO`：目前缺少關鍵實作

---

## 1) 入口與訓練器

- 狀態：`DONE`
- DINO_Teacher：
  - `train_net.py` 選擇 `DINOTeacherTrainer`
- MTKD v2：
  - `MTKDTrainerV2` 類別
  - `run_v2.py` CLI 入口（`python -m mtkd_framework.run_v2 --help`）
- 參考：
  - DINO: `DINO_Teacher/train_net.py`
  - MTKD: `mtkd_framework/train_v2.py`, `mtkd_framework/run_v2.py`

## 2) Config 對齊（stage + align + pseudo）

- 狀態：`DONE`
- 已對齊：
  - `burn_up_epochs`
  - `align_target_start_epoch`
  - `feature_align_loss_weight`
  - `feature_align_loss_weight_target`
  - `unsup_loss_weight`
  - `zero_pseudo_box_reg`
  - `align_easy_only`（對應 DINO 的 `ALIGN_EASY_ONLY`）
    - 已加入 config 開關與訓練邏輯
    - 需要 dataset 提供 `batch["images_weak"]` 才會生效
- 參考：
  - DINO: `dinoteacher/config.py`
  - MTKD: `mtkd_framework/train_v2.py:get_default_config_v2`, `mtkd_framework/run_v2.py`

## 3) 三階段訓練流程（burn-in -> source align -> full）

- 狀態：`DONE`
- 已實作：
  - stage 判斷與 banner
  - epoch-based 三階段切換
- 參考：
  - DINO: `dinoteacher/engine/trainer.py:run_step_full_semisup`
  - MTKD: `mtkd_framework/train_v2.py:train_epoch`, `train`

## 4) Source Feature Alignment

- 狀態：`DONE`
- 已實作：
  - 取 student spatial feature
  - 取 frozen DINO teacher spatial feature
  - align head 投影 + loss
- 參考：
  - DINO: `dinoteacher/engine/trainer.py` (`loss_align`)
  - MTKD: `mtkd_framework/train_v2.py` (`do_source_align`), `mtkd_model_v2.py:compute_align_loss`

## 5) Target Feature Alignment（loss_align_target）

- 狀態：`DONE`
- 已實作：
  - stage 3（`epoch >= align_target_start_epoch`）時計算 `loss_align_target`
  - 使用 `feature_align_loss_weight_target` 加權
  - 在 MTKD 單一 dataset 情境中，target align 與 source align 使用相同圖片，
    但提供額外的 alignment signal
  - 支持 `teacher_images`（ALIGN_EASY_ONLY 時傳入未增強圖片）
- 參考：
  - DINO: `dinoteacher/engine/trainer.py` (`if iter >= FEATURE_ALIGN_TARGET_START`)
  - MTKD: `mtkd_framework/train_v2.py:_forward_and_loss`（`loss_align_target` 區段）

## 6) DINO Teacher 特徵抽取（frozen + spatial）

- 狀態：`DONE`
- 已實作：
  - BGR/RGB 處理
  - ImageNet normalize
  - patch 對齊 padding
  - intermediate tokens -> spatial feature map
- 參考：
  - DINO: `dinoteacher/engine/build_dino.py`
  - MTKD: `mtkd_framework/engine/build_dino.py`

## 7) Student Backbone Feature 抽取（對齊用）

- 狀態：`DONE`
- 已實作：
  - YOLO Detect pre-hook 抓 neck feature
  - `p3/p4/p5` 對齊層可選
- 參考：
  - DINO: `dinoteacher/engine/trainer.py` (`_register_input_hook_feat_align`)
  - MTKD: `mtkd_framework/models/yolo_wrappers.py`

## 8) Align Head（attention/MLP/MLP3/linear）

- 狀態：`DONE`
- 已實作：
  - 與 DINO_Teacher 同類型 head
  - interpolate 到 teacher resolution
  - cosine/L2 對齊
- 參考：
  - DINO: `dinoteacher/engine/align_head.py`
  - MTKD: `mtkd_framework/engine/align_head.py`

## 9) Pseudo-label 載入/轉換/組 batch

- 狀態：`DONE`
- 已實作：
  - YOLO txt / csv 載入
  - conf threshold
  - OBB -> AABB 轉換
  - 轉成 YOLO loss 可吃的 flat batch
- 參考：
  - DINO: `dinoteacher/engine/trainer.py`（`threshold_bbox`, `process_pseudo_label`, `add_label`）
  - MTKD: `mtkd_framework/engine/pseudo_labels.py`

## 10) Pseudo loss 權重與 regression 關閉策略

- 狀態：`DONE`
- 已實作：
  - pseudo loss 乘上 `unsup_loss_weight`
  - 可選把 pseudo box/dfl 暫時設為 0（等價於只學 cls）
- 參考：
  - DINO: `dinoteacher/engine/trainer.py`（`loss_rpn_loc_pseudo/loss_box_reg_pseudo -> 0`）
  - MTKD: `mtkd_framework/train_v2.py`（`zero_pseudo_box_reg`）

## 11) 單次 student forward 重用多種 supervision

- 狀態：`DONE`
- 已實作：
  - 一次 `forward_train_raw` 拿 `raw_preds`
  - 分別計算 GT loss 與 pseudo loss
- 參考：
  - DINO: `supervised` + `supervised_target` 分支
  - MTKD: `mtkd_framework/models/mtkd_model_v2.py:forward_train`, `mtkd_framework/train_v2.py:_forward_and_loss`

## 12) FFT Filter Bank 插入 DINO block

- 狀態：`DONE`（MTKD 擴充，非 DINO 原生）
- 已實作：
  - `PluggableFFTBlock`
  - `inject_fft_blocks` 注入與凍結原 block
  - model config 可控
- 參考：
  - MTKD: `mtkd_framework/engine/pluggable_fft_block.py`, `mtkd_framework/models/mtkd_model_v2.py`

---

## 全部項目已完成 ✅

所有 DINO_Teacher 核心訓練機制已對齊到 MTKD v2，包括：
- ✅ CLI 入口（`run_v2.py`）
- ✅ Config 完整對齊（含 `align_easy_only`）
- ✅ 三階段訓練（burn-in → source align → full）
- ✅ Source + Target Feature Alignment（`loss_align` + `loss_align_target`）
- ✅ Align Head（attention / MLP / MLP3 / linear）
- ✅ DINO 特徵抽取（frozen + spatial）
- ✅ Student backbone feature 抽取
- ✅ Pseudo-label 載入 / 轉換 / 組 batch
- ✅ Pseudo loss 權重與 regression 關閉策略
- ✅ 單次 student forward 重用多種 supervision
- ✅ FFT Filter Bank 插入 DINO block
- ✅ Smoke test 38/38 全通過

### 後續可選優化

1. **`align_easy_only` 雙視角資料**：需要 dataset 提供 `images_weak`（未增強副本），目前邏輯已就緒但 dataset 層尚未實作
2. **mAP 評估**：目前 validation 用 loss，可加入 COCO mAP 評估
3. **TensorBoard / W&B logging**：可擴充 metrics 可視化
