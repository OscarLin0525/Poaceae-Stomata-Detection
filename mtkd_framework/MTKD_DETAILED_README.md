# Multi-Teacher Knowledge Distillation (MTKD) Framework

用於禾本科氣孔檢測的多教師知識蒸餾框架

## 目錄

0. [⭐⭐ 重構計畫：基於 DINO Teacher 的對照修改](#重構計畫基於-dino-teacher-的對照修改)
1. [框架概述](#框架概述)
2. [架構設計](#架構設計)
3. [安裝與依賴](#安裝與依賴)
4. [快速開始](#快速開始)
5. [模組詳解](#模組詳解)
6. [函數參考](#函數參考)
7. [配置說明](#配置說明)
8. [訓練指南](#訓練指南)
9. [自定義擴展](#自定義擴展)
10. [常見問題](#常見問題)
11. [⭐ YOLO Student 整合指南](#yolo-student-整合指南)
12. [⭐ 實作細節與狀態](#實作細節與狀態)

---

## 重構計畫：基於 DINO Teacher 的對照修改

> **參考 ground truth**: `DINO_Teacher/` — *"Large Self-Supervised Models Bridge the Gap in Domain Adaptive Object Detection"* (CVPR 2025, Lavoie et al.)
>
> **修改目標**: `Poaceae-Stomata-Detection/mtkd_framework/`
>
> 本節以 DINO Teacher repo 為 ground truth，逐項對照 MTKD 的現有實作，列出所有需要修改的項目。

### 0.1 可以從 DINO Teacher 直接抄的三件事

DINO Teacher repo 可以直接參考的核心機制有 **三項**，不只是 feature alignment 和 prediction alignment：

| # | 機制 | 說明 | 是否必要 |
|---|------|------|---------|
| **① Feature Alignment** | Student backbone feature 對齊 frozen DINO teacher patch feature（spatial per-pixel） | ✅ 必要 |
| **② Pseudo-Label 對齊** | Teacher 產生 weak prediction → 門檻篩選 → 當作 target supervision 餵給 student | ✅ 必要 |
| **③ Domain Alignment** | GRL + discriminator（source vs target domain invariance） | ⚡ 可選 |

> **注意**：DINO Teacher 裡沒有獨立的 logits/KL consistency loss。它的 "prediction alignment" 實際上就是 **pseudo-label → standard detection loss**，不是 element-wise KD。

---

### 0.2 Feature Alignment：要抄的核心程式碼

以下是 DINO Teacher 中 feature alignment 的完整鏈路，每一步都需要在 MTKD 中有對應實作：

#### 0.2.1 Config 參數

**來源**: `DINO_Teacher/dinoteacher/config.py` (L58-L70)

```python
_C.SEMISUPNET.USE_FEATURE_ALIGN = False
_C.SEMISUPNET.FEATURE_ALIGN_LAYER = 'res4'          # 要抓哪一層 backbone feature
_C.SEMISUPNET.ALIGN_MODEL = "dinov2_vitb14"          # frozen DINO model
_C.SEMISUPNET.DINO_PATCH_SIZE = 14
_C.SEMISUPNET.ALIGN_HEAD_TYPE = "attention"           # attention / MLP / MLP3 / linear
_C.SEMISUPNET.ALIGN_HEAD_PROJ_DIM = 1024
_C.SEMISUPNET.ALIGN_PROJ_GELU = False
_C.SEMISUPNET.ALIGN_HEAD_NORMALIZE = True             # L2 normalize features
_C.SEMISUPNET.ALIGN_EASY_ONLY = True                  # teacher 只看 weak augmentation
_C.SEMISUPNET.FEATURE_ALIGN_TARGET_START = 5000       # 多少 iter 後開始 target alignment
_C.SEMISUPNET.FEATURE_ALIGN_LOSS_WEIGHT = 1.0
_C.SEMISUPNET.FEATURE_ALIGN_LOSS_WEIGHT_TARGET = 1.0
```

**MTKD 對應**: 目前散落在 `train.py` 的 `get_default_config()` 裡，缺少 `ALIGN_HEAD_TYPE`、`ALIGN_EASY_ONLY`、`FEATURE_ALIGN_TARGET_START` 等關鍵參數。

#### 0.2.2 DINO Teacher Feature Extractor（frozen）

**來源**: `DINO_Teacher/dinoteacher/engine/build_dino.py` — class `DinoVitFeatureExtractor` (L29) + `forward()` (L103-L125)

核心流程：
1. **預處理** (L36-L38 init, L105-L110 forward) — BGR→RGB + ImageNet normalize (`mean=[123.675, 116.280, 103.530]`, `std=[58.395, 57.120, 57.375]`)
2. **Pad** (L109) — `ImageList.from_tensors(x, patch_size)` pad 到 `patch_size` 整除
3. **Feature extraction** (L118) — `encoder.get_intermediate_layers(x)[0]` → `[B, num_patches, D]`
4. **Strip CLS token** (L119-L120) —（DINOv1 才需要，`x[:,1:,:]`）
5. **L2 normalize** (L122-L123) — `F.normalize(x, p=2, dim=2)` 沿 embedding dim
6. **Reshape** (L125) — `.view(B, D, f_height, f_width)` → `[B, D, H/p, W/p]` 回到 **spatial feature map**

**MTKD 現狀 (`mtkd_model.py` L149 `DINOFeatureTeacher.forward`)**: ❌ 沒有預處理、❌ 沒有 pad、❌ 沒有 reshape 成 spatial、❌ 只回傳 CLS token 做對齊

#### 0.2.3 Alignment Head（projection + loss）

**來源**: `DINO_Teacher/dinoteacher/engine/align_head.py` (L9-L62)

`TeacherStudentAlignHead`:
- **Projection**: 全部用 `nn.Conv2d(1×1)` — 在 spatial feature map 上做 per-pixel projection
  - `attention`: MHA self-attention + Conv2d
  - `MLP`: Conv2d → ReLU → Conv2d
  - `MLP3`: Conv2d → ReLU → Conv2d → ReLU → Conv2d
  - `linear`: 單一 Conv2d
- **Spatial interpolation**: `F.interpolate(projected, (h, w), mode='bilinear')` 對齊到 DINO 空間解析度
- **Post-norm**: `F.normalize(p=2, dim=1)` 沿 channel 維度
- **Loss**: 如果 normalized → per-pixel dot product cosine loss `(1 - s·t).mean()`；否則 → L2 / 100

**MTKD 現狀 (`student_model.py` L21-L117)**: ❌ 用 `nn.Linear` 不是 `nn.Conv2d`、❌ 無 spatial interpolation、❌ global avg pool 把空間資訊丟掉了

#### 0.2.4 Trainer 掛載 + Hook

**來源**: `DINO_Teacher/dinoteacher/engine/trainer.py` (L65-L80, L661-L668)

Trainer init 中：
```python
# 取 student backbone 在 align_layer 的 channel 數
student_align_dim = model.backbone._out_feature_channels[FEATURE_ALIGN_LAYER]
# 取 DINO teacher 的 embed_dim
teacher_align_dim = [*model.align_teacher.modules()][-2].normalized_shape[0]
# 建 alignment head
model.align_student_head = TeacherStudentAlignHead(cfg, student_align_dim, teacher_align_dim, ...)
# 在 proposal_generator 上掛 forward hook，自動抓 backbone feature
self._register_input_hook_feat_align(model, 'proposal_generator')
```

Hook 機制 (L661-L662)：
```python
def _get_detector_input_hook(self, module, input, output):
    # input[1] 是 backbone feature dict，[RPN_IN_FEATURES[0]] 取出指定層
    self.student_align_feat[self.branch] = input[1][self.cfg.MODEL.RPN.IN_FEATURES[0]]
```

**MTKD 現狀**: 在 `yolo_wrappers.py` 掛了 pre-hook 抓 neck features，但之後 **global avg pool** 掉了。應改為直接傳出 spatial feature map。

#### 0.2.5 Source/Target Align Loss 計算

**來源**: `DINO_Teacher/dinoteacher/engine/trainer.py` (L454-L492)

```python
# === Source alignment（始終啟用） ===
if ALIGN_EASY_ONLY:
    teacher_feat = align_teacher(label_data_k)           # weak aug only
    teacher_feat = teacher_feat.repeat(2,1,1,1)          # 複製給 strong+weak
else:
    teacher_feat = align_teacher(all_label_data)

student_feat = align_student_head(
    hooked_backbone_feat['supervised'],                   # 從 hook 拿
    teacher_feat.shape[2:])                               # 目標空間尺寸
loss_align = align_student_head.align_loss(student_feat, teacher_feat)

# === Target alignment（iter >= FEATURE_ALIGN_TARGET_START 才啟用） ===
if iter >= FEATURE_ALIGN_TARGET_START:
    teacher_feat_target = align_teacher(unlabel_data_k)   # target domain
    student_feat_target = align_student_head(
        hooked_backbone_feat['supervised_target'],
        teacher_feat_target.shape[2:])
    loss_align_target = align_student_head.align_loss(
        student_feat_target, teacher_feat_target)
```

**MTKD 現狀**: 直接在 `MTKDModel.forward()` 裡比對 `adapted_features`（global pooled）和 `cls_token`（DINO CLS）。沒有 source/target 分離、沒有 easy-aug-only。

#### 0.2.6 `forward_backbone` shortcut

**來源**: `DINO_Teacher/dinoteacher/modeling/meta_arch/rcnn.py` (L44-L47)

```python
def forward_backbone(self, batched_inputs):
    images = self.preprocess_image(batched_inputs)
    features = self.backbone(images.tensor)
    return features
```

用途：在 target alignment 的早期階段（`has_target_backbone_feats == 0`），unlabel data 沒有經過 student detection forward，所以需要單獨跑 backbone 取 feature。

**MTKD 現狀**: 不需要此 shortcut，因為 YOLO student 的 forward 已經會回傳 neck features。但需要確保在 burn-in 期間也能拿到 target 圖片的 backbone features。

---

### 0.3 Pseudo-Label 對齊：要抄的核心程式碼

DINO Teacher 的 "prediction alignment" 本質上是用 pseudo-labels 做 supervised loss，**不是** element-wise logits KD。

#### 0.3.1 Teacher 產生 Weak Prediction

**來源**: `DINO_Teacher/dinoteacher/engine/trainer.py` (L360-L401)

兩種 pseudo-label 來源：

**來源 A — VFM Labeller（預先生成）**:
```python
# 從 pickle 檔載入 DINO ViT fine-tuned 的預測
instances = self.dino_pseudogt[image_id]['instances_dino']
# 用 transform data 對齊到 augmented image 座標
boxes = x['tf_data'].apply_box(y.pred_boxes)
```

**來源 B — EMA Mean Teacher（線上生成）**:
```python
with torch.no_grad():
    _, proposals_rpn, proposals_roih, _ = self.model_teacher(
        unlabel_data_k, branch="unsup_data_weak")  # weak augmentation!
```

#### 0.3.2 門檻篩選

**來源**: `DINO_Teacher/dinoteacher/engine/trainer.py` (L233-L288)

```python
def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
    valid_map = proposal_bbox_inst.scores > thres
    new_proposal_inst.gt_boxes = Boxes(new_bbox_loc)
    new_proposal_inst.gt_classes = pred_classes[valid_map]
    new_proposal_inst.scores = scores[valid_map]
    return new_proposal_inst
```

支援三種 proposal type：`rpn`（objectness）、`roih`（ROI head scores）、`dino`（VFM scores）

#### 0.3.3 塞回 Unlabeled Batch

**來源**: `DINO_Teacher/dinoteacher/engine/trainer.py` (L403-L412)

```python
# 把 pseudo-labels 當作 gt 塞回 unlabeled data
unlabel_data_q = self.add_label(
    unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"])
```

`add_label()` 就是把 filterd pseudo instances 塞到 `datum["instances"]` 欄位，跟 ground truth 格式完全一樣。

#### 0.3.4 用 `supervised_target` 分支算 Pseudo Loss

**來源**: `DINO_Teacher/dinoteacher/engine/trainer.py` (L421-L428)

```python
self.branch = "supervised_target"
record_all_unlabel_data, _, _, _ = self.model(
    all_unlabel_data, branch="supervised_target")
# 把 key 加上 _pseudo 後綴
for key in record_all_unlabel_data.keys():
    new_record[key + "_pseudo"] = record_all_unlabel_data[key]
```

實際上跟 supervised loss 走**完全相同的 detection loss**（RPN cls/loc + ROI cls/reg），只是資料換成 pseudo-labeled 的。

#### 0.3.5 Pseudo Loss 權重

**來源**: `DINO_Teacher/dinoteacher/engine/trainer.py` (L498-L522)

```python
for key in record_dict.keys():
    if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo":
        loss_dict[key] = record_dict[key] * 0              # pseudo bbox regression 歸零!
    elif key[-6:] == "pseudo":
        loss_dict[key] = record_dict[key] * UNSUP_LOSS_WEIGHT  # default: 4.0
    elif key == "loss_align":
        loss_dict[key] = record_dict[key] * FEATURE_ALIGN_LOSS_WEIGHT       # default: 1.0
    elif key == "loss_align_target":
        loss_dict[key] = record_dict[key] * FEATURE_ALIGN_LOSS_WEIGHT_TARGET # default: 1.0
    else:  # supervised loss
        loss_dict[key] = record_dict[key] * 1
```

**關鍵細節**：pseudo bbox regression loss 被設為 **0**（只保留 classification），supervised loss 權重 **1.0**，pseudo loss 權重 **4.0**。

**MTKD 現狀**: `PredictionAlignmentLoss` 嘗試 element-wise 比對 student/teacher predictions（`[B, N, 4]` vs `[B, N, 4]`），但因 shape mismatch 會 crash。且完全沒有 GT supervised loss（`_compute_detection_loss` 回傳 0）。

---

### 0.4 Domain Alignment（可選）

**來源**: `DINO_Teacher/dinoteacher/engine/trainer.py` (L436-L451)，繼承自 Adaptive Teacher

- 用 GRL (Gradient Reversal Layer) + image-level discriminator
- Source weak + Target weak 一起送入 `branch="domain"`
- 損失：`loss_D_img_s` + `loss_D_img_t`，權重 `DIS_LOSS_WEIGHT`（default 0.1）
- 控制開關：`self.use_adversarial_invariance = DIS_LOSS_WEIGHT > 0`

**MTKD 現狀**: 完全沒有 domain alignment 實作。若未來需 domain adaptation（如 wheat→barley），可加入。

---

### 0.5 MTKD 現存問題清單

#### Critical（不修會壞 / 結果無意義）

| # | 問題 | 位置 | DINO Teacher 對照 |
|---|------|------|-------------------|
| **C1** | Feature alignment 是 global CLS-token-level（每張圖一個 768-d 向量），丟失所有空間資訊 | `yolo_wrappers.py` L204 → `global_pool(feat).flatten(1)` | `align_head.py` 整個檔案：per-pixel Conv2d + spatial interpolation |
| **C2** | DINO teacher forward 缺少 ImageNet 預處理 — features 無意義 | `mtkd_model.py` L149 (`DINOFeatureTeacher.forward`) | `build_dino.py` L36-L38 (preprocessing init), L103-L125 (forward) |
| **C3** | Prediction alignment shape mismatch — YOLO `[B, 8400, 4]` vs Teacher `[B, ~50, 4]` 直接 crash | `prediction_alignment.py` L428-L438 (`PredictionAlignmentLoss.forward`) | 不存在此問題：pseudo-labels 走 standard detection loss |
| **C4** | Detection loss 是空殼函數，return 0.0 — student 沒有 GT supervision | `mtkd_model.py` L432-L470 (`_compute_detection_loss`) | `trainer.py` L415-L418: supervised branch 是整個 framework 的基礎 |

#### Major（影響訓練效果）

| # | 問題 | 位置 | DINO Teacher 對照 |
|---|------|------|-------------------|
| **M1** | Projection head 用 `nn.Linear` 而非 `nn.Conv2d` | `student_model.py` L21-L117 (`FeatureAdapter`) | `align_head.py` L19-L34: Conv2d(1×1) |
| **M2** | 缺少 progressive training（burn-in → alignment → full） | `train.py` 整體 | `trainer.py` L325-L342: 三階段 |
| **M3** | 無 weak/strong augmentation 區分 | `train.py` 整體 | `trainer.py` L315-L317: `data_q`(strong) / `data_k`(weak) |
| **M4** | 無 EMA teacher update | 無 | `trainer.py` L576-L594: `_update_teacher_model(keep_rate=0.9996)` |

#### Dead Code（~60% 的 loss/model code 從未執行）

| Code | 位置 | 原因 |
|------|------|------|
| `HungarianMatchingLoss` | `prediction_alignment.py` L466-657 | 從未實例化 |
| `AdaptiveMTKDLoss` | `combined_loss.py` L255-310 | 從未實例化 |
| `UncertaintyWeightedMTKDLoss` | `combined_loss.py` L313-425 | 從未實例化 |
| `StudentDetector` / `StudentBackbone` / `DetectionHead` / `SimpleFPN` | `student_model.py` | 用 YOLO，這些全不使用 |
| `PlaceholderDINO` | `mtkd_model.py` L90-107 | Conv2d ≠ DINO |
| `SoftNMS` | `teacher_ensemble.py` L200-290 | 預設 WBF |
| `MultiScaleFeatureAdapter` 使用面 | `student_model.py` L127-151 | 只在 dead code StudentDetector 裡 |
| `AttentionAlignmentLoss` 使用路徑 | `combined_loss.py` L227-231 | `use_attention_alignment=False`，無 attention maps |
| Token matching 觸發條件 | `combined_loss.py` L213 | `dim() == 3` 永遠不成立 |
| `_compute_detection_loss` | `mtkd_model.py` L436-470 | 函數內容全是 `pass`，return 0 |
| `freeze/unfreeze_student_backbone` | `mtkd_model.py` L553-564 | 從未呼叫 |

#### 小 Bug

| Bug | 位置 | 修復方式 |
|-----|------|---------|
| `config.pop("embed_dim")` 汙染原 dict | `mtkd_model.py` L253 | 改 `config.get()` 或 `copy()` |
| `_get_device()` 在無 parameter 時 crash | `combined_loss.py` L248 | 改用 `self.current_epoch.device` |
| YOLO eval-mode forward 影響 BN | `yolo_wrappers.py` L148-158 | Training 中 BN 用 running stats 不合理 |
| Ensemble logit 重建效率低且語義有誤 | `teacher_ensemble.py` L620-635 | 改 pseudo-label 後可刪 |

---

### 0.6 修改計畫（按優先序）

> **原則**：先讓 student 學得動（GT loss），再讓 teacher signal 正確（preprocessing + pseudo-label），最後提升 alignment 品質（spatial）。沒有 GT supervision 的 student backbone 根本不會學東西，後面的 KD 都是空轉。

#### Step 1 — 修復 Detection Loss（GT Supervision）🔴 最高優先

**修改**: `mtkd_framework/models/mtkd_model.py` — `_compute_detection_loss()`

| 要做的事 | 參考來源 |
|---------|---------|
| 接上 YOLO 內建 loss（CIoU + BCE + DFL） | `trainer.py` L415-L418: supervised branch 就是 standard detection loss |
| 確保 GT supervision 始終存在 | DINO Teacher 的 supervised loss 是基礎，KD 只是輔助 |

**為什麼第一步**：目前 `_compute_detection_loss` 回傳 0（`mtkd_model.py` L432-L470 全是 `pass`），student 沒有 gradient signal，backbone 不會更新。所有 alignment / KD 訓號都在對齊一個「凍結的隨機 backbone」。

#### Step 2 — 修復 DINO Teacher 預處理 ⟵ `build_dino.py` 🔴

**修改**: `mtkd_framework/models/mtkd_model.py` — `DINOFeatureTeacher`

| 要做的事 | 參考來源 |
|---------|---------|
| 加 ImageNet mean/std normalize | `build_dino.py` L17-L27 (`dino_preprocessing` class) |
| BGR→RGB 轉換（如果需要） | `build_dino.py` L105-L106 |
| Pad 到 `patch_size` 整除 | `build_dino.py` L109 |
| Reshape patch tokens 成 spatial `[B, D, h, w]` | `build_dino.py` L125 |
| L2 normalize 沿 embedding dim | `build_dino.py` L122-L123 |

**為什麼第二步**：5 分鐘修復但影響巨大。沒有正確的預處理，DINO ViT 輸出的 features 毫無語義，feature alignment loss 在對齊垃圾。即使暫時保留 global alignment，這步也要先做。

#### Step 3 — 改 Prediction Alignment 為 Pseudo-Label 方式 ⟵ `trainer.py` L379-L520 🟠

**修改**: `mtkd_framework/models/mtkd_model.py`, `mtkd_framework/losses/prediction_alignment.py`

| 要做的事 | 參考來源 |
|---------|---------|
| Ensemble teacher predictions → threshold 篩選 → pseudo-labels | `trainer.py` L233-L270 (`threshold_bbox`) |
| 把 pseudo-labels 轉成 YOLO target 格式（normalized xywh + class） | `trainer.py` L405-L412 (`add_label`) |
| 用 standard YOLO detection loss 計算 pseudo loss | `trainer.py` L421-L428 (`supervised_target` branch) |
| Pseudo bbox regression loss × 0（只留 classification） | `trainer.py` L499-L501 |
| Pseudo loss 權重 = `UNSUP_LOSS_WEIGHT`（default 4.0） | `trainer.py` L503-L505 |
| 移除 `PredictionAlignmentLoss` 的 element-wise 對齊 | — |

**為什麼第三步**：這是 DINO Teacher 和 MTKD「差最多」的地方。element-wise KD（8400 vs ~50）根本跑不動；改成 pseudo-label supervision 才是 DINO Teacher 成效的核心路線。

#### Step 4 — 加入 Progressive Training ⟵ `trainer.py` L325-L342 🟠

**修改**: `mtkd_framework/train.py`

| 階段 | 條件 | 內容 | 參考 |
|------|------|------|------|
| **Burn-in** | `iter < BURN_UP_STEP` (default 12000) | GT supervised loss + DINO feature alignment only | `trainer.py` L332-L340 |
| **Full training** | `iter >= BURN_UP_STEP` | + pseudo-label loss + (optional) target alignment | `trainer.py` L347-L432 |

**為什麼第四步**：Pseudo-label 在 student 還沒訓好時品質很差。DINO Teacher 讓 student 先 burn-in 再引入 pseudo-labels，避免 noisy labels 毒害早期訓練。

#### Step 5 — Weak/Strong 雙流 Dataset ⟵ `dataset_mapper.py` 🟡

**修改**: `mtkd_framework/data/stomata_dataset.py` + 新增 `mtkd_framework/data/dual_augment.py`

DINO Teacher 用 `DatasetMapperTwoCropSeparateKeepTf`（`dinoteacher/data/dataset_mapper.py` L168-L350）：每張圖同時產出 strong + weak 兩個版本，且保留 transform trace 供 pseudo box 座標映射。

| 要做的事 | 參考來源 | MTKD 現狀 |
|---------|---------|----------|
| 每張圖產出 `(data_q, data_k)` — strong/weak 兩份 | `dataset_mapper.py` L316-L340：weak 是原始 augmentation，strong 再加 color jitter/grayscale/blur | `stomata_dataset.py` L124：只有單張 + 隨機水平翻轉 |
| Strong augmentation 包含 color jitter + random grayscale + Gaussian blur | `adapteacher/data/detection_utils.py` `build_strong_augmentation()` | 無 |
| 保留 `tf_data = transforms`（geometric transform trace） | `dataset_mapper.py` L315：`dataset_dict['tf_data'] = transforms` | 無 |
| DataLoader 每個 batch 解包為 `label_q, label_k, unlabel_q, unlabel_k` | `trainer.py` L317：`label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data` | 單一 batch |
| Teacher 只看 weak（`_k`）做 inference，pseudo box 放在 strong（`_q`）上做 loss | `trainer.py` L380：`model_teacher(unlabel_data_k, branch="unsup_data_weak")` | 不區分 |
| `tf_data.apply_box()` 把預生成 pseudo box 映射到 augmented image 座標 | `trainer.py` L363：`x['tf_data'].apply_box(y.pred_boxes)` | 不需要（MTKD 是 online inference，不是 offline labels） |

**MTKD 簡化方案**：因為 MTKD 的 ensemble teacher 是 frozen 且 online inference（不是 offline labeller），所以：
- `tf_data` transform trace **不需要**（teacher 直接看 weak image 做 forward，box 座標已在 weak image 空間）
- 只需要實作 **weak/strong 雙流 augmentation** + **labeled/unlabeled 分流**
- Weak: 基本 resize + flip
- Strong: weak + ColorJitter + RandomGrayscale + GaussianBlur（參考 `build_strong_augmentation()`）
- DataLoader 改為每 batch 回傳 `(labeled_strong, labeled_weak, unlabeled_strong, unlabeled_weak)`

#### Step 6 — 改 Feature Alignment 為 Spatial Map 對齊 🟡

分為三個子步驟：

##### Step 6a — 新增 SpatialAlignHead ⟵ `align_head.py`

**新增**: `mtkd_framework/models/align_head.py`

| 要做的事 | 參考來源 |
|---------|---------|
| Conv2d(1×1) projection（MLP / MLP3 / attention / linear） | `align_head.py` L19-L34 |
| `F.interpolate(bilinear)` 到 teacher spatial size | `align_head.py` L43 |
| Post-projection L2 normalize | `align_head.py` L44-L45 |
| Per-pixel cosine loss: `(1 - dot_product).mean()` | `align_head.py` L48-L62 |

##### Step 6b — 改 Student 輸出為 Spatial Feature Map

**修改**: `mtkd_framework/models/yolo_wrappers.py`

| 要做的事 | 參考來源 |
|---------|---------|
| 移除 `global_pool` + `flatten` + `FeatureAdapter` | — |
| 直接輸出 P4 spatial feature map `[B, C, H, W]` | `trainer.py` L661: hook 直接抓 spatial |
| 在 `MTKDModel` 中建立 `SpatialAlignHead` 並用它做 projection + loss | `trainer.py` L65-L75 |

##### Step 6c — Source/Target 分離 + easy-aug-only

**修改**: `mtkd_framework/models/mtkd_model.py`, `mtkd_framework/losses/combined_loss.py`

| 要做的事 | 參考來源 |
|---------|---------|
| Source alignment: teacher 只看 weak aug（`ALIGN_EASY_ONLY`）| `trainer.py` L456-L459 |
| Target alignment: `iter >= FEATURE_ALIGN_TARGET_START` 才啟用 | `trainer.py` L467-L492 |
| 兩個 alignment loss 分開加權 | `trainer.py` L515-L518 |

#### Step 7 — 清理 Dead Code

刪除所有從未執行的代碼（見 0.5 Dead Code 清單），精簡 framework。

#### 修改順序總覽

| 優先 | Step | 修改內容 | 影響 | 工作量 |
|:---:|:---:|----------|------|:------:|
| 🔴 | 1 | GT detection loss | 沒有就全部白做 | 小 |
| 🔴 | 2 | DINO preprocessing | feature alignment 變有意義 | 小 |
| 🟠 | 3 | Pseudo-label supervision | prediction alignment 從 broken 到 functional | 中 |
| 🟠 | 4 | Progressive training | 避免 noisy pseudo-labels 毒害早期 | 小 |
| 🟡 | 5 | Weak/strong 雙流 dataset | pseudo-label 品質 + teacher inference 正確性 | 中 |
| 🟡 | 6 | Spatial feature alignment | 從 global 到 per-pixel，提升 alignment 品質 | 中 |
| ⚪ | 7 | Dead code cleanup | 可讀性 + 維護性 | 小 |

---

### 0.7 修改後的架構流程圖

```
Input Image
    │
    ├──► Frozen DINO ViT Teacher ──► Spatial Patch Features [B, D, h, w]
    │       (ImageNet normalize,                    │
    │        pad to patch_size,                     │ L2 normalize (per-patch)
    │        reshape to spatial)                    ▼
    │                                    ┌─── Per-Pixel Cosine Loss ◄───┐
    │                                    │    L = (1 - s·t).mean()      │
    │                                    │                              │
    ├──► YOLO Student ──► P4 Feature Map [B, C, H, W]                  │
    │       │              │                                            │
    │       │              └──► SpatialAlignHead ──► Proj [B, D, h, w] ─┘
    │       │                   (Conv2d 1×1 + bilinear interp + L2 norm)
    │       │
    │       └──► Detection Output ──────┐
    │               │                   │
    │               ├──► GT Supervised Loss (YOLO built-in: CIoU + BCE + DFL)
    │               │
    │               └──► Pseudo-Label Supervised Loss
    │                       ↑
    │                       │ (threshold-filtered pseudo-labels as targets)
    │                       │
    └──► Frozen YOLO Ensemble Teacher ──► WBF Fused Predictions
             ──► Score Threshold ──► Pseudo-Labels (gt_boxes + gt_classes)
```

**Loss 公式**:

$$L_{\text{total}} = 1.0 \cdot L_{\text{supervised}} + w_{\text{align}} \cdot L_{\text{feature\_align}} + w_{\text{pseudo}} \cdot L_{\text{pseudo}} + w_{\text{align\_target}} \cdot L_{\text{target\_align}}$$

其中 pseudo bbox regression loss 設為 0（只保留 classification pseudo loss）。

---

### 0.8 對照速查表

| 功能 | DINO Teacher 參考位置 | MTKD 現狀位置 | 修改方向 | Step |
|------|----------------------|--------------|---------|:----:|
| **GT detection loss** | `trainer.py` L415-L418 (supervised branch) | `mtkd_model.py` L432-L470 (空殼 `pass`) | 接上 YOLO 內建 loss | 1 |
| DINO 預處理 | `build_dino.py` L36-L38, L103-L125 | `mtkd_model.py` L149 | 加 ImageNet norm + spatial reshape | 2 |
| Pseudo-label supervision | `trainer.py` L405-L428 (add_label → supervised_target) | `prediction_alignment.py` L425 (element-wise KD, broken) | 改為 threshold → pseudo target → detection loss | 3 |
| Pseudo loss 權重 | `trainer.py` L495-L520 | `combined_loss.py` `prediction_weight` | 對齊分項權重（bbox reg ×0, cls ×4.0） | 3 |
| 門檻篩選 pseudo-labels | `trainer.py` L233-L288 | `teacher_ensemble.py` score filter | 加 explicit threshold | 3 |
| Burn-in 階段 | `trainer.py` L332-L340 | 無 | 加 progressive schedule | 4 |
| Weak/strong 雙流 dataset | `dataset_mapper.py` L168-L350 | `stomata_dataset.py` L124 (單流 + flip only) | 新增 dual augmentation + labeled/unlabeled split | 5 |
| Strong augmentation | `adapteacher/data/detection_utils.py` `build_strong_augmentation()` | 無 | 加 ColorJitter + RandomGrayscale + GaussianBlur | 5 |
| Config 參數 | `config.py` L58-L79 | `train.py` `get_default_config()` | 補齊缺失參數 | 2 |
| Align head (Conv2d spatial) | `align_head.py` L9-L62 | `student_model.py` L21-L117 (Linear) | 新增 `align_head.py` | 6a |
| Student spatial output | `trainer.py` L661 (hook 抓 spatial) | `yolo_wrappers.py` L204 (global pool) | 改為輸出 spatial features | 6b |
| Source feature align loss | `trainer.py` L454-L465 | `combined_loss.py` L207-L211 | 改為 spatial per-pixel | 6c |
| Target feature align loss | `trainer.py` L467-L492 | 無 | 新增 | 6c |
| EMA teacher 產生 weak prediction | `trainer.py` L377-L401 | 無 | 用 frozen ensemble teacher 替代 | — |
| `forward_backbone` shortcut | `rcnn.py` L44-L47 | 無（YOLO forward 已回傳 features） | 不需要 | — |
| Domain discriminator (可選) | `trainer.py` L436-L451 | 無 | 需要時再加 | — |

---

### 0.9 驗證方式

1. **Unit test**: 建 `SpatialAlignHead`，確認 output shape `[B, D, h, w]` 和 loss 值 ∈ `[0, 2]`
2. **Shape test**: `DINOFeatureTeacher` → `[B, D, h, w]`；`YOLOStudentDetector` → `[B, C, H, W]`
3. **Integration test**: 一個 batch forward+backward，確認每個 loss 都 > 0 且 gradient 有回傳
4. **Sanity check**: 訓練 10 epochs 觀察 loss 曲線（feature align loss 應下降，detection loss 正常收斂）
5. **Comparison**: 修改前後比較 mAP@50，spatial alignment 應優於 global alignment

---

## 框架概述

### 什麼是 MTKD？

Multi-Teacher Knowledge Distillation (MTKD) 是一種結合多個教師模型來訓練學生模型的知識蒸餾方法。本框架專為物體檢測任務設計，結合：

1. **DINO Feature Teacher**: 提供強大的視覺特徵表示，用於特徵對齊
2. **Detection Teacher(s)**: 預訓練檢測模型（單一或集成），用於預測對齊
3. **Student Detector**: 要訓練的學生模型

### 支援的架構

| 架構 | Feature Teacher | Detection Teacher | Student | 狀態 |
|-----|-----------------|-------------------|---------|------|
| **DETR-like** | DINO ViT | Ensemble (WBF) | DETR | ✅ 已實作 |
| **YOLO** | DINO ViT | YOLOv8 (單一) | YOLOv11 | 🔄 規劃中 |

**推薦：YOLO 架構**（見 [YOLO Student 整合指南](#yolo-student-整合指南)）
- YOLOv8 作為 Teacher：成熟穩定，提供高品質預測
- YOLOv11 作為 Student：C3k2 + C2PSA 架構，學習能力強，收斂快

### 核心優勢

- **多源知識**: 從特徵和預測兩個維度進行知識蒸餾
- **靈活架構**: 支持 DETR-like 或 YOLO 架構
- **易於擴展**: 模組化設計，方便添加新組件
- **Hungarian Matching**: 自動處理不同數量的預測配對

---

## 架構設計

### 整體架構圖

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Input Image                                │
│                              │                                      │
│          ┌───────────────────┼───────────────────┐                 │
│          │                   │                   │                 │
│          ▼                   ▼                   ▼                 │
│  ┌───────────────┐  ┌───────────────┐  ┌────────────────────┐     │
│  │ DINO Teacher  │  │   Student     │  │ Ensemble Teachers  │     │
│  │   (Frozen)    │  │  (Trainable)  │  │     (Frozen)       │     │
│  │               │  │               │  │                    │     │
│  │  ViT-Base/    │  │  Backbone     │  │  Teacher1          │     │
│  │  Large        │  │     ↓         │  │  Teacher2          │     │
│  │               │  │   Neck        │  │     ↓              │     │
│  │               │  │     ↓         │  │  Weighted Box      │     │
│  │               │  │   Head        │  │  Fusion            │     │
│  └───────┬───────┘  └───────┬───────┘  └──────────┬─────────┘     │
│          │                  │                     │                │
│          │ Features         │ Features &          │ Predictions    │
│          │                  │ Predictions         │                │
│          │                  │                     │                │
│          └────────────┬─────┴─────────────────────┘                │
│                       │                                             │
│                       ▼                                             │
│          ┌─────────────────────────────────────────┐               │
│          │            MTKD Loss                    │               │
│          │                                         │               │
│          │  ┌─────────────┐  ┌─────────────────┐  │               │
│          │  │  Feature    │  │   Prediction    │  │               │
│          │  │  Alignment  │  │   Alignment     │  │               │
│          │  │  Loss       │  │   Loss          │  │               │
│          │  └─────────────┘  └─────────────────┘  │               │
│          │                                         │               │
│          │  ┌─────────────────────────────────┐   │               │
│          │  │  Detection Loss (Optional)      │   │               │
│          │  └─────────────────────────────────┘   │               │
│          └─────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
```

### 資料夾結構

```
mtkd_framework/
├── __init__.py                 # 框架入口
├── README.md                   # 本文檔
├── train.py                    # 訓練腳本
│
├── models/                     # 模型模組
│   ├── __init__.py
│   ├── mtkd_model.py          # 主要 MTKD 模型
│   ├── student_model.py       # Student 模型
│   └── teacher_ensemble.py    # Teacher Ensemble
│
├── losses/                     # 損失函數模組
│   ├── __init__.py
│   ├── feature_alignment.py   # 特徵對齊損失
│   ├── prediction_alignment.py # 預測對齊損失
│   └── combined_loss.py       # 組合損失
│
└── utils/                      # 工具函數
    ├── __init__.py
    └── helpers.py             # 輔助函數
```

---

## 安裝與依賴

### 依賴項

```bash
pip install torch>=2.0.0
pip install torchvision>=0.15.0
pip install numpy
pip install scipy  # 用於 Hungarian matching
```

### 可選依賴

```bash
pip install fvcore  # 用於計算 FLOPs
pip install tensorboard  # 用於訓練可視化
```

---

## 快速開始

### 基本使用

```python
from mtkd_framework import MTKDModel, MTKDLoss

# 1. 創建模型
model = MTKDModel(
    student_config={
        "backbone_config": {"backbone_type": "resnet50"},
        "head_config": {"num_classes": 1, "num_queries": 100},
    },
    dino_teacher_config={"model_name": "vit_base"},
    num_classes=1,
)

# 2. 載入預訓練的 teachers
model.dino_teacher.load_pretrained("path/to/dino_checkpoint.pth")
model.ensemble_teachers.add_teacher(teacher1_model, weight=0.6)
model.ensemble_teachers.add_teacher(teacher2_model, weight=0.4)

# 3. 訓練
optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=1e-4)

for epoch in range(100):
    for images, targets in train_loader:
        loss, loss_dict = model.training_step(images, targets, epoch=epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 4. 推理
model.eval()
detections = model.inference(test_images, score_threshold=0.5)
```

### 使用訓練腳本

```bash
# 測試運行（使用 dummy 數據）
python -m mtkd_framework.train --test_run --epochs 2

# 正式訓練
python -m mtkd_framework.train \
    --config config.json \
    --dino_checkpoint path/to/dino.pth \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-4
```

---

## 模組詳解

### 1. Feature Alignment Loss (`losses/feature_alignment.py`)

用於對齊 DINO teacher 和 student 的特徵表示。

#### FeatureAlignmentLoss

```python
class FeatureAlignmentLoss(nn.Module):
    """
    特徵對齊損失

    支持多種損失類型：
    - "l2": MSE 損失
    - "cosine": Cosine Similarity 損失
    - "kl": KL Divergence 損失
    - "smooth_l1": Smooth L1 損失
    """
```

**初始化參數：**

| 參數 | 類型 | 默認值 | 說明 |
|------|------|--------|------|
| `loss_type` | str | "l2" | 損失類型 |
| `temperature` | float | 1.0 | KL divergence 溫度 |
| `normalize` | bool | True | 是否 L2 正規化 |
| `reduction` | str | "mean" | Reduction 方式 |

**使用示例：**

```python
from mtkd_framework.losses import FeatureAlignmentLoss

# 創建損失函數
loss_fn = FeatureAlignmentLoss(loss_type="cosine", normalize=True)

# 計算損失
student_feat = torch.randn(4, 768)  # [B, D]
teacher_feat = torch.randn(4, 768)
loss = loss_fn(student_feat, teacher_feat)
```

**損失公式：**

- **L2 Loss**: $L = \frac{1}{N} \sum_{i=1}^{N} ||s_i - t_i||_2^2$

- **Cosine Loss**: $L = 1 - \frac{s \cdot t}{||s|| \cdot ||t||}$

- **KL Loss**: $L = T^2 \cdot \sum_i p_t(i) \log \frac{p_t(i)}{p_s(i)}$

  其中 $p_t = \text{softmax}(t/T)$, $p_s = \text{softmax}(s/T)$

#### MultiScaleFeatureAlignmentLoss

```python
class MultiScaleFeatureAlignmentLoss(nn.Module):
    """
    多尺度特徵對齊損失

    用於對齊 FPN 等多尺度特徵。
    """
```

**使用示例：**

```python
from mtkd_framework.losses import MultiScaleFeatureAlignmentLoss

loss_fn = MultiScaleFeatureAlignmentLoss(
    num_scales=4,
    student_channels=[256, 256, 256, 256],
    teacher_channels=[768, 768, 768, 768],
    scale_weights=[1.0, 1.0, 1.0, 1.0],
)

student_feats = [torch.randn(4, 256, 56, 56) for _ in range(4)]
teacher_feats = [torch.randn(4, 768, 56, 56) for _ in range(4)]
loss = loss_fn(student_feats, teacher_feats)
```

---

### 2. Prediction Alignment Loss (`losses/prediction_alignment.py`)

用於對齊 ensemble teachers 和 student 的預測。

#### BoxAlignmentLoss

```python
class BoxAlignmentLoss(nn.Module):
    """
    Bounding Box 對齊損失

    支持：
    - "l1": L1 損失
    - "smooth_l1": Smooth L1 損失
    - "giou": Generalized IoU 損失
    - "ciou": Complete IoU 損失
    """
```

**GIoU 計算公式：**

$$\text{GIoU} = \text{IoU} - \frac{|C - (A \cup B)|}{|C|}$$

其中 $C$ 是包含 $A$ 和 $B$ 的最小閉包區域。

**CIoU 計算公式：**

$$\text{CIoU} = \text{IoU} - \frac{\rho^2(b, b^{gt})}{c^2} - \alpha v$$

其中：
- $\rho$: 中心點距離
- $c$: 閉包對角線長度
- $v$: 寬高比一致性
- $\alpha$: 權重參數

**使用示例：**

```python
from mtkd_framework.losses import BoxAlignmentLoss

loss_fn = BoxAlignmentLoss(loss_type="giou", box_format="cxcywh")

student_boxes = torch.randn(100, 4)  # [N, 4] (cx, cy, w, h)
teacher_boxes = torch.randn(100, 4)
loss = loss_fn(student_boxes, teacher_boxes)
```

#### ClassAlignmentLoss

```python
class ClassAlignmentLoss(nn.Module):
    """
    類別預測對齊損失

    使用 KL Divergence 對齊分類預測的概率分佈。
    """
```

**使用示例：**

```python
from mtkd_framework.losses import ClassAlignmentLoss

loss_fn = ClassAlignmentLoss(
    loss_type="kl",
    temperature=4.0,  # 軟化概率分佈
)

student_logits = torch.randn(100, 10)  # [N, num_classes]
teacher_logits = torch.randn(100, 10)
loss = loss_fn(student_logits, teacher_logits)
```

#### PredictionAlignmentLoss

```python
class PredictionAlignmentLoss(nn.Module):
    """
    完整的預測對齊損失

    組合 Box 對齊和 Class 對齊損失。
    """
```

**使用示例：**

```python
from mtkd_framework.losses import PredictionAlignmentLoss

loss_fn = PredictionAlignmentLoss(
    box_loss_type="giou",
    class_loss_type="kl",
    box_weight=2.0,
    class_weight=1.0,
    temperature=4.0,
)

student_pred = {
    "boxes": torch.randn(4, 100, 4),   # [B, N, 4]
    "logits": torch.randn(4, 100, 10),  # [B, N, num_classes]
}
teacher_pred = {
    "boxes": torch.randn(4, 100, 4),
    "logits": torch.randn(4, 100, 10),
}

loss, loss_dict = loss_fn(student_pred, teacher_pred)
```

---

### 3. Teacher Ensemble (`models/teacher_ensemble.py`)

管理多個教師模型並融合它們的預測。

#### WeightedBoxFusion

```python
class WeightedBoxFusion(nn.Module):
    """
    Weighted Box Fusion (WBF)

    將多個模型的 bounding box 預測進行加權融合。
    與 NMS 不同，WBF 會融合重疊的 boxes 而非抑制它們。
    """
```

**WBF 算法流程：**

1. 按信心度排序所有預測
2. 對每個類別分別處理
3. 將 IoU 超過閾值的 boxes 聚類
4. 對每個 cluster 計算加權平均 box
5. 根據參與的模型數量調整信心度

**使用示例：**

```python
from mtkd_framework.models import WeightedBoxFusion

wbf = WeightedBoxFusion(
    iou_threshold=0.55,
    weights=[0.6, 0.4],  # 兩個模型的權重
    conf_type="avg",
)

# 來自兩個模型的預測
boxes_list = [
    torch.tensor([[0.1, 0.1, 0.3, 0.3]]),  # model 1
    torch.tensor([[0.12, 0.11, 0.31, 0.32]]),  # model 2
]
scores_list = [torch.tensor([0.9]), torch.tensor([0.85])]
labels_list = [torch.tensor([0]), torch.tensor([0])]

# 融合
fused_boxes, fused_scores, fused_labels = wbf(boxes_list, scores_list, labels_list)
```

#### TeacherEnsemble

```python
class TeacherEnsemble(nn.Module):
    """
    Teacher Ensemble 模組

    組合多個預訓練的 detection teacher 模型。
    """
```

**使用示例：**

```python
from mtkd_framework.models import TeacherEnsemble

# 創建 ensemble
ensemble = TeacherEnsemble(
    teacher_weights=[0.6, 0.4],
    fusion_method="wbf",
    num_classes=1,
)

# 添加教師模型
ensemble.add_teacher(teacher1_model, weight=0.6)
ensemble.add_teacher(teacher2_model, weight=0.4)

# 獲取融合預測
predictions = ensemble(images)
# predictions = {"boxes": [...], "scores": [...], "labels": [...], "valid_mask": [...]}
```

---

### 4. Student Model (`models/student_model.py`)

學生檢測模型，包含特徵適配器。

#### FeatureAdapter

```python
class FeatureAdapter(nn.Module):
    """
    特徵適配器

    將 student 的特徵維度轉換到與 teacher 相同。

    支持類型：
    - "linear": 線性投影
    - "mlp": 兩層 MLP
    - "attention": 使用 attention 的適配
    """
```

**使用示例：**

```python
from mtkd_framework.models.student_model import FeatureAdapter

adapter = FeatureAdapter(
    student_dim=256,
    teacher_dim=768,
    adapter_type="mlp",
)

student_feat = torch.randn(4, 256)
aligned_feat = adapter(student_feat)  # [4, 768]
```

#### StudentDetector

```python
class StudentDetector(nn.Module):
    """
    Student Detector

    完整的學生檢測模型：
    - Backbone: 特徵提取
    - Neck (FPN): 多尺度融合
    - Head: 檢測預測
    - Feature Adapter: 與 teacher 對齊
    """
```

**使用示例：**

```python
from mtkd_framework.models import StudentDetector

student = StudentDetector(
    backbone_config={"backbone_type": "resnet50", "pretrained": True},
    head_config={"num_classes": 1, "num_queries": 100},
    dino_teacher_dim=768,
)

outputs = student(images, return_adapted_features=True)
# outputs = {
#     "boxes": [...],
#     "logits": [...],
#     "adapted_features": [...],  # 用於特徵對齊
# }
```

---

### 5. MTKD Model (`models/mtkd_model.py`)

整合所有組件的完整模型。

#### MTKDModel

```python
class MTKDModel(nn.Module):
    """
    Multi-Teacher Knowledge Distillation Model

    整合：
    1. DINO Feature Teacher
    2. Ensemble Detection Teachers
    3. Student Detector
    4. MTKD Loss
    """
```

**主要方法：**

| 方法 | 說明 |
|------|------|
| `forward()` | 完整前向傳播，返回輸出和損失 |
| `training_step()` | 訓練步驟，返回損失和損失字典 |
| `inference()` | 推理，返回檢測結果 |
| `get_trainable_parameters()` | 獲取可訓練參數 |

**使用示例：**

```python
from mtkd_framework import MTKDModel

model = MTKDModel(
    student_config={...},
    dino_teacher_config={"model_name": "vit_base"},
    ensemble_config={"teacher_weights": [0.6, 0.4]},
    loss_config={
        "feature_weight": 1.0,
        "prediction_weight": 2.0,
    },
)

# 訓練
loss, loss_dict = model.training_step(images, targets, epoch=epoch)

# 推理
detections = model.inference(images, score_threshold=0.5)
```

---

### 6. Combined Loss (`losses/combined_loss.py`)

組合所有損失的模組。

#### MTKDLoss

```python
class MTKDLoss(nn.Module):
    """
    組合 MTKD 損失

    包含：
    - Feature Alignment Loss
    - Prediction Alignment Loss
    - Detection Loss (可選)

    支持動態權重調整和 warmup。
    """
```

**損失公式：**

$$L_{total} = \alpha \cdot L_{feat} + \beta \cdot L_{pred} + \gamma \cdot L_{det}$$

其中：
- $L_{feat}$: 特徵對齊損失
- $L_{pred}$: 預測對齊損失
- $L_{det}$: 檢測損失（與 ground truth）
- $\alpha, \beta, \gamma$: 權重係數

**使用示例：**

```python
from mtkd_framework.losses import MTKDLoss

loss_fn = MTKDLoss(
    feature_loss_config={"loss_type": "cosine"},
    prediction_loss_config={"box_loss_type": "giou"},
    feature_weight=1.0,
    prediction_weight=2.0,
    warmup_epochs=5,
    weight_schedule="cosine",
)

total_loss, loss_dict = loss_fn(
    student_features=student_feat,
    dino_teacher_features=dino_feat,
    student_predictions=student_pred,
    ensemble_teacher_predictions=ensemble_pred,
    epoch=current_epoch,
)
```

#### UncertaintyWeightedMTKDLoss

```python
class UncertaintyWeightedMTKDLoss(MTKDLoss):
    """
    基於不確定性的自動加權損失

    使用可學習的權重參數，根據同方差不確定性自動學習損失權重。
    參考: "Multi-Task Learning Using Uncertainty to Weigh Losses"
    """
```

**損失公式：**

$$L_{total} = \sum_i \frac{1}{2\sigma_i^2} L_i + \log \sigma_i$$

其中 $\sigma_i$ 是可學習的參數。

---

## 函數參考

### 工具函數 (`utils/helpers.py`)

#### save_checkpoint

```python
def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    save_path: str,
    scheduler: Optional = None,
    extra_info: Optional[Dict] = None,
):
    """保存訓練 checkpoint"""
```

#### load_checkpoint

```python
def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    optimizer: Optional = None,
    scheduler: Optional = None,
    strict: bool = True,
) -> Dict:
    """載入訓練 checkpoint"""
```

#### AverageMeterDict

```python
class AverageMeterDict:
    """追蹤多個指標的平均值"""

    def update(self, values: Dict[str, float], n: int = 1): ...
    def get_averages(self) -> Dict[str, float]: ...
```

#### EarlyStopping

```python
class EarlyStopping:
    """Early Stopping 機制"""

    def __call__(self, score: float) -> bool:
        """返回是否應該停止訓練"""
```

---

## 配置說明

### 默認配置

```python
config = {
    "model": {
        "num_classes": 1,
        "student_config": {
            "backbone_config": {
                "backbone_type": "resnet50",
                "pretrained": True,
            },
            "head_config": {
                "num_classes": 1,
                "num_queries": 100,
            },
        },
        "dino_teacher_config": {
            "model_name": "vit_base",
            "patch_size": 16,
        },
        "ensemble_config": {
            "fusion_method": "wbf",
            "fusion_config": {"iou_threshold": 0.55},
        },
        "loss_config": {
            "feature_weight": 1.0,
            "prediction_weight": 2.0,
            "warmup_epochs": 5,
        },
    },
    "training": {
        "epochs": 100,
        "batch_size": 8,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
    },
}
```

---

## 訓練指南

### 完整訓練流程

```python
import torch
from mtkd_framework import MTKDModel
from mtkd_framework.train import MTKDTrainer, get_default_config

# 1. 準備配置
config = get_default_config()
config["training"]["epochs"] = 100
config["training"]["batch_size"] = 8

# 2. 創建數據載入器
train_loader = create_your_dataloader(...)
val_loader = create_your_dataloader(...)

# 3. 創建模型
model = MTKDModel(
    student_config=config["model"]["student_config"],
    dino_teacher_config=config["model"]["dino_teacher_config"],
    loss_config=config["model"]["loss_config"],
)

# 載入預訓練的 DINO
model.dino_teacher.load_pretrained("dino_vitb16.pth")

# 添加預訓練的檢測 teachers
model.ensemble_teachers.add_teacher(load_detection_model("teacher1.pth"))
model.ensemble_teachers.add_teacher(load_detection_model("teacher2.pth"))

# 4. 創建訓練器
trainer = MTKDTrainer(
    config=config,
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
)

# 5. 開始訓練
trainer.train()
```

### 訓練技巧

1. **學習率調度**: 使用 cosine annealing 配合 warmup
2. **梯度裁剪**: 設置 `gradient_clip_max_norm=1.0`
3. **混合精度**: 啟用 `mixed_precision=True` 加速訓練
4. **損失權重 warmup**: 設置 `warmup_epochs=5` 逐漸增加 KD 損失權重

---

## 自定義擴展

### 自定義 Student Backbone

```python
import torch.nn as nn
from mtkd_framework.models import StudentDetector

class CustomBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # 你的自定義架構
        self.layers = nn.Sequential(...)
        self._out_channels = [64, 128, 256, 512]

    @property
    def out_channels_list(self):
        return self._out_channels

    def forward(self, x):
        features = {}
        # 返回多尺度特徵
        return features

# 使用自定義 backbone
student = StudentDetector(
    custom_backbone=CustomBackbone(),
    head_config={...},
)
```

### 自定義損失函數

```python
from mtkd_framework.losses import FeatureAlignmentLoss

class CustomFeatureLoss(FeatureAlignmentLoss):
    def __init__(self, *args, custom_param=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_param = custom_param

    def forward(self, student_feat, teacher_feat, mask=None):
        # 自定義損失計算
        base_loss = super().forward(student_feat, teacher_feat, mask)
        custom_term = ...
        return base_loss + self.custom_param * custom_term
```

### 添加新的 Teacher

```python
from mtkd_framework.models import TeacherEnsemble

# 創建 ensemble
ensemble = TeacherEnsemble(num_classes=1)

# 添加自定義 teacher
class MyDetector(nn.Module):
    def forward(self, images):
        # 返回 {"boxes": ..., "scores": ..., "labels": ...}
        return predictions

teacher = MyDetector()
teacher.load_state_dict(torch.load("my_detector.pth"))
ensemble.add_teacher(teacher, weight=0.5)
```

---

## 常見問題

### Q1: 如何處理 teacher 和 student 特徵維度不匹配？

使用 `FeatureAdapter` 自動對齊維度：

```python
from mtkd_framework.models.student_model import FeatureAdapter

adapter = FeatureAdapter(
    student_dim=256,
    teacher_dim=768,
    adapter_type="mlp",
)
```

### Q2: 如何只使用特徵對齊或預測對齊？

調整損失權重：

```python
loss_config = {
    "feature_weight": 1.0,  # 只用特徵對齊
    "prediction_weight": 0.0,  # 禁用預測對齊
}
```

### Q3: 如何處理不同數量的預測？

使用 `HungarianMatchingLoss` 進行匹配：

```python
from mtkd_framework.losses import HungarianMatchingLoss

loss_fn = HungarianMatchingLoss(
    box_cost_weight=5.0,
    class_cost_weight=2.0,
)
```

---

## YOLO Student 整合指南

> ⚠️ **實作狀態：規劃中 (Planning Stage)**
>
> 本章節為 YOLO 整合的設計規格書，尚未有實際 Python 實作。
>
> | 項目 | 狀態 |
> |------|------|
> | YOLOv8Teacher | ❌ 待實作 |
> | YOLOv11StudentDetector | ❌ 待實作 |
> | YOLOOutputWrapper | ❌ 待實作 |
> | YOLOFeatureAdapter | ❌ 待實作 |
>
> 目前 MTKD 框架已實作的 Student 為 DETR-like 架構（`StudentDetector`）。

本章節說明如何將 **YOLOv11** 作為 Student 整合到 MTKD 框架中，從 **DINO Teacher**（特徵）和 **YOLOv8 Teacher**（預測）學習。

### 架構設計理念

MTKD 框架採用雙 Teacher 設計，各司其職：

| 角色 | 模型 | 輸出 | 用途 |
|------|------|------|------|
| **Feature Teacher** | DINO ViT-B (Frozen .pth) | CLS token + Patch tokens | Feature Alignment |
| **Detection Teacher** | YOLOv8 (Frozen .pt) | Boxes + Scores + Labels | Prediction Alignment |
| **Student** | YOLOv11 (Trainable) | Features + Predictions | 學習兩者 |

**為什麼選擇 YOLOv11 作為 Student？**

| 比較項目 | YOLOv8 | YOLOv11 |
|---------|--------|---------|
| 核心模組 | C2f | C3k2 (更高效的梯度流) |
| 注意力機制 | 無 | C2PSA (Position-Sensitive Attention) |
| 訓練收斂速度 | 較慢 (約 178 epochs 達到 0.01 loss) | 較快 (約 36 epochs 達到相同 loss) |
| 輸出格式 | boxes, scores, labels | 與 YOLOv8 相容 |
| 推薦用途 | **作為成熟的 Teacher** | **作為學習能力強的 Student** |

### 整體架構

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Input Image (B, 3, H, W)                        │
│                                       │                                      │
│         ┌─────────────────────────────┼─────────────────────────┐           │
│         │                             │                         │           │
│         ▼                             ▼                         ▼           │
│  ┌──────────────────┐    ┌─────────────────────────┐   ┌──────────────────┐│
│  │  DINO Teacher    │    │   YOLOv11 Student       │   │  YOLOv8 Teacher  ││
│  │  (Frozen .pth)   │    │     (Trainable)         │   │  (Frozen .pt)    ││
│  │                  │    │                         │   │                  ││
│  │  ViT-B/16        │    │  ┌─────────────┐        │   │  Backbone        ││
│  │  patch_size=16   │    │  │  Backbone   │        │   │  Neck            ││
│  │                  │    │  │ (C3k2+C2PSA)│        │   │  Head            ││
│  │  輸出:           │    │  └──────┬──────┘        │   │                  ││
│  │  • cls_token     │    │         │               │   │  輸出:           ││
│  │    (B, 768)      │    │  ┌──────┴──────┐        │   │  • boxes         ││
│  │  • patch_tokens  │    │  │    Neck     │        │   │    (B, N, 4)     ││
│  │    (B, 196, 768) │    │  │   (PANet)   │        │   │  • scores        ││
│  │                  │    │  └──────┬──────┘        │   │    (B, N)        ││
│  └────────┬─────────┘    │         │               │   │  • labels        ││
│           │              │  ┌──────┴──────┐        │   │    (B, N)        ││
│           │              │  │  P3 P4 P5   │◄───┐   │   │                  ││
│           │              │  │  Features   │    │   │   └────────┬─────────┘│
│           │              │  └──────┬──────┘    │   │            │          │
│           │              │         │           │   │            │          │
│           │              │  ┌──────┴──────┐    │   │            │          │
│           │              │  │    Head     │    │   │            │          │
│           │              │  │ (Decoupled) │    │   │            │          │
│           │              │  └──────┬──────┘    │   │            │          │
│           │              │         │           │   │            │          │
│           │              │  ┌──────┴──────┐    │   │            │          │
│           │              │  │    NMS      │    │   │            │          │
│           │              │  └──────┬──────┘    │   │            │          │
│           │              │         │           │   │            │          │
│           │              │  輸出:  │           │   │            │          │
│           │              │  • boxes (B, M, 4)  │   │            │          │
│           │              │  • scores (B, M)    │   │            │          │
│           │              │  • labels (B, M)    │   │            │          │
│           │              └─────────┼───────────┘   │            │          │
│           │                        │               │            │          │
│           │              ┌─────────┴─────────┐     │            │          │
│           │              │                   │     │            │          │
│           ▼              ▼                   ▼     │            ▼          │
│  ┌────────────────────────────────────────────────────────────────────────┐│
│  │                           MTKD Loss                                     ││
│  │                                                                         ││
│  │   ┌────────────────────────┐      ┌────────────────────────────────┐   ││
│  │   │   Feature Alignment    │      │    Prediction Alignment        │   ││
│  │   │                        │      │                                │   ││
│  │   │  DINO cls_token        │      │  YOLOv8 Teacher predictions    │   ││
│  │   │       ↕                │      │         ↕                      │   ││
│  │   │  YOLO11 P4 (adapted)   │      │  YOLOv11 Student predictions   │   ││
│  │   │                        │      │                                │   ││
│  │   │  • Cosine Similarity   │      │  • Hungarian Matching          │   ││
│  │   │  • L2 Distance         │      │  • GIoU Loss (boxes)           │   ││
│  │   │                        │      │  • KL Divergence (logits)      │   ││
│  │   └────────────────────────┘      └────────────────────────────────┘   ││
│  │                                                                         ││
│  │   L_total = λ_feat × L_feature + λ_pred × L_prediction                  ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

### 資料流詳解

```
1. Input Image → 同時輸入三個模型

2. DINO Teacher (Frozen):
   Image (B, 3, 224, 224)
     → Patch Embedding (16×16)
     → 12 Transformer Blocks
     → Output: cls_token (B, 768), patch_tokens (B, 196, 768)

3. YOLOv8 Teacher (Frozen):
   Image (B, 3, 640, 640)
     → Backbone → Neck → Head → NMS
     → Output: boxes, scores, labels (數量不固定)

4. YOLOv11 Student (Trainable):
   Image (B, 3, 640, 640)
     → Backbone (提取 P4 特徵用於 Feature Alignment)
     → Neck → Head → NMS
     → Output: features (P4), boxes, scores, labels

5. Loss Computation:
   L_feature = cosine_loss(adapt(YOLO11_P4), DINO_cls)
   L_prediction = hungarian_match(YOLO11_pred, YOLOv8_pred)
   L_total = λ_feat × L_feature + λ_pred × L_prediction
```

### 維度對照表

| 階段 | Tensor | Shape | 說明 |
|-----|--------|-------|------|
| **DINO Teacher 輸入** | image | (B, 3, 224, 224) | 需要 resize |
| **DINO CLS token** | cls_token | (B, 768) | 全局語義特徵 |
| **DINO Patch tokens** | patch_tokens | (B, 196, 768) | 14×14 空間特徵 |
| **YOLO 輸入** | image | (B, 3, 640, 640) | 原始輸入尺寸 |
| **YOLOv11 P4 特徵** | P4 | (B, 512, 40, 40) | stride=16 |
| **P4 Adapted** | adapted_P4 | (B, 768) | GAP 後投影 |
| **YOLOv8/v11 預測** | predictions | 變長 | NMS 後數量不固定 |

### YOLO vs DETR 格式對比

| 特性 | DETR (當前實作) | YOLO | 解決方案 |
|-----|----------------|------|---------|
| 預測數量 | 固定 (num_queries=100) | 不固定 (NMS 後) | Hungarian Matching |
| Box 格式 | cxcywh normalized | xyxy 或 cxcywh | 格式轉換層 |
| Logits | [N, C+1] 含背景類 | [N, C] 或 objectness 分開 | 格式統一 |
| 特徵尺度 | 單尺度 (來自 Decoder) | 多尺度 P3/P4/P5 | 使用 P4 (stride=16) |

---

### YOLOv8Teacher

封裝凍結的 YOLOv8 模型作為 Detection Teacher：

```python
class YOLOv8Teacher(nn.Module):
    """
    YOLOv8 Teacher for Prediction Alignment

    載入預訓練的 YOLOv8 .pt 權重，完全凍結，
    只輸出預測結果供 Student 學習。
    """

    def __init__(
        self,
        weights_path: str,  # .pt 檔案路徑
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 300,
        device: str = "cuda",
    ):
        super().__init__()
        from ultralytics import YOLO

        # 載入 YOLOv8 模型
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

        # 完全凍結
        for param in self.model.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: (B, 3, H, W) - 已正規化的圖像

        Returns:
            {
                "boxes": List[Tensor],   # 每張圖的 boxes [N_i, 4] (xyxy)
                "scores": List[Tensor],  # 每張圖的 scores [N_i]
                "labels": List[Tensor],  # 每張圖的 labels [N_i]
            }
        """
        # Ultralytics YOLO 推理
        results = self.model.predict(
            images,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            verbose=False,
        )

        # 解析結果
        boxes_list = []
        scores_list = []
        labels_list = []

        for result in results:
            boxes = result.boxes
            boxes_list.append(boxes.xyxy)      # (N, 4)
            scores_list.append(boxes.conf)     # (N,)
            labels_list.append(boxes.cls)      # (N,)

        return {
            "boxes": boxes_list,
            "scores": scores_list,
            "labels": labels_list,
        }
```

### YOLOv11StudentDetector

**推薦使用 YOLOv11** 作為 Student，因為其 C3k2 和 C2PSA 模組提供更好的學習能力：

```python
class YOLOv11StudentDetector(nn.Module):
    """
    YOLOv11 Student Detector for MTKD

    可訓練的 YOLOv11 模型，同時輸出：
    1. P4 特徵 → 用於 Feature Alignment (對齊 DINO)
    2. 預測結果 → 用於 Prediction Alignment (對齊 YOLOv8 Teacher)
    """

    def __init__(
        self,
        model_variant: str = "yolo11n",  # yolo11n/s/m/l/x
        num_classes: int = 1,
        dino_dim: int = 768,
        pretrained: bool = True,
        conf_threshold: float = 0.001,  # 訓練時用低閾值
        iou_threshold: float = 0.65,
    ):
        super().__init__()
        from ultralytics import YOLO

        # 載入 YOLOv11
        if pretrained:
            self.model = YOLO(f"{model_variant}.pt")
        else:
            self.model = YOLO(f"{model_variant}.yaml")

        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # P4 特徵適配器 (512 → 768)
        # YOLOv11 P4 通道數因模型大小而異
        p4_channels = self._get_p4_channels(model_variant)
        self.feature_adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(p4_channels, dino_dim),
            nn.LayerNorm(dino_dim),
        )

        # 註冊 hook 提取 P4 特徵
        self.p4_features = None
        self._register_hooks()

    def _get_p4_channels(self, variant: str) -> int:
        """根據模型變體返回 P4 通道數"""
        channels_map = {
            "yolo11n": 256,
            "yolo11s": 256,
            "yolo11m": 512,
            "yolo11l": 512,
            "yolo11x": 512,
        }
        return channels_map.get(variant, 512)

    def _register_hooks(self):
        """註冊 forward hook 提取 P4 特徵"""
        def hook_fn(module, input, output):
            self.p4_features = output

        # P4 位於 neck 的特定層（需要根據實際模型結構調整）
        # 這裡假設使用 Ultralytics 的標準結構
        # 實際使用時需要根據 model.model 結構確定正確的層
        pass  # 實作時需要根據具體模型結構設置

    def forward(
        self,
        images: torch.Tensor,
        return_features: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: (B, 3, H, W)
            return_features: 是否返回適配後的特徵

        Returns:
            {
                "boxes": List[Tensor],        # NMS 後的 boxes
                "scores": List[Tensor],       # NMS 後的 scores
                "labels": List[Tensor],       # NMS 後的 labels
                "adapted_features": Tensor,   # (B, 768) - 用於 Feature Alignment
            }
        """
        # YOLO forward（同時觸發 hook 提取 P4）
        results = self.model.predict(
            images,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        # 解析預測結果
        boxes_list = []
        scores_list = []
        labels_list = []

        for result in results:
            boxes = result.boxes
            boxes_list.append(boxes.xyxy)
            scores_list.append(boxes.conf)
            labels_list.append(boxes.cls)

        outputs = {
            "boxes": boxes_list,
            "scores": scores_list,
            "labels": labels_list,
        }

        # 特徵適配
        if return_features and self.p4_features is not None:
            adapted = self.feature_adapter(self.p4_features)  # (B, 768)
            outputs["adapted_features"] = adapted

        return outputs
```

### YOLOOutputWrapper

將 YOLO 的 NMS 後輸出轉換為 MTKD 統一格式：

```python
class YOLOOutputWrapper(nn.Module):
    """
    包裝 YOLO 輸出為 MTKD 格式

    YOLO 輸出 (NMS 後):
        boxes: List[Tensor] - 每張圖的 [N_i, 4] (xyxy)
        scores: List[Tensor] - 每張圖的 [N_i]
        labels: List[Tensor] - 每張圖的 [N_i]

    MTKD 格式:
        {
            "boxes": [B, max_det, 4] (cxcywh normalized),
            "logits": [B, max_det, C],
            "valid_mask": [B, max_det]
        }
    """

    def __init__(
        self,
        max_detections: int = 100,
        num_classes: int = 1,
        box_format: str = "xyxy",  # YOLO 輸出格式
        normalize_boxes: bool = True,
    ):
        super().__init__()
        self.max_detections = max_detections
        self.num_classes = num_classes
        self.box_format = box_format
        self.normalize_boxes = normalize_boxes

    def forward(
        self,
        yolo_boxes: List[torch.Tensor],  # List of [N_i, 4]
        yolo_scores: List[torch.Tensor],  # List of [N_i]
        yolo_labels: List[torch.Tensor],  # List of [N_i]
        image_sizes: torch.Tensor,        # [B, 2] (H, W)
    ) -> Dict[str, torch.Tensor]:
        B = len(yolo_boxes)
        device = yolo_boxes[0].device

        # 初始化輸出 tensors
        boxes = torch.zeros(B, self.max_detections, 4, device=device)
        logits = torch.zeros(B, self.max_detections, self.num_classes + 1, device=device)
        logits[..., -1] = 1.0  # 背景類初始化為 1
        valid_mask = torch.zeros(B, self.max_detections, dtype=torch.bool, device=device)

        for b in range(B):
            n_det = min(len(yolo_boxes[b]), self.max_detections)
            if n_det == 0:
                continue

            # 複製檢測結果
            b_boxes = yolo_boxes[b][:n_det]
            b_scores = yolo_scores[b][:n_det]
            b_labels = yolo_labels[b][:n_det].long()

            # Box 格式轉換: xyxy → cxcywh
            if self.box_format == "xyxy":
                x1, y1, x2, y2 = b_boxes.unbind(-1)
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                b_boxes = torch.stack([cx, cy, w, h], dim=-1)

            # 正規化 boxes
            if self.normalize_boxes:
                img_h, img_w = image_sizes[b]
                b_boxes = b_boxes / torch.tensor([img_w, img_h, img_w, img_h], device=device)

            boxes[b, :n_det] = b_boxes

            # 將 scores 和 labels 轉換為 logits
            # 使用 logit = log(p / (1-p)) 的逆運算
            for i in range(n_det):
                label = b_labels[i]
                score = b_scores[i].clamp(1e-6, 1 - 1e-6)
                logits[b, i, label] = torch.log(score / (1 - score))
                logits[b, i, -1] = torch.log((1 - score) / score)  # 背景

            valid_mask[b, :n_det] = True

        return {
            "boxes": boxes,
            "logits": logits,
            "valid_mask": valid_mask,
        }
```

### YOLOFeatureAdapter

適配 YOLO 多尺度特徵到 DINO 格式：

```python
class YOLOFeatureAdapter(nn.Module):
    """
    將 YOLO 多尺度特徵適配到 DINO 格式

    策略選項:
    1. "global": 全局平均池化後對齊
    2. "p4": 只使用 P4 (stride=16) 與 DINO patch 對齊
    3. "multi_scale": 多尺度聚合後對齊
    """

    def __init__(
        self,
        yolo_channels: List[int] = [256, 512, 1024],  # P3, P4, P5
        dino_dim: int = 768,
        strategy: str = "p4",
        adapter_type: str = "mlp",
    ):
        super().__init__()
        self.strategy = strategy
        self.dino_dim = dino_dim

        if strategy == "global":
            # 使用 P5 的通道數
            self.adapter = FeatureAdapter(
                student_dim=yolo_channels[-1],
                teacher_dim=dino_dim,
                adapter_type=adapter_type,
            )
        elif strategy == "p4":
            # P4 stride=16，與 DINO patch_size=16 對應
            self.adapter = FeatureAdapter(
                student_dim=yolo_channels[1],  # P4 channels
                teacher_dim=dino_dim,
                adapter_type=adapter_type,
            )
        elif strategy == "multi_scale":
            # 先聚合再適配
            total_channels = sum(yolo_channels)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.adapter = FeatureAdapter(
                student_dim=total_channels,
                teacher_dim=dino_dim,
                adapter_type=adapter_type,
            )

    def forward(
        self,
        yolo_features: Dict[str, torch.Tensor],  # {"P3": ..., "P4": ..., "P5": ...}
        dino_patch_size: Tuple[int, int] = (14, 14),
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            {
                "global_features": [B, dino_dim],  # 與 DINO CLS token 對齊
                "spatial_features": [B, H*W, dino_dim],  # 與 DINO patch tokens 對齊 (可選)
            }
        """
        if self.strategy == "global":
            # 全局特徵
            p5 = yolo_features.get("P5", list(yolo_features.values())[-1])
            global_feat = F.adaptive_avg_pool2d(p5, 1).flatten(1)  # [B, C]
            adapted_global = self.adapter(global_feat)  # [B, dino_dim]

            return {"global_features": adapted_global}

        elif self.strategy == "p4":
            # P4 特徵（stride=16，與 DINO patch 對齊）
            p4 = yolo_features.get("P4", list(yolo_features.values())[1])

            # 全局特徵
            global_feat = F.adaptive_avg_pool2d(p4, 1).flatten(1)
            adapted_global = self.adapter(global_feat)

            # 空間特徵（與 DINO patch tokens 對齊）
            p4_resized = F.interpolate(p4, size=dino_patch_size, mode="bilinear", align_corners=False)
            B, C, H, W = p4_resized.shape
            spatial_feat = p4_resized.flatten(2).transpose(1, 2)  # [B, H*W, C]
            adapted_spatial = self.adapter(spatial_feat)  # [B, H*W, dino_dim]

            return {
                "global_features": adapted_global,
                "spatial_features": adapted_spatial,
            }

        elif self.strategy == "multi_scale":
            # 多尺度聚合
            pooled = []
            for feat in yolo_features.values():
                pooled.append(self.pool(feat).flatten(1))
            concat_feat = torch.cat(pooled, dim=-1)  # [B, sum(channels)]
            adapted_global = self.adapter(concat_feat)

            return {"global_features": adapted_global}
```

### Prediction Alignment 策略

YOLOv11 Student 的預測與 YOLOv8 Teacher 的預測對齊。由於兩者 NMS 後的檢測數量可能不同，使用 **Hungarian Matching** 進行最優配對：

```python
from mtkd_framework.losses import HungarianMatchingLoss

# 建立損失函數
hungarian_loss = HungarianMatchingLoss(
    box_cost_weight=5.0,
    class_cost_weight=2.0,
    box_loss_type="giou",
    class_loss_type="kl",
)

# YOLOv11 Student predictions
student_pred = yolo11_student(images)
# student_pred["boxes"]: List[Tensor] - 每張圖 N_i 個檢測
# student_pred["scores"]: List[Tensor]
# student_pred["labels"]: List[Tensor]

# YOLOv8 Teacher predictions
teacher_pred = yolo8_teacher(images)
# teacher_pred["boxes"]: List[Tensor] - 每張圖 M_i 個檢測
# teacher_pred["scores"]: List[Tensor]
# teacher_pred["labels"]: List[Tensor]

# Hungarian Matching 自動配對不同數量的預測
# 對於每張圖，找到 min(N_i, M_i) 個最優配對
loss, loss_dict = hungarian_loss(student_pred, teacher_pred)
# loss_dict: {"box_loss": ..., "class_loss": ..., "total_loss": ...}
```

### Feature Alignment 策略

YOLOv11 Student 的 P4 特徵與 DINO Teacher 的 CLS token 對齊：

```python
import torch.nn.functional as F

# YOLOv11 P4 特徵（已通過 adapter 投影到 768 維）
student_features = yolo11_student(images)["adapted_features"]  # (B, 768)

# DINO CLS token
dino_output = dino_teacher(images)
dino_cls = dino_output["cls_token"]  # (B, 768)

# Cosine Similarity Loss
feature_loss = 1 - F.cosine_similarity(student_features, dino_cls, dim=-1).mean()

# 或使用 L2 Loss
# feature_loss = F.mse_loss(student_features, dino_cls)
```

**為什麼用 P4 對齊 DINO？**

| 特徵層 | Stride | 對於 640×640 輸入 | 說明 |
|-------|--------|------------------|------|
| P3 | 8 | 80×80 | 太細，語義不足 |
| **P4** | **16** | **40×40** | **與 DINO patch_size=16 對應** |
| P5 | 32 | 20×20 | 過於抽象 |

### 完整使用範例

```python
import torch
import torch.nn.functional as F
from mtkd_framework.losses import HungarianMatchingLoss

# ============================================
# 1. 載入三個模型
# ============================================

# DINO Feature Teacher (Frozen)
from dinov3.models import build_model as build_dino
dino_teacher = build_dino(model_name="vit_base", patch_size=16)
dino_teacher.load_state_dict(torch.load("dino_vitb16.pth"))
dino_teacher.eval()
for param in dino_teacher.parameters():
    param.requires_grad = False

# YOLOv8 Detection Teacher (Frozen)
yolo8_teacher = YOLOv8Teacher(
    weights_path="yolov8_stomata.pt",  # 您的預訓練 YOLOv8 權重
    conf_threshold=0.25,
    iou_threshold=0.45,
)

# YOLOv11 Student (Trainable)
yolo11_student = YOLOv11StudentDetector(
    model_variant="yolo11n",
    num_classes=1,
    dino_dim=768,
    pretrained=True,
)

# ============================================
# 2. 設定損失函數
# ============================================

# Feature Alignment: Cosine Loss
def feature_alignment_loss(student_feat, teacher_feat):
    return 1 - F.cosine_similarity(student_feat, teacher_feat, dim=-1).mean()

# Prediction Alignment: Hungarian Matching
hungarian_loss = HungarianMatchingLoss(
    box_cost_weight=5.0,
    class_cost_weight=2.0,
    box_loss_type="giou",
    class_loss_type="kl",
)

# ============================================
# 3. 訓練迴圈
# ============================================

# 只有 Student 可訓練
optimizer = torch.optim.AdamW(
    yolo11_student.parameters(),
    lr=1e-4,
    weight_decay=1e-4,
)

lambda_feat = 1.0
lambda_pred = 2.0

for epoch in range(100):
    for images, _ in train_loader:
        images = images.cuda()

        # ---- Forward ----
        # DINO: 需要 resize 到 224x224
        dino_images = F.interpolate(images, size=(224, 224), mode="bilinear")
        with torch.no_grad():
            dino_out = dino_teacher(dino_images)
            dino_cls = dino_out["cls_token"]  # (B, 768)

        # YOLOv8 Teacher
        with torch.no_grad():
            teacher_pred = yolo8_teacher(images)

        # YOLOv11 Student
        student_out = yolo11_student(images, return_features=True)
        student_feat = student_out["adapted_features"]  # (B, 768)
        student_pred = {
            "boxes": student_out["boxes"],
            "scores": student_out["scores"],
            "labels": student_out["labels"],
        }

        # ---- Loss ----
        L_feature = feature_alignment_loss(student_feat, dino_cls)
        L_prediction, pred_loss_dict = hungarian_loss(student_pred, teacher_pred)

        loss = lambda_feat * L_feature + lambda_pred * L_prediction

        # ---- Backward ----
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
        print(f"  Feature: {L_feature.item():.4f}")
        print(f"  Prediction: {L_prediction.item():.4f}")

# ============================================
# 4. 儲存訓練好的 Student
# ============================================
torch.save(yolo11_student.state_dict(), "yolo11_student_mtkd.pt")
```

### 配置範例

```python
yolo_mtkd_config = {
    "model": {
        "num_classes": 1,

        # DINO Feature Teacher
        "dino_teacher": {
            "model_name": "vit_base",
            "patch_size": 16,
            "embed_dim": 768,
            "weights_path": "dino_vitb16.pth",
            "frozen": True,  # 完全凍結
        },

        # YOLOv8 Detection Teacher
        "yolo8_teacher": {
            "weights_path": "yolov8_stomata.pt",
            "conf_threshold": 0.25,
            "iou_threshold": 0.45,
            "frozen": True,  # 完全凍結
        },

        # YOLOv11 Student (Trainable)
        "yolo11_student": {
            "model_variant": "yolo11n",  # n/s/m/l/x
            "pretrained": True,
            "dino_dim": 768,
            "conf_threshold": 0.001,  # 訓練時低閾值
            "iou_threshold": 0.65,
        },
    },

    "loss": {
        "feature_weight": 1.0,       # λ_feat
        "prediction_weight": 2.0,    # λ_pred
        "feature_loss_type": "cosine",
        "prediction_loss_type": "hungarian",
        "box_loss_type": "giou",
        "class_loss_type": "kl",
    },

    "training": {
        "epochs": 100,
        "batch_size": 16,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "warmup_epochs": 5,
        "scheduler": "cosine",
    },
}
```

### 注意事項

1. **圖像尺寸**：
   - DINO 需要 224×224 輸入，需要 resize
   - YOLO 使用 640×640（或其他標準尺寸）

2. **NMS 閾值**：
   - Teacher: 正常閾值 (conf=0.25) 產生高品質預測
   - Student: 低閾值 (conf=0.001) 保留更多預測供配對

3. **特徵提取**：
   - 使用 forward hooks 從 YOLOv11 提取 P4 特徵
   - P4 stride=16 與 DINO patch_size=16 對應

4. **梯度流**：
   - DINO Teacher: `requires_grad=False`
   - YOLOv8 Teacher: `requires_grad=False`
   - YOLOv11 Student: `requires_grad=True`（只有這個可訓練）

5. **Box 格式**：Hungarian Matching 內部處理格式轉換

---

## 實作細節與狀態

本章節詳細記錄 MTKD 框架各模組的完整實作細節，包括已實作的功能、API 細節和內部運作機制。

### 實作狀態總覽

> ⚠️ 本 framework 大部分代碼為語言模型生成，經與 DINO Teacher (CVPR 2025) 對照後發現多項架構性問題。
> 詳見 [重構計畫](#重構計畫基於-dino-teacher-的對照修改)。

| 模組 | 檔案 | 實作狀態 | 說明 |
|------|------|---------|------|
| **Feature Alignment Loss** | `losses/feature_alignment.py` | ⚠️ 需重構 | 做的是 per-image global alignment，應改為 per-pixel spatial alignment（見 §0.2） |
| **SpatialAlignHead** | `models/align_head.py` | ❌ 待新增 | 需仿照 `DINO_Teacher/dinoteacher/engine/align_head.py` 實作 Conv2d spatial projection |
| **DINO Feature Teacher** | `models/mtkd_model.py` | ⚠️ 需修復 | 缺少 ImageNet 預處理 + spatial reshape（見 §0.2.2） |
| **Prediction Alignment Loss** | `losses/prediction_alignment.py` | ❌ 需重寫 | 存在 shape mismatch crash；應改為 pseudo-label → detection loss（見 §0.3） |
| **Combined Loss** | `losses/combined_loss.py` | ⚠️ 需精簡 | `AdaptiveMTKDLoss`、`UncertaintyWeightedMTKDLoss` 為 dead code |
| **Student Model (DETR)** | `models/student_model.py` | 🗑️ Dead code | 預設用 YOLO，DETR 路徑從未使用 |
| **YOLO Wrappers** | `models/yolo_wrappers.py` | ⚠️ 需修改 | 移除 global pool，改為輸出 spatial feature map |
| **Teacher Ensemble** | `models/teacher_ensemble.py` | ⚠️ 需調整 | 需加 score threshold 篩選產出 pseudo-labels |
| **MTKD Model** | `models/mtkd_model.py` | ⚠️ 需重構 | `_compute_detection_loss` 是空殼（return 0），需接 YOLO loss |
| **Training Pipeline** | `train.py` | ⚠️ 需重構 | 缺少 progressive training / burn-in / weak-strong aug 分離 |
| **YOLOv8Teacher** | `models/yolo_wrappers.py` | ✅ 已實作 | `YOLODetectionTeacher` |
| **YOLOv11StudentDetector** | `models/yolo_wrappers.py` | ✅ 已實作 | `YOLOStudentDetector`（需修改 feature 輸出方式） |

---

### MTKDLoss 變體詳解

#### 1. 標準 MTKDLoss

**檔案**: `losses/combined_loss.py:18-191`

```python
class MTKDLoss(nn.Module):
    """
    標準 MTKD 組合損失

    特性:
    - 動態權重調整（Warmup + Schedule）
    - 支援檢測損失（與 Ground Truth）
    - 損失項詳細追蹤
    """
```

**初始化參數詳解**:

| 參數 | 類型 | 默認值 | 說明 |
|------|------|--------|------|
| `feature_loss_config` | dict | `{}` | FeatureAlignmentLoss 配置 |
| `prediction_loss_config` | dict | `{}` | PredictionAlignmentLoss 配置 |
| `feature_weight` | float | 1.0 | 特徵對齊損失權重 |
| `prediction_weight` | float | 1.0 | 預測對齊損失權重 |
| `detection_weight` | float | 0.0 | 檢測損失權重（0 表示禁用）|
| `warmup_epochs` | int | 0 | 權重 warmup 輪數 |
| `weight_schedule` | str | "constant" | 權重調度策略 |
| `min_weight_ratio` | float | 0.1 | 最小權重比例 |

**權重調度策略**:

```python
# 支援的調度類型
weight_schedule: Literal["constant", "linear", "cosine"] = "constant"

# Warmup 期間的權重計算
if epoch < warmup_epochs:
    warmup_factor = epoch / warmup_epochs
else:
    warmup_factor = 1.0

# 調度計算
if weight_schedule == "linear":
    schedule_factor = 1 - (1 - min_weight_ratio) * (epoch / total_epochs)
elif weight_schedule == "cosine":
    schedule_factor = min_weight_ratio + (1 - min_weight_ratio) * (1 + cos(π * epoch / total_epochs)) / 2
else:
    schedule_factor = 1.0

final_weight = base_weight * warmup_factor * schedule_factor
```

**Forward 簽名**:

```python
def forward(
    self,
    student_features: torch.Tensor,          # [B, D] 或 [B, N, D]
    dino_teacher_features: torch.Tensor,      # [B, D] 或 [B, N, D]
    student_predictions: Dict[str, Tensor],   # {"boxes": [B, N, 4], "logits": [B, N, C]}
    ensemble_teacher_predictions: Dict,       # 同上
    targets: Optional[List[Dict]] = None,     # Ground truth（用於檢測損失）
    epoch: int = 0,
    total_epochs: int = 100,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """
    Returns:
        total_loss: 加權總損失
        loss_dict: {
            "feature_align_loss": ...,
            "pred_align_total_loss": ...,
            "pred_align_box_loss": ...,
            "pred_align_class_loss": ...,
            "detection_loss": ...,  # 如果啟用
            "total_loss": ...,
            "feature_weight": ...,
            "prediction_weight": ...,
        }
    """
```

#### 2. AdaptiveMTKDLoss

**檔案**: `losses/combined_loss.py:194-293`

自動調整損失權重，根據各損失項的相對大小動態平衡。

```python
class AdaptiveMTKDLoss(MTKDLoss):
    """
    自適應 MTKD 損失

    特性:
    - 使用 EMA 追蹤各損失項的統計資訊
    - 根據損失的標準差自動調整權重
    - 防止單一損失項主導訓練
    """
```

**額外初始化參數**:

| 參數 | 類型 | 默認值 | 說明 |
|------|------|--------|------|
| `ema_decay` | float | 0.999 | EMA 衰減係數 |
| `loss_scale_method` | str | "std" | 縮放方法 ("std", "mean", "max") |

**自適應權重計算**:

```python
# EMA 更新
self.loss_mean = ema_decay * self.loss_mean + (1 - ema_decay) * current_loss
self.loss_sq_mean = ema_decay * self.loss_sq_mean + (1 - ema_decay) * current_loss ** 2

# 計算標準差
std = sqrt(self.loss_sq_mean - self.loss_mean ** 2)

# 權重縮放（反比於標準差）
adaptive_weight = base_weight / (std + epsilon)
```

#### 3. UncertaintyWeightedMTKDLoss

**檔案**: `losses/combined_loss.py:296-413`

基於同方差不確定性的可學習損失權重。

```python
class UncertaintyWeightedMTKDLoss(MTKDLoss):
    """
    基於不確定性的 MTKD 損失

    理論基礎:
    - 論文: "Multi-Task Learning Using Uncertainty to Weigh Losses"
    - 使用可學習的 log_variance 參數
    - 損失項的權重與其不確定性成反比

    公式:
    L_total = Σ (1 / (2 * exp(log_var_i))) * L_i + log_var_i / 2
    """
```

**可學習參數**:

```python
# 在 __init__ 中初始化
self.log_var_feature = nn.Parameter(torch.zeros(1))   # log(σ²) for feature loss
self.log_var_prediction = nn.Parameter(torch.zeros(1))  # log(σ²) for prediction loss
self.log_var_detection = nn.Parameter(torch.zeros(1))   # log(σ²) for detection loss
```

**損失計算**:

```python
# 不確定性加權
precision_feature = torch.exp(-self.log_var_feature)
weighted_feature_loss = precision_feature * feature_loss + self.log_var_feature / 2

precision_prediction = torch.exp(-self.log_var_prediction)
weighted_pred_loss = precision_prediction * pred_loss + self.log_var_prediction / 2

total_loss = weighted_feature_loss + weighted_pred_loss + ...
```

**訓練建議**:

```python
# 需要將 log_var 參數加入 optimizer
optimizer = torch.optim.AdamW([
    {"params": model.student.parameters(), "lr": 1e-4},
    {"params": loss_fn.parameters(), "lr": 1e-3},  # 較高學習率
])
```

---

### WeightedBoxFusion (WBF) 詳解

**檔案**: `models/teacher_ensemble.py:21-178`

完整的 WBF 實作，用於融合多個檢測模型的預測。

```python
class WeightedBoxFusion(nn.Module):
    """
    Weighted Box Fusion

    與 NMS 的區別:
    - NMS: 刪除重疊的 boxes，只保留最高分
    - WBF: 融合重疊的 boxes，取加權平均

    優點:
    - 利用多模型的互補資訊
    - 產生更穩定、準確的 boxes
    - 融合後的置信度更可靠
    """
```

**初始化參數**:

| 參數 | 類型 | 默認值 | 說明 |
|------|------|--------|------|
| `iou_threshold` | float | 0.55 | 融合 IoU 閾值 |
| `skip_box_thr` | float | 0.0 | 忽略低於此分數的 boxes |
| `weights` | List[float] | None | 各模型權重 |
| `conf_type` | str | "avg" | 置信度計算方式 |

**置信度計算方式**:

```python
# conf_type 選項
"avg"       # 加權平均
"max"       # 取最大值
"box_and_model_avg"  # Box 數量 + 模型權重加權
"absent_model_aware_avg"  # 考慮缺失模型的平均
```

**WBF 算法流程（實際程式碼邏輯）**:

```python
def forward(self, boxes_list, scores_list, labels_list):
    """
    Args:
        boxes_list: List[Tensor] - 每個模型的 boxes [N_i, 4]
        scores_list: List[Tensor] - 每個模型的 scores [N_i]
        labels_list: List[Tensor] - 每個模型的 labels [N_i]

    Returns:
        fused_boxes: Tensor [M, 4]
        fused_scores: Tensor [M]
        fused_labels: Tensor [M]
    """
    # 1. 按類別分組
    for label in unique_labels:
        class_boxes = filter_by_label(all_boxes, label)

        # 2. 按分數排序
        sorted_boxes = sort_by_score(class_boxes)

        # 3. 聚類重疊 boxes
        clusters = []
        for box in sorted_boxes:
            matched = False
            for cluster in clusters:
                if iou(box, cluster.representative) > iou_threshold:
                    cluster.add(box)
                    matched = True
                    break
            if not matched:
                clusters.append(new_cluster(box))

        # 4. 對每個 cluster 計算加權平均
        for cluster in clusters:
            weights = [model_weights[box.model_id] * box.score for box in cluster]
            fused_box = weighted_average(cluster.boxes, weights)
            fused_score = calculate_confidence(cluster, conf_type)
            results.append((fused_box, fused_score, label))

    return fused_boxes, fused_scores, fused_labels
```

---

### Soft-NMS 詳解

**檔案**: `models/teacher_ensemble.py:181-276`

Soft-NMS 作為 WBF 的替代方案，通過降低重疊 box 分數而非刪除。

```python
class SoftNMS(nn.Module):
    """
    Soft Non-Maximum Suppression

    與傳統 NMS 的區別:
    - 傳統 NMS: 直接刪除重疊的低分 boxes
    - Soft-NMS: 根據 IoU 降低重疊 boxes 的分數

    優點:
    - 保留密集物體的檢測
    - 減少漏檢
    """
```

**初始化參數**:

| 參數 | 類型 | 默認值 | 說明 |
|------|------|--------|------|
| `iou_threshold` | float | 0.3 | 開始降分的 IoU 閾值 |
| `score_threshold` | float | 0.001 | 最終保留的分數閾值 |
| `sigma` | float | 0.5 | Gaussian 衰減參數 |
| `method` | str | "gaussian" | 衰減方法 |

**衰減方法**:

```python
# method = "linear"
if iou > iou_threshold:
    score = score * (1 - iou)

# method = "gaussian" (更平滑)
score = score * exp(-iou^2 / sigma)
```

---

### TeacherEnsemble 詳解

**檔案**: `models/teacher_ensemble.py:279-655`

管理多個 Teacher 模型並融合預測的完整實作。

```python
class TeacherEnsemble(nn.Module):
    """
    Teacher Ensemble 模組

    特性:
    - 動態添加/移除 Teacher
    - 支援從 checkpoints 批量載入
    - 自動管理模型權重
    - 可選 WBF 或 Soft-NMS 融合
    """
```

**主要方法**:

```python
def add_teacher(
    self,
    model: nn.Module,
    weight: float = 1.0,
    name: Optional[str] = None,
):
    """
    添加 Teacher 模型

    Args:
        model: 檢測模型（必須輸出 boxes, scores, labels）
        weight: 模型權重
        name: 模型名稱（用於識別）
    """
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    self.teachers.append(model)
    self.weights.append(weight)

def load_teachers_from_checkpoints(
    self,
    checkpoint_paths: List[str],
    model_class: Type[nn.Module],
    weights: Optional[List[float]] = None,
    **model_kwargs,
):
    """
    從多個 checkpoint 載入 Teachers

    Args:
        checkpoint_paths: checkpoint 路徑列表
        model_class: 模型類別
        weights: 各模型權重
        **model_kwargs: 模型初始化參數
    """

def forward(
    self,
    images: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    獲取融合預測

    Returns:
        {
            "boxes": [B, max_det, 4],
            "scores": [B, max_det],
            "labels": [B, max_det],
            "valid_mask": [B, max_det],
        }
    """
```

---

### MTKDTrainer 訓練器詳解

**檔案**: `train.py:139-437`

完整的訓練管線實作。

```python
class MTKDTrainer:
    """
    MTKD 訓練器

    特性:
    - 混合精度訓練 (AMP)
    - 梯度裁剪
    - 學習率調度（Warmup + Cosine Annealing）
    - Early Stopping
    - Checkpoint 保存/載入
    - 訓練指標追蹤
    """
```

**初始化參數**:

| 參數 | 類型 | 默認值 | 說明 |
|------|------|--------|------|
| `config` | dict | 必填 | 訓練配置 |
| `model` | MTKDModel | 必填 | MTKD 模型 |
| `train_loader` | DataLoader | 必填 | 訓練數據 |
| `val_loader` | DataLoader | None | 驗證數據 |
| `device` | str | "cuda" | 運算設備 |

**配置結構（get_default_config）**:

```python
def get_default_config() -> Dict:
    return {
        "model": {
            "num_classes": 1,
            "student_config": {
                "backbone_config": {
                    "backbone_type": "resnet50",
                    "pretrained": True,
                    "out_channels": [256, 512, 1024, 2048],
                },
                "head_config": {
                    "num_classes": 1,
                    "num_queries": 100,
                    "hidden_dim": 256,
                    "num_heads": 8,
                    "num_encoder_layers": 6,
                    "num_decoder_layers": 6,
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
                    "conf_type": "avg",
                },
            },
            "loss_config": {
                "feature_weight": 1.0,
                "prediction_weight": 2.0,
                "detection_weight": 0.0,
                "warmup_epochs": 5,
                "weight_schedule": "cosine",
                "feature_loss_config": {
                    "loss_type": "cosine",
                    "normalize": True,
                },
                "prediction_loss_config": {
                    "box_loss_type": "giou",
                    "class_loss_type": "kl",
                    "temperature": 4.0,
                },
            },
        },
        "training": {
            "epochs": 100,
            "batch_size": 8,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "gradient_clip_max_norm": 1.0,
            "mixed_precision": True,
            "lr_scheduler": "cosine",
            "warmup_epochs": 5,
            "min_lr": 1e-6,
        },
        "validation": {
            "val_interval": 1,
            "save_best": True,
            "metric": "loss",
        },
        "checkpoint": {
            "save_dir": "./checkpoints",
            "save_interval": 10,
            "keep_last_n": 5,
        },
        "early_stopping": {
            "enabled": True,
            "patience": 20,
            "min_delta": 1e-4,
        },
    }
```

**訓練流程**:

```python
def train(self):
    """
    主訓練循環

    流程:
    1. 初始化 optimizer, scheduler, scaler
    2. 每個 epoch:
       a. 訓練一個 epoch (train_epoch)
       b. 驗證（如果有 val_loader）
       c. 更新 scheduler
       d. 保存 checkpoint
       e. 檢查 early stopping
    """
    for epoch in range(self.start_epoch, self.config["training"]["epochs"]):
        # 訓練
        train_metrics = self.train_epoch(epoch)

        # 驗證
        if self.val_loader and epoch % self.config["validation"]["val_interval"] == 0:
            val_metrics = self.validate(epoch)

        # Scheduler step
        if self.scheduler is not None:
            self.scheduler.step()

        # Checkpoint
        if epoch % self.config["checkpoint"]["save_interval"] == 0:
            self.save_checkpoint(epoch)

        # Early stopping
        if self.early_stopping(val_metrics["loss"]):
            print(f"Early stopping at epoch {epoch}")
            break

def train_epoch(self, epoch: int) -> Dict[str, float]:
    """
    訓練一個 epoch

    使用:
    - Mixed precision (GradScaler)
    - Gradient clipping
    - Loss 追蹤
    """
    self.model.train()
    meter = AverageMeterDict()

    for batch_idx, (images, targets) in enumerate(self.train_loader):
        images = images.to(self.device)

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            loss, loss_dict = self.model.training_step(
                images, targets, epoch=epoch,
                total_epochs=self.config["training"]["epochs"]
            )

        # Backward
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()

        # Gradient clipping
        if self.gradient_clip_max_norm > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip_max_norm
            )

        self.scaler.step(self.optimizer)
        self.scaler.update()

        meter.update(loss_dict)

    return meter.get_averages()
```

---

### HungarianMatchingLoss 詳解

**檔案**: `losses/prediction_alignment.py:396-657`

使用 Hungarian 算法解決預測數量不一致的問題。

```python
class HungarianMatchingLoss(nn.Module):
    """
    Hungarian Matching Loss

    用途:
    - 當 student 和 teacher 預測數量不同時
    - DETR 風格的一對一匹配
    - 自動找出最優配對

    算法:
    1. 計算 cost matrix (box cost + class cost)
    2. 使用 Hungarian 算法找最優匹配
    3. 只對匹配的預測計算損失
    """
```

**Cost Matrix 計算**:

```python
def _compute_cost_matrix(
    self,
    student_pred: Dict[str, Tensor],
    teacher_pred: Dict[str, Tensor],
) -> Tensor:
    """
    計算配對成本矩陣

    Cost = box_cost_weight * box_cost + class_cost_weight * class_cost

    box_cost: 1 - GIoU(student_box, teacher_box)
    class_cost: 1 - CosineSim(student_logit, teacher_logit)
    """
    # Box cost
    student_boxes = student_pred["boxes"]  # [B, N_s, 4]
    teacher_boxes = teacher_pred["boxes"]  # [B, N_t, 4]

    # 計算所有 pairs 的 GIoU
    giou_matrix = self._compute_giou_matrix(student_boxes, teacher_boxes)
    box_cost = 1 - giou_matrix  # [B, N_s, N_t]

    # Class cost
    student_logits = F.softmax(student_pred["logits"], dim=-1)
    teacher_logits = F.softmax(teacher_pred["logits"], dim=-1)

    # Cosine similarity
    s_norm = F.normalize(student_logits, dim=-1)  # [B, N_s, C]
    t_norm = F.normalize(teacher_logits, dim=-1)  # [B, N_t, C]
    class_sim = torch.bmm(s_norm, t_norm.transpose(1, 2))  # [B, N_s, N_t]
    class_cost = 1 - class_sim

    # Total cost
    cost_matrix = self.box_cost_weight * box_cost + self.class_cost_weight * class_cost

    return cost_matrix
```

**Hungarian 求解**:

```python
def _hungarian_matching(self, cost_matrix: Tensor) -> List[Tuple[Tensor, Tensor]]:
    """
    使用 scipy.optimize.linear_sum_assignment 求解

    Returns:
        indices: List[(row_indices, col_indices)] for each batch
    """
    from scipy.optimize import linear_sum_assignment

    indices = []
    for b in range(cost_matrix.shape[0]):
        cost_b = cost_matrix[b].detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_b)
        indices.append((
            torch.tensor(row_ind, device=cost_matrix.device),
            torch.tensor(col_ind, device=cost_matrix.device)
        ))
    return indices
```

---

### 工具類別

#### AverageMeterDict

**用途**: 追蹤多個訓練指標的移動平均

```python
class AverageMeterDict:
    def __init__(self):
        self.meters = {}

    def update(self, values: Dict[str, float], n: int = 1):
        for k, v in values.items():
            if k not in self.meters:
                self.meters[k] = {"sum": 0, "count": 0}
            self.meters[k]["sum"] += v * n
            self.meters[k]["count"] += n

    def get_averages(self) -> Dict[str, float]:
        return {
            k: v["sum"] / v["count"]
            for k, v in self.meters.items()
        }
```

#### EarlyStopping

**用途**: 提前停止訓練以防止過擬合

```python
class EarlyStopping:
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

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

    def _is_improvement(self, score: float) -> bool:
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta
```

---

### 效能考量

#### 記憶體優化

```python
# 1. Teachers 完全凍結，不儲存梯度
for teacher in ensemble_teachers:
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

# 2. 使用 torch.no_grad() 進行 teacher 推理
with torch.no_grad():
    dino_features = dino_teacher(images)
    ensemble_predictions = ensemble_teachers(images)

# 3. 混合精度訓練減少記憶體
with torch.cuda.amp.autocast():
    student_output = student(images)
    loss = loss_fn(...)
```

#### 計算效率

| 操作 | 成本 | 優化方式 |
|------|------|---------|
| DINO 推理 | 高 | 批次處理 + 凍結 |
| WBF | 中 | 向量化操作 |
| Hungarian Matching | O(N³) | 限制 max_detections |
| Feature Adapter | 低 | 簡單線性層 |

---

## 參考文獻

1. DINO: Emerging Properties in Self-Supervised Vision Transformers
2. Knowledge Distillation: A Survey
3. Weighted Boxes Fusion: Ensembling boxes from different object detection models
4. Multi-Task Learning Using Uncertainty to Weigh Losses
5. YOLOv8: Ultralytics YOLO
6. DETR: End-to-End Object Detection with Transformers
7. Soft-NMS: Improving Object Detection With One Line of Code

---

## License

MIT License
