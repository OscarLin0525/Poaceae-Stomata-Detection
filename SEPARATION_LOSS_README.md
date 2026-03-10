# Separation Loss for Stomata Detection

## 概述

**Separation Loss (分離損失)** 是一種新的正則化機制，設計用於確保相鄰 stomata 特徵在空間上保持分離，避免高響應區域"連在一起"。

### 問題背景

在之前的分析中發現：
- DINO 特徵圖上，相鄰 stomata 的高響應區域經常會連接成同一個連通組件
- FFT block 無法有效提供特徵分散效果（merge rate 幾乎不變）
- 需要一個明確的損失項來鼓勵 stomata 之間形成"低谷"（valley），確保中間有間隔

### 解決方案：Valley Separation Loss

對於每對 GT stomata 中心：
1. **採樣連接路徑**：在兩個中心之間採樣 N 個點（預設 5 個）
2. **計算對比度**：對比度 = 該位置到最近 GT 特徵的 L2 距離
3. **檢查谷點**：中點的對比度應該顯著高於端點（即：中點處特徵與 GT 不相似）
4. **懲罰違規**：如果中點對比度不夠高，則增加損失

**數學公式**：
```
對於每對 GT center (i, j):
  - endpoints_contrast = mean(contrast[i], contrast[j])
  - midpoint_contrast = contrast[中點]
  - target = endpoints_contrast × (1 + margin)
  - loss += ReLU(target - midpoint_contrast)
```

## 使用方法

### 1. 基本啟用

在訓練命令中添加 `--separation-loss-weight` 參數：

```bash
python mtkd_framework/run_v2.py \
  --separation-loss-weight 0.5 \
  --epochs 50 \
  --batch-size 8
```

### 2. 完整配置示例

```bash
python mtkd_framework/run_v2.py \
  --dataset-root Stomata_Dataset \
  --image-subdir barley_all/images/train \
  --label-subdir barley_all/labels/train \
  \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-4 \
  \
  --separation-loss-weight 0.5 \
  --separation-target-layer 10 \
  --separation-sample-points 5 \
  --separation-valley-margin 0.2 \
  \
  --feature-align-weight 1.0 \
  --burn-up-epochs 5 \
  --align-target-start 10 \
  \
  --output-dir outputs/mtkd_with_separation
```

### 3. 參數說明

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--separation-loss-weight` | 0.0 | 分離損失的權重（0 表示停用） |
| `--separation-target-layer` | 10 | 從哪個 DINO layer 提取特徵 |
| `--separation-sample-points` | 5 | 連接線上採樣多少個點 |
| `--separation-valley-margin` | 0.2 | 谷點深度要求（20% margin） |

### 4. 推薦設定

**情境 1：輕度正則化**
```bash
--separation-loss-weight 0.1 \
--separation-valley-margin 0.1
```
適合：stomata 間距本身就較大的數據集

**情境 2：中等正則化（推薦）**
```bash
--separation-loss-weight 0.5 \
--separation-valley-margin 0.2
```
適合：一般情況，barley dataset 的預設選擇

**情境 3：強正則化**
```bash
--separation-loss-weight 1.0 \
--separation-valley-margin 0.3 \
--separation-sample-points 7
```
適合：stomata 非常密集，經常黏在一起的困難案例

## 技術細節

### 實現架構

```
mtkd_framework/
├── losses/
│   └── separation.py          # ValleySeparationLoss 類
├── train_v2.py                # MTKDTrainerV2 集成
└── run_v2.py                  # CLI 參數
```

### 關鍵組件

**1. ValleySeparationLoss (`losses/separation.py`)**
- `forward(features, gt_centers_list)`: 計算分離損失
- `_compute_contrast_map()`: 計算每個位置到最近 GT 的距離
- `_sample_line()`: 在兩點間線性採樣

**2. MTKDTrainerV2 輔助方法 (`train_v2.py`)**
- `_extract_dino_layer_features()`: 使用 hook 提取指定 layer 的特徵
- `_extract_gt_centers_from_batch()`: 從 YOLO batch 解析 GT 中心座標

**3. 訓練流程集成**
```python
# 在 _forward_and_loss() 中：
if self.separation_loss_fn is not None:
    # 1. 提取 DINO layer-10 features
    layer_features = self._extract_dino_layer_features(images, layer=10)
    
    # 2. 解析 GT centers
    gt_centers_list = self._extract_gt_centers_from_batch(gt_batch, feature_shape)
    
    # 3. 計算 separation loss
    sep_loss = self.separation_loss_fn(layer_features, gt_centers_list)
    
    # 4. 加權並添加到總損失
    weighted_sep = sep_loss * self.separation_loss_w
    losses.append(weighted_sep)
```

### 損失項監控

訓練時會在日誌中看到：
```
Epoch 10 [50/782] Loss: 8.3245 loss_det: 7.8123 loss_align: 0.4521 loss_separation: 0.0601
```

其中 `loss_separation` 就是分離損失的原始值（未加權）。

## 驗證效果

### 1. 訓練後可視化

使用 `visualize_layer10_fft_compare.py` 的修改版來比較：

```python
# 比較 separation loss ON/OFF
model_off = build_model(separation_loss_weight=0.0, checkpoint="off.pt")
model_on = build_model(separation_loss_weight=0.5, checkpoint="on.pt")

# 提取特徵並計算連通性
connectivity_off = compute_connectivity(model_off, val_images)
connectivity_on = compute_connectivity(model_on, val_images)

print(f"Merge rate OFF: {connectivity_off.mean():.4f}")
print(f"Merge rate ON:  {connectivity_on.mean():.4f}")  # 應該更低
```

### 2. 定量指標

- **merge_rate**: GT center pairs 落在同一連通組件的比例（越低越好）
- **components_count**: 高響應區域的連通組件數量（應該接近 GT stomata 數量）
- **valley_depth**: 連接線中點的對比度 vs 端點對比度（應該顯著更高）

### 3. 預期效果

✅ **成功指標**：
- merge_rate 降低 20-50%
- 訓練損失曲線穩定（separation loss 逐漸減小）
- 檢測 mAP 不下降或略有提升

⚠️ **可能問題**：
- 如果 separation_loss_weight 太大，可能過度正則化導致 mAP 下降
- 如果 valley_margin 太大，可能導致特徵坍塌

## 除錯建議

### 問題 1：separation loss 為 0

**原因**：
- GT 中心太少（< 2 per image）
- 隨機初始化下，特徵已經分離
- margin 設定太小

**解決**：
- 檢查 GT label 是否正確載入
- 增大 `--separation-valley-margin`

### 問題 2：訓練損失震盪

**原因**：
- separation_loss_weight 太大
- learning rate 太高

**解決**：
- 降低 `--separation-loss-weight` 到 0.1-0.5
- 啟用 `--warmup-epochs 5`

### 問題 3：檢測 mAP 下降

**原因**：
- 過度正則化抑制了有用的特徵
- separation loss 與 detection loss 目標衝突

**解決**：
- 降低 separation_loss_weight
- 調整 `--separation-target-layer` 到更早的 layer（如 layer 7 或 8）
- 僅在 stage 2/3 啟用 separation loss（需要程式碼修改）

## 與 FFT Block 的比較

| 特性 | FFT Block | Separation Loss |
|------|-----------|-----------------|
| **機制** | 頻域週期性先驗 | 顯式空間分離約束 |
| **訓練方式** | 學習 gate 權重 | 直接懲罰違規 |
| **運算成本** | 中等（FFT + MLP） | 低（只計算損失） |
| **推理成本** | 有（需要 FFT forward） | 無（純訓練時損失） |
| **效果** | 幾乎無效（gate→0） | **待驗證** |

**結論**：Separation loss 是更直接的方法，不需要額外的推理開銷，但需要實驗驗證是否比 FFT 更有效。

## 後續改進方向

1. **多尺度分離**：在多個 DINO layers (7, 9, 11) 同時施加 separation loss
2. **自適應 margin**：根據 stomata 密度動態調整 valley_margin
3. **對比損失**：替代 valley loss，使用 contrastive 或 triplet loss
4. **連通性損失**：直接在連通組件層級計算損失（需要 scipy/differentiable approximation）

## 測試腳本

```bash
# 1. 單元測試
python test_separation_loss.py

# 2. 快速驗證（1 epoch smoke test）
python mtkd_framework/run_v2.py \
  --separation-loss-weight 0.5 \
  --epochs 1 --batch-size 2

# 3. 完整訓練比較
# A. 無 separation loss (baseline)
python mtkd_framework/run_v2.py \
  --output-dir outputs/baseline \
  --epochs 50

# B. 有 separation loss
python mtkd_framework/run_v2.py \
  --separation-loss-weight 0.5 \
  --output-dir outputs/with_separation \
  --epochs 50

# 4. 比較結果
python compare_separation_results.py \
  --baseline outputs/baseline \
  --separation outputs/with_separation
```

## 參考資料

- **Conversation summary**: "我感覺有沒有加差不多ㄟ，加入吼應該會變得更散佚點，部會有兩個連在一起得情況"
- **Prior analysis**: FFT gate sweep showing no separation effect (merge_rate 0.0326 → 0.0327)
- **Connectivity metrics**: `visualize_layer10_fft_compare.py` with scipy connected-component analysis

---

**作者**: AI Assistant  
**日期**: 2026-03-10  
**版本**: 1.0
