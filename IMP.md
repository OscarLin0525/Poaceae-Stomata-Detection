# Stomata-Aware Adaptive Bypass for DINO

## 1. 模組定位與用途

這個模組是一個輕量級的即插即用 Adapter，以 residual bypass 的形式掛在預訓練 DINO 的中間層。推薦位置是 Layer 6 到 Layer 8。

它的目標不是改寫 DINO 的語義能力，而是基於 DINO 的理解，自適應地**加強已檢測到的 stomata** 並**補齊缺失的 stomata**。

主要用途：

1. **Stomata 定位**：從 DINO feature 學習每個位置是否有 stomata（語意感知）
2. **選擇性加強**：只在檢測到 stomata 的位置強化特徵信號
3. **自適應填補**：基於上下文和檢測結果，預測並填補缺失的 stomata
4. **為下游 student 提供更好的 feature target**：讓 MTKD 對齊到更準確、完整的 stomata 結構

## 2. 核心設計：三階段處理

與固定的空間卷積不同，新設計是**動態、自適應的**：

### 階段 1：Stomata 定位器
從 DINO 中間特徵學習，偵測每個 token 中是否有 stomata
- 輸入：`[B, N, C]` 
- 輸出：`[B, N, 1]` 機率圖 (sigmoid activated)
- 實現：簡單的 FC 層或小 Transformer

### 階段 2：選擇性加強
針對**已偵測到的 stomata 位置**，學習加強因子來強化使能
- 只有定位器高 confidence 的地方才會被加強
- 避免盲目地在整條行上強化（不會造成假的連續條紋）
- 保留 stomata 的離散性質

### 階段 3：缺失補齊
對於**定位器沒有檢測到的位置**，但根據上下文應該有 stomata 的地方：
- 預測合理的 feature 內容（填補頭）
- 乘以「缺失遮罩」（1 - 定位機率）只在空白位置補齊
- 基於局部上下文，填補真的漏檢

## 3. 為什麼比固定卷積更好

**舊方法（固定行/列卷積）問題**：
- 一整條行都被加強 → 創造假 stomata
- 無法區分真 stomata vs 空白區域
- 盲目地強制規律性，不關心實際位置

**新方法（自適應加強+填補）優勢**：
- 基於 DINO 語義理解 stomata 在哪裡
- 只加強真實的 stomata，保留離散性  
- 智能補齊，而不是盲目強化
- 多品種 dataset 上泛化能力更好

## 4. 資料流與介面

輸入：`[B, N, C]`

- `B` 是 batch size
- `N` 是 token 數，包含 CLS token 與 patch tokens  
- `C` 是 embedding dimension，常見是 768 (ViT-L16) 或 384 (ViT-B16)

輸出：`[B, N, C]`

- shape 不變
- 內容是根據 stomata 定位動態調整的語義特徵
- 完全相容於原始 DINO 輸出

## 5. 實作結構

### Step 1：Stomata 定位器

```python
class StomataLocator(nn.Module):
    # [B, N, C] -> [B, N, 1]
    # 學習每個 token 是否包含 stomata
```

### Step 2：強度頭（加強層）  

```python
self.strength_head = nn.Sequential(
    nn.Linear(embed_dim, bottleneck_dim),
    nn.GELU(),
    nn.Linear(bottleneck_dim, embed_dim),  # zero-init
)
# 用途：每個位置學習不同的加強幅度
```

### Step 3：填補頭（補齊層）

```python
self.fill_head = nn.Sequential(
    nn.Linear(embed_dim, bottleneck_dim),
    nn.GELU(), 
    nn.Linear(bottleneck_dim, embed_dim),  # zero-init
)
# 用途：預測缺失位置應該有的 feature 內容
```

### Step 4：零初始化安全閘門

- 強度頭與填補頭都是 zero-init
- 用 learnable `alpha` residual gate 控制整體貢獻強度
- 一開始 bypass 的輸出 = 0，完全不改變 DINO 特徵

### Step 5：完整前向過程

```
x: DINO feature [B, N, C]
   ↓
stomata_prob = locator(x)  # [B, N, 1]
   ↓
enhanced = x + strength_head(x) * stomata_prob
   ↓  
missing_pred = fill_head(enhanced)  # [B, N, C]
missing_mask = 1 - stomata_prob      # [B, N, 1]
   ↓
filled = enhanced + missing_pred * missing_mask
   ↓
output = alpha * filled  # residual gate
```

## 6. 掛載與訓練

### 離線預訓練（Stage 1）

1. 凍結 DINO 主幹
2. 只訓練 bypass 參數（locator + strength + fill）
3. Loss：基於 stomata 位置的對齊 loss
4. 目標：讓 alpha 從 0 增長到有意義的值，三個子模組都開始學習

### 在線 MTKD 微調（Stage 2）  

1. 加載 Stage 1 的 bypass 檢查點
2. 在完整 MTKD 訓練中與 student/teacher 一起微調
3. 少量梯度更新，保留 bypass 的結構先驗

## 7. 預期效果

如果做得對，最可能看到：

1. **Alpha 增長**：從 0 慢慢增長到 0.1-1.0 之間，表示 bypass 在學習
2. **定位精度**：PCA heatmap 上的亮點應該更準確指向 stomata 位置
3. **補齊效果**：缺失的 stomata（特別是行內間隙）會被預測填補
4. **Feature 品質**：Student 對齊時應該看到更清晰的 stomata 結構信號
5. **檢測性能**：下游 YOLO 檢測 mAP/recall 應該有所提升

## 8. 驗收標準

### Sanity check

Epoch 0、還沒更新權重時，開 bypass 與關 bypass 的輸出應該幾乎一致。這表示 zero-init 有成功保護原始 DINO。

### 訓練後效果

隨著訓練進行：

1. Alignment loss 應該下降。
2. feature map 應該更集中在 stomata row。
3. 下游 detection 的 recall 應該提升，尤其是漏檢數量下降。

## 7. 建議結論

對你的 dataset 來說，多頻率版本比單一 kernel 更合理，也更有機會泛化到不同品種與不同 row spacing。

如果你要做實作，建議直接把 bypass 設計成三分支版本，優先用 `3 / 5 / 7` 這組 kernel，再看需不需要加 dilation。