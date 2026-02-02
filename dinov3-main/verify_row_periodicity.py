"""
行週期性驗證實驗
================

驗證 FilterBank 的核心假設：
- 有 stomata 的行 → FFT 有明顯主頻率（週期性強）
- 沒有 stomata / 只有 noise 的行 → FFT 無明顯主頻率（週期性弱）

使用方式：
1. 準備一張帶有 stomata 的測試圖片
2. 標記哪些行有 stomata、哪些行沒有
3. 執行此腳本

依賴：
- torch
- torchvision
- PIL
- matplotlib (可選，用於視覺化)
"""

import sys
import os

# 將 dinov3 加入 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms


def load_dinov3_model(model_name: str = "vitb16", device: str = "cuda"):
    """
    載入 DINOv3 預訓練模型
    """
    from dinov3.hub.backbones import dinov3_vits16, dinov3_vitb16, dinov3_vitl16

    model_dict = {
        "vits16": dinov3_vits16,
        "vitb16": dinov3_vitb16,
        "vitl16": dinov3_vitl16,
    }

    if model_name not in model_dict:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_dict.keys())}")

    print(f"Loading DINOv3 {model_name}...")
    model = model_dict[model_name](pretrained=True)
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

    return model


def load_and_preprocess_image(image_path: str, img_size: int = 224):
    """
    載入並預處理圖片
    """
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (W, H)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_tensor = transform(image).unsqueeze(0)  # (1, 3, H, W)

    return image_tensor, original_size


def extract_spatial_features(model, image_tensor, device: str = "cuda"):
    """
    提取 DINO 特徵並轉為空間格式

    Returns:
        features: (1, H', W', C) 空間特徵，例如 (1, 14, 14, 768)
    """
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model.forward_features(image_tensor)
        patch_tokens = output["x_norm_patchtokens"]  # (B, N, C)

        B, N, C = patch_tokens.shape
        H = W = int(N ** 0.5)

        features = patch_tokens.view(B, H, W, C)  # (1, 14, 14, 768)

    return features


def compute_row_periodicity(features: torch.Tensor, row_idx: int):
    """
    計算單行的週期性強度

    Args:
        features: (1, H', W', C) 空間特徵
        row_idx: 要分析的行索引 (0 到 H'-1)

    Returns:
        periodicity_score: float - 週期性強度 (0~1)
        dominant_freq: float - 主頻率 (正規化)
        spectrum: np.array - 頻譜
    """
    # 取出該行的特徵: (W', C)
    row_features = features[0, row_idx, :, :]  # (W', C)

    # 計算 row 內的 self-similarity
    # 這代表該行各位置特徵的相似度模式
    row_norm = F.normalize(row_features, dim=-1)  # (W', C)

    # 方法 1：用特徵的 L2 norm 作為 signal
    signal = row_features.norm(dim=-1).cpu().numpy()  # (W',)

    # 方法 2（替代）：用與行平均的相似度作為 signal
    # row_mean = row_features.mean(dim=0, keepdim=True)
    # signal = F.cosine_similarity(row_features, row_mean, dim=-1).cpu().numpy()

    # 去除 DC 成分（減去平均值）
    signal = signal - signal.mean()

    # FFT
    fft_result = np.fft.rfft(signal)
    spectrum = np.abs(fft_result)

    # 排除 DC (index 0)，找主頻率
    if len(spectrum) > 1:
        spectrum_no_dc = spectrum[1:]
        dominant_freq_idx = np.argmax(spectrum_no_dc) + 1
        dominant_freq = dominant_freq_idx / len(signal)  # 正規化頻率

        # 週期性強度 = 主頻率能量 / 總能量
        total_energy = np.sum(spectrum_no_dc ** 2)
        peak_energy = spectrum[dominant_freq_idx] ** 2

        if total_energy > 0:
            periodicity_score = peak_energy / total_energy
        else:
            periodicity_score = 0.0
    else:
        dominant_freq = 0.0
        periodicity_score = 0.0

    return periodicity_score, dominant_freq, spectrum


def compute_row_periodicity_v2(features: torch.Tensor, row_idx: int):
    """
    計算單行的週期性強度（基於自相關）

    自相關方法對於檢測週期性更穩健

    Args:
        features: (1, H', W', C) 空間特徵
        row_idx: 要分析的行索引

    Returns:
        periodicity_score: float - 週期性強度
        dominant_period: int - 主週期（以 patch 為單位）
        autocorr: np.array - 自相關函數
    """
    # 取出該行的特徵: (W', C)
    row_features = features[0, row_idx, :, :]  # (W', C)
    W = row_features.shape[0]

    # 計算 pairwise similarity matrix
    row_norm = F.normalize(row_features, dim=-1)
    sim_matrix = (row_norm @ row_norm.T).cpu().numpy()  # (W', W')

    # 從 similarity matrix 提取自相關
    # autocorr[k] = 平均 sim(i, i+k)
    autocorr = []
    for lag in range(W):
        similarities = []
        for i in range(W - lag):
            similarities.append(sim_matrix[i, i + lag])
        autocorr.append(np.mean(similarities))

    autocorr = np.array(autocorr)

    # 找第一個顯著的峰值（排除 lag=0）
    # 這代表主週期
    if len(autocorr) > 2:
        # 尋找 local maxima
        peaks = []
        for i in range(1, len(autocorr) - 1):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                peaks.append((i, autocorr[i]))

        if peaks:
            # 選最高的峰
            best_peak = max(peaks, key=lambda x: x[1])
            dominant_period = best_peak[0]
            peak_value = best_peak[1]

            # 週期性強度 = 峰值高度相對於 lag=0 的比例
            periodicity_score = peak_value / autocorr[0] if autocorr[0] > 0 else 0
        else:
            dominant_period = 0
            periodicity_score = 0.0
    else:
        dominant_period = 0
        periodicity_score = 0.0

    return periodicity_score, dominant_period, autocorr


def pixel_row_to_patch_row(pixel_y: int, original_h: int, img_size: int = 224, patch_size: int = 16):
    """
    將像素 y 座標轉換為 patch row 索引
    """
    # 轉換到 resize 後的座標
    y_resized = pixel_y * img_size / original_h
    # 轉換到 patch row
    patch_row = int(y_resized // patch_size)
    # Clamp
    num_patches = img_size // patch_size
    patch_row = max(0, min(patch_row, num_patches - 1))
    return patch_row


def verify_row_periodicity(
    model,
    image_path: str,
    stomata_row_pixel_ys: list,
    non_stomata_row_pixel_ys: list,
    img_size: int = 224,
    patch_size: int = 16,
    device: str = "cuda",
    method: str = "autocorr",  # "fft" or "autocorr"
):
    """
    驗證行週期性假設

    Args:
        model: DINOv3 模型
        image_path: 測試圖片路徑
        stomata_row_pixel_ys: 有 stomata 的行的 y 座標（像素）
        non_stomata_row_pixel_ys: 沒有 stomata 的行的 y 座標（像素）
        img_size: 輸入尺寸
        patch_size: patch 大小
        device: 設備
        method: "fft" 或 "autocorr"

    Returns:
        result: dict 包含驗證結果
    """
    print("=" * 60)
    print("行週期性驗證實驗")
    print("=" * 60)

    # 1. 載入圖片
    print(f"\n[1/4] 載入圖片: {image_path}")
    image_tensor, original_size = load_and_preprocess_image(image_path, img_size)
    orig_w, orig_h = original_size
    print(f"  原始尺寸: {original_size}")
    print(f"  Resize 到: {img_size}x{img_size}")

    num_patches = img_size // patch_size
    print(f"  Patch grid: {num_patches}x{num_patches}")

    # 2. 轉換行座標
    print(f"\n[2/4] 轉換行座標")

    stomata_patch_rows = list(set([
        pixel_row_to_patch_row(y, orig_h, img_size, patch_size)
        for y in stomata_row_pixel_ys
    ]))
    non_stomata_patch_rows = list(set([
        pixel_row_to_patch_row(y, orig_h, img_size, patch_size)
        for y in non_stomata_row_pixel_ys
    ]))

    print(f"  Stomata rows (pixel): {stomata_row_pixel_ys}")
    print(f"  Stomata rows (patch): {stomata_patch_rows}")
    print(f"  Non-stomata rows (pixel): {non_stomata_row_pixel_ys}")
    print(f"  Non-stomata rows (patch): {non_stomata_patch_rows}")

    # 3. 提取特徵
    print(f"\n[3/4] 提取 DINO 特徵")
    features = extract_spatial_features(model, image_tensor, device)
    print(f"  特徵形狀: {features.shape}")  # (1, 14, 14, 768)

    # 4. 計算週期性
    print(f"\n[4/4] 計算各行週期性 (方法: {method})")

    stomata_results = []
    non_stomata_results = []

    compute_func = compute_row_periodicity_v2 if method == "autocorr" else compute_row_periodicity

    print("\n  Stomata 行分析:")
    for row in stomata_patch_rows:
        score, freq_or_period, _ = compute_func(features, row)
        stomata_results.append({
            "row": row,
            "periodicity_score": score,
            "dominant": freq_or_period,
        })
        period_str = f"period={freq_or_period}" if method == "autocorr" else f"freq={freq_or_period:.3f}"
        print(f"    Row {row}: score={score:.4f}, {period_str}")

    print("\n  Non-stomata 行分析:")
    for row in non_stomata_patch_rows:
        score, freq_or_period, _ = compute_func(features, row)
        non_stomata_results.append({
            "row": row,
            "periodicity_score": score,
            "dominant": freq_or_period,
        })
        period_str = f"period={freq_or_period}" if method == "autocorr" else f"freq={freq_or_period:.3f}"
        print(f"    Row {row}: score={score:.4f}, {period_str}")

    # 統計
    stomata_avg_score = np.mean([r["periodicity_score"] for r in stomata_results]) if stomata_results else 0
    non_stomata_avg_score = np.mean([r["periodicity_score"] for r in non_stomata_results]) if non_stomata_results else 0

    ratio = stomata_avg_score / (non_stomata_avg_score + 1e-8)
    is_valid = stomata_avg_score > non_stomata_avg_score

    # 輸出結果
    print("\n" + "=" * 60)
    print("驗證結果")
    print("=" * 60)
    print(f"  Stomata 行平均週期性分數:     {stomata_avg_score:.4f}")
    print(f"  Non-stomata 行平均週期性分數: {non_stomata_avg_score:.4f}")
    print(f"  比值 (stomata/non-stomata):   {ratio:.2f}x")
    print("-" * 60)

    if is_valid:
        print("  ✅ 假設成立：stomata 行週期性 > non-stomata 行週期性")
        print("  → FilterBank 的週期性過濾機制預期有效")
    else:
        print("  ❌ 假設不成立：stomata 行週期性 ≤ non-stomata 行週期性")
        print("  → 建議：")
        print("    1. 檢查標記的行是否正確")
        print("    2. 確認圖片中 stomata 確實有週期性排列")
        print("    3. 嘗試不同的分析方法 (fft vs autocorr)")

    print("=" * 60)

    result = {
        "stomata_avg_score": stomata_avg_score,
        "non_stomata_avg_score": non_stomata_avg_score,
        "ratio": ratio,
        "is_valid": is_valid,
        "stomata_results": stomata_results,
        "non_stomata_results": non_stomata_results,
        "method": method,
    }

    return result


def visualize_row_analysis(
    image_path: str,
    features: torch.Tensor,
    stomata_patch_rows: list,
    non_stomata_patch_rows: list,
    method: str = "autocorr",
    save_path: str = None,
):
    """
    視覺化行週期性分析結果
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping visualization")
        return

    compute_func = compute_row_periodicity_v2 if method == "autocorr" else compute_row_periodicity

    # 收集所有行的數據
    all_rows = sorted(set(stomata_patch_rows + non_stomata_patch_rows))
    n_rows = len(all_rows)

    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, row in enumerate(all_rows):
        score, dominant, signal = compute_func(features, row)

        # 標記類型
        if row in stomata_patch_rows:
            row_type = "Stomata"
            color = "green"
        else:
            row_type = "Non-stomata"
            color = "red"

        # 左圖：信號/自相關
        axes[idx, 0].plot(signal, color=color, linewidth=2)
        axes[idx, 0].set_title(f"Row {row} ({row_type}) - {'Autocorr' if method == 'autocorr' else 'Signal'}")
        axes[idx, 0].set_xlabel("Lag" if method == "autocorr" else "Position")
        axes[idx, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        # 右圖：週期性分數
        axes[idx, 1].bar([0], [score], color=color, alpha=0.7)
        axes[idx, 1].set_ylim(0, 1)
        axes[idx, 1].set_title(f"Periodicity Score: {score:.4f}")
        axes[idx, 1].set_xticks([])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()

    plt.close()


# ============================================================
# 使用範例
# ============================================================

if __name__ == "__main__":
    # ============================================================
    # 1. 設定參數
    # ============================================================

    # 測試圖片路徑（請修改為你的圖片路徑）
    IMAGE_PATH = "/path/to/your/stomata_image.jpg"

    # 模型設定
    MODEL_NAME = "vitb16"  # 可選: "vits16", "vitb16", "vitl16"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 分析方法
    METHOD = "autocorr"  # "fft" 或 "autocorr"（推薦 autocorr）

    # ============================================================
    # 2. 標記行（像素 Y 座標）
    # ============================================================
    # 請根據你的圖片標記：
    # - 有 stomata 的行（選擇有明顯週期性 stomata 的 y 座標）
    # - 沒有 stomata 的行（選擇只有背景或 noise 的 y 座標）

    # 有 stomata 的行（像素 Y 座標）
    # 建議選擇 3-5 行，每行應該有明顯的週期性 stomata
    STOMATA_ROW_PIXEL_YS = [
        # 範例座標，請替換為實際座標
        50,   # 第 1 行有 stomata
        150,  # 第 2 行有 stomata
        250,  # 第 3 行有 stomata
    ]

    # 沒有 stomata 的行（像素 Y 座標）
    # 建議選擇 3-5 行，這些行只有背景、葉脈、或隨機 noise
    NON_STOMATA_ROW_PIXEL_YS = [
        # 範例座標，請替換為實際座標
        100,  # 第 1 行無 stomata（兩行 stomata 之間）
        200,  # 第 2 行無 stomata
        300,  # 第 3 行無 stomata
    ]

    # ============================================================
    # 3. 執行驗證
    # ============================================================

    print("\n" + "=" * 60)
    print("FilterBank 行週期性驗證實驗")
    print("=" * 60 + "\n")

    # 檢查圖片是否存在
    if not os.path.exists(IMAGE_PATH):
        print(f"錯誤: 圖片不存在: {IMAGE_PATH}")
        print("請修改 IMAGE_PATH 變數為正確的圖片路徑")
        sys.exit(1)

    # 檢查標記數量
    if len(STOMATA_ROW_PIXEL_YS) < 2:
        print("警告: Stomata 行標記數量太少（建議至少 3 行）")
    if len(NON_STOMATA_ROW_PIXEL_YS) < 2:
        print("警告: Non-stomata 行標記數量太少（建議至少 3 行）")

    # 載入模型
    model = load_dinov3_model(MODEL_NAME, DEVICE)

    # 執行驗證
    result = verify_row_periodicity(
        model=model,
        image_path=IMAGE_PATH,
        stomata_row_pixel_ys=STOMATA_ROW_PIXEL_YS,
        non_stomata_row_pixel_ys=NON_STOMATA_ROW_PIXEL_YS,
        img_size=224,
        patch_size=16,
        device=DEVICE,
        method=METHOD,
    )

    # ============================================================
    # 4. 結果解讀
    # ============================================================
    print("\n" + "=" * 60)
    print("結果解讀指南")
    print("=" * 60)
    print("""
    週期性分數解讀：

    分數接近 1.0: 該行有非常強的週期性
    分數 > 0.5:   該行有明顯週期性
    分數 0.2-0.5: 該行有弱週期性
    分數 < 0.2:   該行幾乎無週期性

    比值 (ratio) 解讀：

    ratio > 2.0  : 很好！假設強烈成立，FilterBank 預期效果好
    ratio > 1.5  : 不錯。假設成立，FilterBank 應該有效
    ratio > 1.0  : 勉強。假設成立但邊緣
    ratio <= 1.0 : 不成立。stomata 行沒有比較強的週期性

    如果假設不成立，可能原因：
    1. Stomata 在該圖片中本身就不是很週期性
    2. 解析度問題：14x14 patch grid 太粗，無法捕捉週期
    3. 標記的行選擇不當

    建議嘗試：
    1. 選擇 stomata 更密集、更規律的圖片
    2. 使用更大的輸入尺寸 (336, 448) 增加 patch 數量
    3. 同時執行 fft 和 autocorr 兩種方法比較
    """)
