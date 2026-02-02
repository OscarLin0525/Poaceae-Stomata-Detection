"""
群體一致性驗證實驗
===================

驗證 L_intra 的核心假設：stomata 群體一致性 > noise 群體一致性

使用方式：
1. 準備一張帶有 stomata 的測試圖片
2. 手動標記 stomata 和 noise 的像素座標
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
from PIL import Image
from torchvision import transforms


def load_dinov3_model(model_name: str = "vitb16", device: str = "cuda"):
    """
    載入 DINOv3 預訓練模型

    Args:
        model_name: 模型名稱，可選 "vits16", "vitb16", "vitl16"
        device: "cuda" 或 "cpu"

    Returns:
        model: DINOv3 模型
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

    Args:
        image_path: 圖片路徑
        img_size: 目標尺寸 (會 resize 成 img_size x img_size)

    Returns:
        image_tensor: (1, 3, img_size, img_size)
        original_size: (orig_h, orig_w) 原始尺寸，用於座標轉換
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


def convert_pixel_to_patch_coords(
    pixel_positions: list,
    original_size: tuple,
    img_size: int = 224,
    patch_size: int = 16,
):
    """
    將原始圖片的像素座標轉換為 patch 座標

    Args:
        pixel_positions: [(x1, y1), (x2, y2), ...] 原始圖片的像素座標
        original_size: (orig_w, orig_h) 原始圖片尺寸
        img_size: resize 後的尺寸
        patch_size: DINO patch 大小

    Returns:
        patch_positions: [(px1, py1), (px2, py2), ...] patch 座標
    """
    orig_w, orig_h = original_size
    num_patches = img_size // patch_size  # 14 for 224/16

    patch_positions = []
    for (x, y) in pixel_positions:
        # 轉換到 resize 後的座標
        x_resized = x * img_size / orig_w
        y_resized = y * img_size / orig_h

        # 轉換到 patch 座標
        px = int(x_resized // patch_size)
        py = int(y_resized // patch_size)

        # Clamp to valid range
        px = max(0, min(px, num_patches - 1))
        py = max(0, min(py, num_patches - 1))

        patch_positions.append((px, py))

    return patch_positions


def extract_features(model, image_tensor, device: str = "cuda"):
    """
    提取 DINO 特徵

    Args:
        model: DINOv3 模型
        image_tensor: (1, 3, H, W)
        device: 設備

    Returns:
        features: (1, H', W', C) 空間特徵
        cls_token: (1, C) CLS token
    """
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model.forward_features(image_tensor)

        # DINOv3 輸出格式
        patch_tokens = output["x_norm_patchtokens"]  # (B, N, C)
        cls_token = output["x_norm_clstoken"]        # (B, C)

        # 計算空間維度
        B, N, C = patch_tokens.shape
        H = W = int(N ** 0.5)  # 假設是正方形

        # Reshape 為空間格式
        features = patch_tokens.view(B, H, W, C)  # (1, 14, 14, 768)

    return features, cls_token


def compute_pairwise_similarity(features: torch.Tensor) -> float:
    """
    計算特徵之間的 pairwise cosine similarity (排除對角線)

    Args:
        features: (N, C)

    Returns:
        mean_similarity: 平均相似度
    """
    N = features.shape[0]
    if N < 2:
        return 0.0

    # Normalize
    features_norm = F.normalize(features, dim=-1)

    # Pairwise similarity
    sim_matrix = features_norm @ features_norm.T  # (N, N)

    # 排除對角線
    mask = ~torch.eye(N, dtype=torch.bool, device=features.device)
    mean_sim = sim_matrix[mask].mean().item()

    return mean_sim


def verify_group_consistency(
    model,
    image_path: str,
    stomata_pixel_positions: list,
    noise_pixel_positions: list,
    img_size: int = 224,
    patch_size: int = 16,
    device: str = "cuda",
):
    """
    驗證群體一致性假設

    Args:
        model: DINOv3 模型
        image_path: 測試圖片路徑
        stomata_pixel_positions: stomata 的像素座標 [(x1, y1), (x2, y2), ...]
        noise_pixel_positions: noise 的像素座標 [(x1, y1), (x2, y2), ...]
        img_size: 輸入尺寸
        patch_size: patch 大小
        device: 設備

    Returns:
        result: dict 包含驗證結果
    """
    print("=" * 60)
    print("群體一致性驗證實驗")
    print("=" * 60)

    # 1. 載入圖片
    print(f"\n[1/4] 載入圖片: {image_path}")
    image_tensor, original_size = load_and_preprocess_image(image_path, img_size)
    print(f"  原始尺寸: {original_size}")
    print(f"  Resize 到: {img_size}x{img_size}")

    # 2. 轉換座標
    print(f"\n[2/4] 轉換座標")
    stomata_patches = convert_pixel_to_patch_coords(
        stomata_pixel_positions, original_size, img_size, patch_size
    )
    noise_patches = convert_pixel_to_patch_coords(
        noise_pixel_positions, original_size, img_size, patch_size
    )
    print(f"  Stomata: {len(stomata_pixel_positions)} 個位置")
    print(f"  Noise: {len(noise_pixel_positions)} 個位置")

    # 去重（同一個 patch 只取一次）
    stomata_patches = list(set(stomata_patches))
    noise_patches = list(set(noise_patches))
    print(f"  去重後 - Stomata patches: {len(stomata_patches)}, Noise patches: {len(noise_patches)}")

    # 3. 提取特徵
    print(f"\n[3/4] 提取 DINO 特徵")
    features, cls_token = extract_features(model, image_tensor, device)
    print(f"  特徵形狀: {features.shape}")  # (1, 14, 14, 768)

    # 4. 計算群體一致性
    print(f"\n[4/4] 計算群體一致性")

    # 提取對應位置的特徵
    stomata_features = torch.stack([
        features[0, py, px, :] for (px, py) in stomata_patches
    ])  # (N_stomata, C)

    noise_features = torch.stack([
        features[0, py, px, :] for (px, py) in noise_patches
    ])  # (N_noise, C)

    # 計算一致性
    stomata_consistency = compute_pairwise_similarity(stomata_features)
    noise_consistency = compute_pairwise_similarity(noise_features)

    # 計算比值
    ratio = stomata_consistency / (noise_consistency + 1e-8)

    # 判斷假設是否成立
    is_valid = stomata_consistency > noise_consistency

    # 輸出結果
    print("\n" + "=" * 60)
    print("驗證結果")
    print("=" * 60)
    print(f"  Stomata 群體一致性: {stomata_consistency:.4f}")
    print(f"  Noise 群體一致性:   {noise_consistency:.4f}")
    print(f"  比值 (stomata/noise): {ratio:.2f}x")
    print("-" * 60)

    if is_valid:
        print("  ✅ 假設成立：stomata 群體一致性 > noise 群體一致性")
        print("  → L_intra 預期有效，可以進行 FilterBank 訓練")
    else:
        print("  ❌ 假設不成立：stomata 群體一致性 <= noise 群體一致性")
        print("  → L_intra 可能無效，建議：")
        print("    1. 檢查標記的位置是否正確")
        print("    2. 增加 noise 類別的多樣性（反光、葉脈、細胞壁等）")
        print("    3. 考慮使用替代方案（見 FilterBankREADME.md 附錄 E.8）")

    print("=" * 60)

    result = {
        "stomata_consistency": stomata_consistency,
        "noise_consistency": noise_consistency,
        "ratio": ratio,
        "is_valid": is_valid,
        "stomata_patches": stomata_patches,
        "noise_patches": noise_patches,
    }

    return result


def visualize_positions(
    image_path: str,
    stomata_pixel_positions: list,
    noise_pixel_positions: list,
    save_path: str = None,
):
    """
    視覺化標記的位置（可選）

    需要 matplotlib
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not installed, skipping visualization")
        return

    image = Image.open(image_path)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)

    # 標記 stomata
    for (x, y) in stomata_pixel_positions:
        circle = plt.Circle((x, y), 10, color='green', fill=False, linewidth=2)
        ax.add_patch(circle)

    # 標記 noise
    for (x, y) in noise_pixel_positions:
        circle = plt.Circle((x, y), 10, color='red', fill=False, linewidth=2)
        ax.add_patch(circle)

    # 圖例
    stomata_patch = mpatches.Patch(color='green', label=f'Stomata ({len(stomata_pixel_positions)})')
    noise_patch = mpatches.Patch(color='red', label=f'Noise ({len(noise_pixel_positions)})')
    ax.legend(handles=[stomata_patch, noise_patch], loc='upper right')

    ax.set_title("Marked Positions")
    ax.axis('off')

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

    # ============================================================
    # 2. 標記位置（像素座標）
    # ============================================================
    # 請根據你的圖片手動標記 5-10 個 stomata 和 5-10 個 noise 位置
    # 座標格式: (x, y)，其中 x 是水平方向，y 是垂直方向
    # 左上角是 (0, 0)

    # Stomata 位置（綠色圓圈標記的位置）
    # 建議選擇清晰可見的 stomata
    STOMATA_POSITIONS = [
        # 範例座標，請替換為實際座標
        (100, 50),   # 第 1 個 stomata
        (180, 52),   # 第 2 個 stomata
        (260, 48),   # 第 3 個 stomata
        (120, 150),  # 第 4 個 stomata
        (200, 148),  # 第 5 個 stomata
        (280, 152),  # 第 6 個 stomata
        # 可以添加更多...
    ]

    # Noise 位置（紅色圓圈標記的位置）
    # 建議包含多種類型的 noise：反光、葉脈、細胞壁、模糊區域等
    NOISE_POSITIONS = [
        # 範例座標，請替換為實際座標
        (50, 80),    # 反光
        (150, 100),  # 葉脈
        (250, 90),   # 細胞壁
        (80, 200),   # 模糊區域
        (300, 180),  # 另一個反光
        # 可以添加更多...
    ]

    # ============================================================
    # 3. 執行驗證
    # ============================================================

    print("\n" + "=" * 60)
    print("FilterBank 群體一致性驗證實驗")
    print("=" * 60 + "\n")

    # 檢查圖片是否存在
    if not os.path.exists(IMAGE_PATH):
        print(f"錯誤: 圖片不存在: {IMAGE_PATH}")
        print("請修改 IMAGE_PATH 變數為正確的圖片路徑")
        sys.exit(1)

    # 檢查標記數量
    if len(STOMATA_POSITIONS) < 3:
        print("警告: Stomata 標記數量太少（建議至少 5 個）")
    if len(NOISE_POSITIONS) < 3:
        print("警告: Noise 標記數量太少（建議至少 5 個）")

    # 載入模型
    model = load_dinov3_model(MODEL_NAME, DEVICE)

    # 執行驗證
    result = verify_group_consistency(
        model=model,
        image_path=IMAGE_PATH,
        stomata_pixel_positions=STOMATA_POSITIONS,
        noise_pixel_positions=NOISE_POSITIONS,
        img_size=224,
        patch_size=16,
        device=DEVICE,
    )

    # 可選：視覺化標記位置
    # visualize_positions(
    #     IMAGE_PATH,
    #     STOMATA_POSITIONS,
    #     NOISE_POSITIONS,
    #     save_path="marked_positions.png"
    # )

    # ============================================================
    # 4. 結果解讀
    # ============================================================
    print("\n" + "=" * 60)
    print("結果解讀指南")
    print("=" * 60)
    print("""
    比值 (ratio) 解讀：

    ratio > 1.5  : 很好！假設強烈成立，FilterBank 預期效果好
    ratio > 1.2  : 不錯。假設成立，FilterBank 應該有效
    ratio > 1.0  : 勉強。假設成立但邊緣，效果可能有限
    ratio <= 1.0 : 不成立。需要考慮替代方案

    如果假設不成立，建議：
    1. 增加 noise 的多樣性（確保包含多種類型）
    2. 使用更多 stomata 樣本
    3. 嘗試不同的 DINO 模型 (vits16, vitl16)
    4. 參考 FilterBankREADME.md 附錄 E.8 的替代方案
    """)
