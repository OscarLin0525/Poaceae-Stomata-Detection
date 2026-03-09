"""
Model Verification Script

This script verifies that the student and teacher models can be
instantiated correctly and that their output tensor shapes are compatible.

Steps:
1. Create a dummy input image tensor.
2. Load a real DINOv3 model as the teacher.
3. Instantiate the StudentDetector model.
4. Perform a forward pass on both models.
5. Print and verify the shapes of the outputs, especially the features
   used for knowledge distillation alignment.
"""

import torch
import sys
from pathlib import Path

# -- 設定 Project Root，確保可以正確 import --
# 這個 verify_models.py 在 mtkd_framework/models/ 裡面
# Project Root 是 Poaceae-Stomata-Detection/
# 所以我們需要往上走三層
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))
# -- DINOv3 的 import 路徑 --
dinov3_root = project_root / "dinov3-main"
sys.path.append(str(dinov3_root))


from mtkd_framework.models.student_model import StudentDetector

def get_teacher_model(model_name="dinov3_vits14_reg"):
    """
    Loads a DINOv3 model from torch.hub.
    This will serve as our teacher.
    """
    print(f"Loading teacher model: {model_name}...")
    try:
        # 使用 facebookresearch/dinov3 repo
        teacher_model = torch.hub.load('facebookresearch/dinov3', model_name)
        teacher_model.eval()
        print("Teacher model loaded successfully.")
        return teacher_model
    except Exception as e:
        print(f"Error loading teacher model: {e}")
        print("Please ensure you have an internet connection and the DINOv3 repository is accessible.")
        return None

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # =========================================
    # 1. 建立虛擬輸入
    # =========================================
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 224, 224).to(device)
    print(f"\n--- Input ---")
    print(f"Dummy image shape: {dummy_images.shape}")

    # =========================================
    # 2. 載入 Teacher 模型並取得輸出
    # =========================================
    # DINOv3 ViT-S/14 (Register version)
    # Register-based models (e.g., dinov3_vits14_reg) provide improved performance
    teacher_model = get_teacher_model("dinov3_vits14_reg")
    if teacher_model is None:
        return

    teacher_model.to(device)
    # DINO feature dimension for ViT-S is 384
    teacher_dim = 384

    with torch.no_grad():
        # DINOv3 forward returns a dict of features
        # 'x_norm_clstoken' is the final class token feature
        # 'x_norm_patchtokens' is the final patch tokens
        teacher_features = teacher_model.forward_features(dummy_images)
        teacher_cls_token = teacher_features['x_norm_clstoken']
        teacher_patch_tokens = teacher_features['x_norm_patchtokens']
    
    print(f"\n--- Teacher (DINOv3) ---")
    print(f"Teacher CLS token shape: {teacher_cls_token.shape}")
    print(f"Teacher patch tokens shape: {teacher_patch_tokens.shape}")
    
    # 模擬 Teacher 的多尺度輸出 (用於 student 的 multi-scale adapter)
    # 在真實訓練中，這可能來自 DINO 的不同層，但這裡我們先用 patch tokens 模擬
    # 學生端的 FPN 輸出是 4 個 scale
    teacher_multi_scale_sim = [
        teacher_patch_tokens.permute(0, 2, 1).reshape(batch_size, teacher_dim, 16, 16)
    ] * 4


    # =========================================
    # 3. 實例化 Student 模型
    # =========================================
    print(f"\n--- Student ---")
    student_model = StudentDetector(
        backbone_config={"backbone_type": "resnet50", "pretrained": False}, # 在 CI/無網路環境下設為 False
        dino_teacher_dim=teacher_dim,
    ).to(device)

    print("StudentDetector instantiated.")

    # =========================================
    # 4. 執行 Student Forward Pass
    # =========================================
    # return_adapted_features=True 來獲取與 teacher 對齊的特徵
    student_outputs = student_model(
        dummy_images, 
        return_features=True, 
        return_adapted_features=True
    )
    print("Student forward pass completed.")

    # =========================================
    # 5. 印出並驗證 Shape
    # =========================================
    print("\n--- Outputs & Verification ---")

    # 檢測頭輸出
    print(f"Student detection 'logits' shape: {student_outputs['logits'].shape}")
    print(f"Student detection 'boxes' shape: {student_outputs['boxes'].shape}")
    
    # 全局特徵對齊 (CLS Token)
    student_adapted_global = student_outputs['adapted_features']
    print(f"Student adapted global feature shape: {student_adapted_global.shape}")
    
    # 多尺度特徵對齊
    student_adapted_multiscale = student_outputs['adapted_multi_scale_features']
    print(f"Student adapted multi-scale feature count: {len(student_adapted_multiscale)}")
    for i, feat in enumerate(student_adapted_multiscale):
        print(f"  - Scale {i} shape: {feat.shape}")


    # ** 核心驗證 **
    print("\n--- Core Verification ---")
    # 驗證全局特徵維度
    if student_adapted_global.shape == teacher_cls_token.shape:
        print(f"✅ SUCCESS: Student adapted global features ({student_adapted_global.shape}) match Teacher CLS token ({teacher_cls_token.shape}).")
    else:
        print(f"❌ FAILURE: Shape mismatch in global features!")
        print(f"   - Student: {student_adapted_global.shape}")
        print(f"   - Teacher: {teacher_cls_token.shape}")
        
    # 驗證多尺度特徵維度
    # student_model.py 的 multi_scale_adapter 會將 FPN 的 [256, 256, 256, 256] 通道適配到 [384, 384, 384, 384]
    # 我們檢查通道數是否正確
    correct_ms_channels = all(feat.shape[1] == teacher_dim for feat in student_adapted_multiscale)
    if len(student_adapted_multiscale) == 4 and correct_ms_channels:
         print(f"✅ SUCCESS: Student adapted multi-scale features have the correct channel count ({teacher_dim}).")
    else:
        print(f"❌ FAILURE: Student adapted multi-scale features have incorrect shapes or count.")


if __name__ == "__main__":
    main()
