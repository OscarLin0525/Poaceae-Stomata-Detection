1. Whole MTKD Framework
python -m mtkd_framework.run_v2 \
  --config mtkd_rice_dino_pattern_distill.yaml

2. Pseudo Label Refined 
python train_dino_bypass_offline.py \
  --config mtkd_rice_dino_pattern_distill.yaml \
  --num-samples 0 \
  --export-pseudo-dir outputs/pseudo_label_rice_dino_bypass_clean \
  --export-split train \
  --pseudo-edge-as-incomplete \
  --pseudo-complete-class-id 0 \
  --pseudo-incomplete-class-id 1 \
  --pseudo-hard-edge-margin-px 6 \
  --pseudo-soft-edge-policy keep

3. DINO + Frequency Bank
python train_dino_bypass_offline.py \
  --config mtkd_rice_dino_pattern_distill.yaml \
  --num-samples 0 \
  --output-dir outputs/dino_bypass_rice_pseudo_train_visual_rerun \
  --export-pseudo-dir outputs/pseudo_label_rice_dino_bypass_rerun \
  --export-split train

4. Wheat Model Init Pseudo Label
  python predict_oscar.py

In first stage:
1. Generate the init pseudo label first with wheat and barley.
2. Refined the pseudo label with pseudo label refinement.
3. Train student with refined pseudo label first.
4. Run mtkd framework with gt wheat and pseudo label barley and wheat. validation with wheat and barley dataset.
In second stage:
5. Generate the init rice pseudo label with pre-trained student.
6. Run the dino + frequency complement module to complete the horizontal pattern and then generate the pseudo label
7. Run mtkd framework with gt wheat and pseudo label barley and wheat and rice pseudo label validation with wheat and barley rice dataset.
8. Test with student weight file.