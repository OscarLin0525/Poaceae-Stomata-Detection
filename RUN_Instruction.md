1. Whole MTKD Framework
python -m mtkd_framework.run_v2 \
  --config mtkd_rice_dino_pattern_distill.yaml

2. Pseudo Label Refinement: Boundary-Distance Class Conversion

This teacher only changes the semantic class of existing pseudo boxes. It does
not add boxes and does not modify box geometry. A pseudo box originally assigned
to class 0 is converted to class 1 when its normalized distance to the closest
image border is less than or equal to the selected threshold.

Wheat-GT calibrated command (`0.16%`, approximately `3.1 px` on a `1944 px`
short image side):

python refine_pseudo_labels_by_boundary.py \
  --input-root outputs/pseudo_label_wheatbarley \
  --output-root outputs/pseudo_label_wheatbarley_border_refined_wheat_hard016 \
  --splits train val \
  --source-class 0 \
  --target-class 1 \
  --hard-threshold-percent 0.16 \
  --copy-images \
  --overwrite

Outputs:
- Refined labels: `outputs/pseudo_label_wheatbarley_border_refined_wheat_hard016/labels/<split>`
- Copied images: `outputs/pseudo_label_wheatbarley_border_refined_wheat_hard016/images/<split>`
- Figures: `boundary_refinement_report/boundary_distance_histogram.pdf` and `boundary_refinement_report/class_counts_before_after.pdf`
- Statistics: `boundary_refinement_report/boundary_refinement_report.json` and `boundary_refinement_report/per_instance_boundary_distances.csv`

3. DINO + Frequency Bank
python train_dino_bypass_offline.py \
  --config mtkd_rice_dino_pattern_distill.yaml \
  --num-samples 0 \
  --output-dir outputs/dino_bypass_rice_pseudo_train_visual_rerun \
  --export-pseudo-dir outputs/pseudo_label_rice_dino_bypass_rerun \
  --export-split train

Outputs:
- Completed pseudo labels: `outputs/pseudo_label_rice_dino_bypass_rerun/labels/train`
- Feature and prediction panels: `outputs/dino_bypass_rice_pseudo_train_visual_rerun`
- Completion counts: `bbox_counts_before_after.pdf` and `bbox_counts_before_after.png`
- `Added BBoxes` counts final after-pattern predictions that do not match an original before-pattern prediction.

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
