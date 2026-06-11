# Poaceae-Stomata-Detection
0. Prepare
    Create virtual environment
    $ python -m venv VENV_NAME
    $ source VENV_NAME/bin/activate
    $ pip install -r requirements.txt

    Prepare pseudo label
    cd Myultralytics //wheat model
    python predict.py //generate wheat and barley pseudo label

1. First Stage Training
    cd Poaceae-Stomata-Detection
    python -m mtkd_framework.run_v2 --config mtkd_first_stage.yaml
    // source data: wheat ground truth
    // target data: wheat and barley pseudo label

2. Second Stage Training
    python refine_pseudo_labels_by_boundary.py // with wheat and barley pseudo label

    python train_dino_bypass_offline.py --config mtkd_second_stage.yaml // for rice pseudo label
    '''
    dino_bypass:
    input_dir: .../Stomata_Dataset_two_class/RICE/images/train
    yolo_weights: .../student_best.pt
    output_dir: .../outputs/dino_bypass_rice_template_visual
    '''

    python -m mtkd_framework.run_v2 --config mtkd_second_stage.yaml
    '''
    pseudo_labels:
    label_dir: "/home/oscar/Poaceae-Stomata-Detection/outputs/pseudo_label_rice_dino_bypass_clean/labels/train"
    '''
    // source data: wheat ground truth and pseudo label and barley pseudo label
    // target data: rice pseudo label
    