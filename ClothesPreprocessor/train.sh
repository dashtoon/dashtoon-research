#!/bin/bash
timestamp=$(date +%Y%m%d-%H%M%S)
DETECTRON2_DATASETS="/mnt/disk1/repos/dashtoon-research/ClothesPreprocessor/datasets" python train_net.py \
        --config-file configs/DensePose-EFFV2-BiFPN-Keypoint-Seg.yaml \
        OUTPUT_DIR ./output/${timestamp} \
        CUDNN_BENCHMARK True