#!/bin/bash
timestamp=$(date +%Y%m%d-%H%M%S)
python train_net.py \
        --config-file /mnt/data/repos/dashtoon-research/ClothesPreprocessor/configs/DensePose-EFFV2-BiFPN-Keypoint-Seg.yaml \
        OUTPUT_DIR ./output/${timestamp} \
        CUDNN_BENCHMARK True \
        SOLVER.CHECKPOINT_PERIOD 20