#!/bin/bash
timestamp=$(date +%Y%m%d-%H%M%S)
CUDA_VISIBLE_DEVICES=2,3 DETECTRON2_DATASETS="/mnt/data1/ayushman/datasets" python train_net.py \
        --config-file configs/DensePose-EFFV2-BiFPN-Keypoint-Seg.yaml \
        --num-gpus 2 \
        --num-machines 1 \
        OUTPUT_DIR ./output/${timestamp} \
        CUDNN_BENCHMARK True
