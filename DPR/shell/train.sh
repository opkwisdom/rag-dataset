#!/bin/bash
BASE_DIR=/home/junho/KeyRAG/DPR

TRAIN_DATASETS=kisti_train
DEV_DATASETS=kisti_dev
BIENCODER="biencoder_default"     # Train configuration - YAML 파일
OUTPUT_DIR="$BASE_DIR/trained_encoder_model"

CUDA_VISIBLE_DEVICES=0,1 python ../train_dense_encoder.py \
    train_datasets=[$TRAIN_DATASETS] \
    dev_datasets=[$DEV_DATASETS] \
    train=$BIENCODER \
    output_dir=$OUTPUT_DIR