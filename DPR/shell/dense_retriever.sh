#!/bin/bash

# Shell script to run dense retrieval
BASE_DIR=/home/junho/KeyRAG/DPR
# Set variables
MODEL_FILE="$BASE_DIR/trained_encoder_model/dpr_biencoder.1"
QA_DATASET="kisti_test"
CTX_DATASETS="dpr_kisti"
# 만약 여러 개의 인코딩된 파일을 사용할 경우
ENCODED_CTX_FILES=$(for i in $(seq 0 9); do echo -n "\"/home/junho/KeyRAG/DPR/trained_encoder_model/embeddings_$i\","; done | sed 's/,$//')
OUT_FILE="$BASE_DIR/outputs/dpr_retrieval_results.json"

# Execute dense retrieval
python ../dense_retriever.py \
    model_file=$MODEL_FILE \
    qa_dataset=$QA_DATASET \
    ctx_datatsets=[$CTX_DATASETS] \
    encoded_ctx_files=[$ENCODED_CTX_FILES] \
    out_file=$OUT_FILE \


