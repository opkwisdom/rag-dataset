#!/bin/bash

# Shell script to run dense retrieval
BASE_DIR=/home/junho/KeyRAG/DPR
# Set variables
INDEX_PATH="/home/junho/KeyRAG/bm25/index"
QA_DATASET="kisti_test"
CTX_DATASETS="dpr_kisti"
# 만약 여러 개의 인코딩된 파일을 사용할 경우
OUT_FILE="$BASE_DIR/outputs/retrieval_results_true.json"

# Execute dense retrieval
python ../bm25_search.py \
    index_path=$INDEX_PATH \
    qa_dataset=$QA_DATASET \
    ctx_datatsets=[$CTX_DATASETS] \
    out_file=$OUT_FILE \