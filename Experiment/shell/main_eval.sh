#!/bin/bash

TASK="query_only"
HF_MODEL="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
MODEL_NAME="EXAONE-3.5-7.8B-Instruct"

TASK_LIST=("query_only" "keyphrase_only" "context_only" "context_keyphrase")

for TASK in "${TASK_LIST[@]}"; do
    echo "Running task: ${TASK}"
    CUDA_VISIBLE_DEVICES=0 python code/main_eval.py \
        --model_name_or_path "${HF_MODEL}" \
        --input_dir "result/rag-eval/meta" \
        --output_dir "result/llm-eval/${MODEL_NAME}" \
        --task "${TASK}" \
        --infile_name "${TASK}" \
        --outfile_name "EXAONE_${TASK}" \
        --batch_size 6
done

TASK="keysentece"
for VERSION in 3 4 5; do
    echo "Running task: ${TASK} for version ${VERSION}"
    CUDA_VISIBLE_DEVICES=0 python code/main_eval.py \
        --model_name_or_path "${HF_MODEL}" \
        --input_dir "result/rag-eval/meta" \
        --output_dir "result/llm-eval/${MODEL_NAME}" \
        --task "${TASK}" \
        --infile_name "${TASK}_${VERSION}" \
        --outfile_name "EXAONE_${TASK}_v${VERSION}" \
        --batch_size 6
done