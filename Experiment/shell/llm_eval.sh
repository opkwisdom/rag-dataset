#!/bin/bash

TASK="context_keyphrase"
GPT_MODEL="gpt-4o-mini"
SLICE=1000

TASK_LIST=("query_only" "keyphrase_only" "context_only" "context_keyphrase")

for TASK in "${TASK_LIST[@]}"; do
    echo Running task: ${TASK}
    CUDA_VISIBLE_DEVICES=0 python code/llm_eval.py \
        --model_name_or_path "${GPT_MODEL}" \
        --input_dir "result/rag-eval/meta" \
        --output_dir "result/llm-eval/${GPT_MODEL}" \
        --task "${TASK}" \
        --infile_name "${TASK}" \
        --outfile_name "${GPT_MODEL}_test_${TASK}" \
        --batch_size 6 \
        --max_new_tokens 512
done

TASK="keysentece"
for VERSION in 3 4 5; do
    echo "Running task: ${TASK} for version ${VERSION}"
    CUDA_VISIBLE_DEVICES=0 python code/llm_eval.py \
        --model_name_or_path "${GPT_MODEL}" \
        --input_dir "result/rag-eval/meta" \
        --output_dir "result/llm-eval/${GPT_MODEL}" \
        --task "${TASK}" \
        --infile_name "${TASK}_${VERSION}" \
        --outfile_name "EXAONE_${TASK}_v${VERSION}" \
        --batch_size 6 \
        --max_new_tokens 512
done