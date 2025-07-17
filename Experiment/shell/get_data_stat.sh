#!/bin/bash

INFILE="/home/junho/KeyRAG/data/K-RAG/train_dataset/test.json"
OUTFILE="result/data-stat/test.json"

python code/get_data_stat.py \
    --input_file "${INFILE}" \
    --output_file "${OUTFILE}"