#!/bin/bash

INFILE="/home/junho/KeyRAG/data/KISTI/idf_cand/qas_idf_candidate.json"
OUTFILE="result/data-stat/data_stat_2.json"

python code/get_data_stat.py \
    --input_file "${INFILE}" \
    --output_file "${OUTFILE}"