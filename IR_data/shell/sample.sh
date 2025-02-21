#!/bin/bash

INFILE_PATH="../data/ir_data_v1.jsonl"
OUTFILE_PATH="../data/ir_data_v1_sampled.jsonl"
N_SAMPLES=50000

python3 ../code/sampling.py \
    --infile_path=$INFILE_PATH \
    --outfile_path=$OUTFILE_PATH \
    --n_samples=$N_SAMPLES