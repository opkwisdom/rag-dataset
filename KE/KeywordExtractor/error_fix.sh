#!/bin/bash

INFILE_PATH="data/qas_hard_rag"
TARGET_IDS_PATH="data/zero_match_ids.txt"
OUTFILE_PATH="collection/error_fix"
NUM_THREADS=1

b=1

CURRENT_DATE=$(date +"%m-%d")
mkdir -p "collection"
mkdir -p "$CURRENT_DATE"

python efficient/candidate_generator_error_fix.py \
    --infile_path $INFILE_PATH \
    --outfile_path $OUTFILE_PATH \
    --target_ids_path $TARGET_IDS_PATH