#!/bin/bash

# 입력 폴더와 출력 파일 설정
PASSAGE_PATH="../../data/passages.jsonl"
INFILE_PATH="../data/query_sorted.jsonl"
INFILE="../collection/query"
OUTFILE="../collection/ir_data"
METHOD="SM"
INDEX_PATH="/home/nlplab/etri/QA/bm25/index"
NUM_THREADS=32
K=5000
CURRENT_DATE=$(date +"%m-%d")
OUTPUT_DIR=../"$CURRENT_DATE"
OUTPUT_FILE="$OUTPUT_DIR/ir_data_$METHOD.jsonl"

# 디렉토리 생성
mkdir -p ../collection
mkdir -p "$OUTPUT_DIR"

python ../code/split_data.py \
    --infile_path $INFILE_PATH \
    --outfile_path $INFILE \
    --num_files $NUM_THREADS

b=1
NUM=$(($NUM_THREADS - $b))

# NUM THREADS만큼 병렬 처리
for i in $(seq 0 $NUM)
do
    python3 ../code/query_labeling_by_query.py \
        --passage_path=$PASSAGE_PATH \
        --infile=$INFILE \
        --outfile=$OUTFILE \
        --method=$METHOD \
        --index_path=$INDEX_PATH \
        --file_num $i \
        --k $K &
done

wait
echo "Processing queries completed"

for jsonl_file in "$OUTFILE"*.jsonl
do
    if [ -f "$jsonl_file" ]; then
        cat "$jsonl_file" >> "$OUTPUT_FILE"
    fi
done

rm -rf ../collection

echo "Complete $METHOD. Output saved to $OUTPUT_FILE"