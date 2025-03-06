#!/bin/bash

INFILE_PATH="data/qas_hard_rag.json"
INTER_PATH="collection/qas_hard_rag"
OUTFILE_PATH="collection/qas_rag_candidate"
NUM_THREADS=16

# 0부터 NUM_THREADS까지 수를 매개변수로 하여 파이썬 파일을 동시에 실행
b=1

CURRENT_DATE=$(date +"%m-%d")
mkdir -p "collection"
mkdir -p "$CURRENT_DATE"

# Step 1: Split data
echo "Splitting data into $NUM_THREADS files..."

NUM=$(($NUM_THREADS - $b))
# 입력 폴더와 출력 파일 설정


# Step 2: Run candidate generator using multiple threads
echo "Running candidate generator using $NUM_THREADS threads..."
for i in $(seq 0 $NUM)
do
    python efficient/candidate_generator.py \
        --infile_path $INTER_PATH \
        --outfile_path $OUTFILE_PATH \
        --file_postfix $i &  # & 기호를 사용해 백그라운드에서 병렬 실행
done

# 모든 백그라운드 작업이 끝날 때까지 대기
wait
echo "All scripts have finished executing."

# 입력 폴더와 출력 파일 설정
input_folder="collection"  # 예: './data'
output_file="$CURRENT_DATE/qas_rag_candidate.jsonl"

# 폴더 내에 있는 모든 qas_hard_rag로 시작하는 jsonl 파일을 찾습니다.
for jsonl_file in "$input_folder"/qas_rag_candidate*.jsonl
do
    if [ -f "$jsonl_file" ]; then
        cat "$jsonl_file" >> "$output_file"
    fi
done

echo "Merged JSON Lines files into $output_file"